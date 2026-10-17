"""LSTM forecasting pipeline + artifact persistence."""
import json
import os

import joblib
import numpy as np
from tensorflow import keras

from config import (ARTIFACT_DIR, BRISBANE_INPUTS, FEATURES, LSTM_PARAMS, RANDOM_STATE, TS_WINDOWS)
from src.data import load_dataset
from src.metrics import regression_metrics
from src.realdata import load_brisbane
from src.timeseries import (build_lstm, inverse_wqi, predict_scaled_wqi, prepare_sequences,
                            simulate_series, to_delta, train_lstm)
from src.wqi import PROFILES, classify_wqi, quartile_cutoffs


def load_series(dataset="brisbane", freq="D"):
    """Return (series, features, profile_name, simulated) for a named dataset."""
    if dataset == "brisbane":
        return load_brisbane(freq=freq), BRISBANE_INPUTS, "brisbane", False
    if dataset == "simulated":
        return simulate_series(load_dataset()), FEATURES, "potability", True
    raise ValueError(f"Unknown dataset: {dataset}")


def run_lstm_pipeline(ts=None, features=None, profile_name="potability", simulated=False,
                      freq="D", seed=RANDOM_STATE, epochs=None, window=None, dataset=None):
    """Train the LSTM and compare it with the naive 'next = current' persistence baseline."""
    if ts is None:
        ts, features, profile_name, simulated = load_series(dataset or "simulated", freq)
    features = list(features or FEATURES)
    profile = PROFILES[profile_name]
    window = window or TS_WINDOWS.get(freq, LSTM_PARAMS["window"])

    seq = prepare_sequences(ts, window, profile=profile, features=features)
    n_feat = seq["train"][0].shape[2]
    model = build_lstm(window, n_feat, seed=seed)
    scaler, wi = seq["scaler"], seq["wqi_idx"]
    delta = lambda split: (seq[split][0], to_delta(*seq[split], wi))
    hist = train_lstm(model, delta("train"), delta("val"), epochs=epochs)

    X_test, y_test = seq["test"]
    true = inverse_wqi(scaler, wi, y_test)
    pred = inverse_wqi(scaler, wi, predict_scaled_wqi(model, X_test, wi))
    persistence = inverse_wqi(scaler, wi, X_test[:, -1, wi])

    # quality level of every step, using quartiles of the training-period WQI
    train_wqi = inverse_wqi(scaler, wi, seq["train"][1])
    cutoffs = quartile_cutoffs(train_wqi)
    level = lambda v: classify_wqi(v, cutoffs)
    results = dict(
        simulated=simulated, dataset=dataset or ("simulated" if simulated else "custom"),
        profile=profile_name, features=features, freq=freq, window=window, n_steps=len(ts),
        lstm=regression_metrics(true, pred), persistence=regression_metrics(true, persistence),
        level_accuracy=float(np.mean(level(pred) == level(true))),
        level_accuracy_persistence=float(np.mean(level(persistence) == level(true))),
        cutoffs=cutoffs, n_test=len(true),
        loss_curve={k: [float(v) for v in hist.history[k]] for k in ("loss", "val_loss")},
        test_dates=[str(d) for d in seq["dates_test"].iloc[:len(true)]]
        if len(seq["dates_test"]) >= len(true) else [],
        test_true=true.tolist(), test_pred=pred.tolist(), test_persistence=persistence.tolist(),
    )
    return dict(model=model, scaler=scaler, cols=seq["cols"], results=results, seq=seq)


def save_lstm(out, directory=ARTIFACT_DIR):
    os.makedirs(directory, exist_ok=True)
    out["model"].save(os.path.join(directory, "lstm.keras"))
    joblib.dump(dict(scaler=out["scaler"], cols=out["cols"]), os.path.join(directory, "lstm_scaler.pkl"))
    keys = ("window", "cutoffs", "simulated", "profile", "features", "freq", "dataset")
    with open(os.path.join(directory, "lstm_meta.json"), "w") as f:
        json.dump({k: out["results"][k] for k in keys}, f, indent=2)


def load_lstm(directory=ARTIFACT_DIR):
    with open(os.path.join(directory, "lstm_meta.json")) as f:
        meta = json.load(f)
    s = joblib.load(os.path.join(directory, "lstm_scaler.pkl"))
    return dict(model=keras.models.load_model(os.path.join(directory, "lstm.keras")),
                scaler=s["scaler"], cols=s["cols"], **meta)
