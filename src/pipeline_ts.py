"""LSTM forecasting pipeline + artifact persistence."""
import json
import os

import joblib
import numpy as np
from tensorflow import keras

from config import ARTIFACT_DIR, LSTM_PARAMS, RANDOM_STATE
from src.data import load_dataset
from src.metrics import regression_metrics
from src.timeseries import (build_lstm, inverse_wqi, load_time_series, predict_scaled_wqi,
                            prepare_sequences, to_delta, train_lstm)
from src.wqi import classify_wqi, quartile_cutoffs


def run_lstm_pipeline(ts=None, df=None, seed=RANDOM_STATE, epochs=None, window=LSTM_PARAMS["window"]):
    """Train the LSTM and compare it with the naive 'tomorrow = today' persistence baseline."""
    if ts is None:
        df = load_dataset() if df is None else df
        ts, simulated = load_time_series(df=df, seed=seed)
    else:
        simulated = False
    seq = prepare_sequences(ts, window)
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
    level_acc = float(np.mean(classify_wqi(pred, cutoffs) == classify_wqi(true, cutoffs)))
    level_acc_base = float(np.mean(classify_wqi(persistence, cutoffs) == classify_wqi(true, cutoffs)))

    results = dict(
        simulated=simulated, window=window, n_steps=len(ts),
        lstm=regression_metrics(true, pred), persistence=regression_metrics(true, persistence),
        level_accuracy=level_acc, level_accuracy_persistence=level_acc_base, cutoffs=cutoffs,
        loss_curve={k: [float(v) for v in hist.history[k]] for k in ("loss", "val_loss")},
        test_dates=[str(d.date()) for d in seq["dates_test"].iloc[:len(true)]] if len(seq["dates_test"]) >= len(true) else [],
        test_true=true.tolist(), test_pred=pred.tolist(),
    )
    return dict(model=model, scaler=scaler, cols=seq["cols"], results=results, seq=seq)


def save_lstm(out, directory=ARTIFACT_DIR):
    os.makedirs(directory, exist_ok=True)
    out["model"].save(os.path.join(directory, "lstm.keras"))
    joblib.dump(dict(scaler=out["scaler"], cols=out["cols"]), os.path.join(directory, "lstm_scaler.pkl"))
    meta = {k: out["results"][k] for k in ("window", "cutoffs", "simulated")}
    with open(os.path.join(directory, "lstm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_lstm(directory=ARTIFACT_DIR):
    with open(os.path.join(directory, "lstm_meta.json")) as f:
        meta = json.load(f)
    s = joblib.load(os.path.join(directory, "lstm_scaler.pkl"))
    return dict(model=keras.models.load_model(os.path.join(directory, "lstm.keras")),
                scaler=s["scaler"], cols=s["cols"], **meta)
