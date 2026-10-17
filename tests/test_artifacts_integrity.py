"""The committed model files must reproduce the numbers stored in results_*.json."""
import numpy as np
import pytest

from config import FEATURES, RANDOM_STATE
from src.artifacts import BinaryBundle, MulticlassBundle, WqiBinaryBundle, load_results
from src.data import split_data
from src.metrics import binary_metrics
from src.pipeline import make_wqi_binary_labels, make_wqi_labels
from src.pipeline_ts import load_lstm
from src.timeseries import forecast_next

pytestmark = pytest.mark.skipif(not BinaryBundle.exists(), reason="trained files missing")


def test_binary_files_reproduce_reported_test_metrics(df):
    bundle, res = BinaryBundle.load(), load_results("binary")
    s = split_data(df[FEATURES], df["Potability"], seed=RANDOM_STATE)
    m = binary_metrics(s.y_test.values, bundle.predict_proba(s.X_test))
    for k, v in res["hybrid_average"].items():
        assert m[k] == pytest.approx(v, abs=1e-4), k
    assert bundle.alpha == res["alpha"]


def test_binary_stacking_reproduces_reported_metrics(df):
    bundle, res = BinaryBundle.load(), load_results("binary")
    s = split_data(df[FEATURES], df["Potability"], seed=RANDOM_STATE)
    m = binary_metrics(s.y_test.values, bundle.predict_proba(s.X_test, "stacking"))
    assert m["accuracy"] == pytest.approx(res["hybrid_stacking"]["accuracy"], abs=1e-4)


def test_wqi_binary_files_reproduce_reported_test_metrics(df):
    bundle, res = WqiBinaryBundle.load(), load_results("wqibin")
    labels, cutoff = make_wqi_binary_labels(df)
    s = split_data(df[FEATURES], labels, seed=RANDOM_STATE)
    m = binary_metrics(s.y_test.values, bundle.predict_proba(s.X_test))
    for k, v in res["hybrid_average"].items():
        assert m[k] == pytest.approx(v, abs=1e-4), k
    assert bundle.wqi_cutoff == pytest.approx(cutoff)


def test_multiclass_files_reproduce_reported_accuracy(df):
    import pandas as pd
    bundle, res = MulticlassBundle.load(), load_results("wqi")
    _, labels, _ = make_wqi_labels(df)
    s = split_data(df[FEATURES], pd.Series(labels, index=df.index), seed=RANDOM_STATE)
    acc = (bundle.predict_proba(s.X_test).argmax(axis=1) == s.y_test.values).mean()
    assert acc == pytest.approx(res["hybrid"]["accuracy"], abs=1e-4)


def test_shap_weights_in_meta_match_paper_equation_3():
    for cls in (BinaryBundle, MulticlassBundle):
        w = cls.load().meta["shap_weights"]
        assert sum(w.values()) == pytest.approx(1.0) and set(w) == set(FEATURES)


def test_lstm_files_reproduce_reported_metrics():
    """Re-run inference on the real Brisbane data with the saved LSTM and compare to results_lstm.json."""
    from src.metrics import regression_metrics
    from src.realdata import load_brisbane
    from src.timeseries import inverse_wqi, predict_scaled_wqi, prepare_sequences
    from src.wqi import PROFILES
    b, res = load_lstm(), load_results("lstm")
    assert b["simulated"] is False and res["simulated"] is False and b["dataset"] == "brisbane"
    ts = load_brisbane(freq=b["freq"])
    seq = prepare_sequences(ts, b["window"], profile=PROFILES[b["profile"]], features=b["features"])
    X, y = seq["test"]
    wi = seq["wqi_idx"]
    pred = inverse_wqi(b["scaler"], wi, predict_scaled_wqi(b["model"], X, wi))
    true = inverse_wqi(b["scaler"], wi, y)
    m = regression_metrics(true, pred)
    assert m["rmse"] == pytest.approx(res["lstm"]["rmse"], abs=1e-3)
    assert m["r2"] == pytest.approx(res["lstm"]["r2"], abs=1e-3)
    assert len(true) == res["n_test"]


def test_lstm_files_forecast_from_real_window():
    from src.realdata import load_brisbane
    b = load_lstm()
    ts = load_brisbane(freq=b["freq"])
    value = forecast_next(b["model"], b["scaler"], b["cols"], ts[b["features"]].tail(b["window"]),
                          b["window"], __import__("src.wqi", fromlist=["PROFILES"]).PROFILES[b["profile"]])
    assert np.isfinite(value) and 30 < value < 120
