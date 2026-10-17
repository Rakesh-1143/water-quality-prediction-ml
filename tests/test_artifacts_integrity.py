"""The committed model files must reproduce the numbers stored in results_*.json."""
import numpy as np
import pytest

from config import FEATURES, RANDOM_STATE
from src.artifacts import BinaryBundle, MulticlassBundle, load_results
from src.data import split_data
from src.metrics import binary_metrics
from src.pipeline import make_wqi_labels
from src.pipeline_ts import load_lstm
from src.timeseries import forecast_next, simulate_series

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


def test_lstm_files_forecast(df):
    b = load_lstm()
    value = forecast_next(b["model"], b["scaler"], b["cols"], simulate_series(df, 40)[FEATURES], b["window"])
    assert np.isfinite(value)
