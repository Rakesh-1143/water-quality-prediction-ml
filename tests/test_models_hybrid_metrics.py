import numpy as np
import pytest

from src.hybrid import fit_stacker, fuse_average, select_alpha, stack_predict
from src.metrics import binary_metrics, multiclass_metrics, regression_metrics, roc_points
from src.nn_models import build_ann, predict_proba, train_ann
from tests.test_shap_init import synthetic


def test_ann_learns_synthetic_signal_and_is_reproducible():
    X, y = synthetic()
    runs = []
    for _ in range(2):
        m = build_ann(9, 2, seed=3)
        h = train_ann(m, X[:400], y[:400], X[400:], y[400:], epochs=25)
        runs.append(predict_proba(m, X[400:]))
    assert h.history["loss"][-1] < h.history["loss"][0]
    assert runs[0] == pytest.approx(runs[1], abs=1e-5)
    p = runs[0]
    assert p.ndim == 1 and p.min() >= 0 and p.max() <= 1
    assert ((p >= 0.5) == y[400:]).mean() > 0.8


def test_early_stopping_stops_before_max_epochs():
    X, y = synthetic(300)
    m = build_ann(9, 2, seed=0)
    h = train_ann(m, X[:200], y[:200], X[200:], y[200:], epochs=300,
                  params=dict(batch_size=32, patience=3))
    assert len(h.history["loss"]) < 300


def test_fuse_average_formula():
    assert fuse_average(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.7) == pytest.approx([0.7, 0.0])
    assert fuse_average(np.ones((2, 3)), np.zeros((2, 3)), 0.5) == pytest.approx(np.full((2, 3), 0.5))
    with pytest.raises(ValueError):
        fuse_average(0.5, 0.5, 1.5)


def test_select_alpha_prefers_better_model_and_breaks_ties_at_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y * 0.8 + 0.1
    noise = rng.random(300)
    alpha, scores = select_alpha(good, noise, y)
    assert alpha > 0.5 and set(scores) == {0.3, 0.4, 0.5, 0.6, 0.7}
    assert scores[0.7] >= scores[0.3]
    alpha_tie, _ = select_alpha(good, good, y)
    assert alpha_tie == 0.5


def test_stacker_combines_inputs_into_probability():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p1, p2 = y * 0.6 + rng.random(200) * 0.4, rng.random(200)
    s = fit_stacker(p1, p2, y)
    out = stack_predict(s, p1, p2)
    assert out.shape == (200,) and out.min() >= 0 and out.max() <= 1
    assert s.coef_[0][0] > abs(s.coef_[0][1])  # relies on the informative model


def test_binary_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])  # predictions 0,1,0,1
    m = binary_metrics(y, p)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((p - y) ** 2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(p - y)))


def test_metrics_clip_out_of_range_scores():
    m = binary_metrics([0, 1], [-0.5, 1.7])  # e.g. raw SVR outputs
    assert m["accuracy"] == 1.0 and m["rmse"] == 0.0


def test_roc_points_endpoints():
    r = roc_points([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6])
    assert r["fpr"][0] == 0 and r["tpr"][-1] == 1


def test_multiclass_and_regression_metrics():
    y = np.array([0, 1, 2, 2])
    proba = np.eye(3)[[0, 1, 2, 1]] * 0.8 + 0.1
    m = multiclass_metrics(y, proba / proba.sum(axis=1, keepdims=True), 3)
    assert m["accuracy"] == 0.75 and np.array(m["confusion_matrix"]).sum() == 4
    r = regression_metrics([1, 2, 3], [1, 2, 4])
    assert r["mae"] == pytest.approx(1 / 3) and r["r2"] < 1
