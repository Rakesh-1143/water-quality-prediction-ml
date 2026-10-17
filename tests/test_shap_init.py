import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from src.nn_models import build_ann, predict_proba
from src.shap_init import (compute_shap_importance, normalize_importance, scale_first_layer,
                           shap_values_array)


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(FEATURES))).astype("float32")
    y = (X[:, 0] + 0.1 * rng.random(n) > 0.55).astype(int)  # only feature 0 matters
    return X, y


def test_importance_finds_the_informative_feature():
    X, y = synthetic()
    imp = compute_shap_importance(X, y)
    assert list(imp.index) == FEATURES
    assert (imp >= 0).all()
    assert imp.idxmax() == FEATURES[0]


def test_normalize_sums_to_one_and_keeps_order():
    w = normalize_importance(pd.Series([0.3, 0.1, 0.6]))
    assert w.sum() == pytest.approx(1.0)
    assert list(w.round(2)) == [0.3, 0.1, 0.6]
    with pytest.raises(ValueError):
        normalize_importance(pd.Series([0.0, 0.0]))


def test_paper_table_iv_values_normalise_to_one():
    # Paper Table V lists weights that sum to 1.079, which contradicts its own Eq. 3; the
    # correct Eq. 3 normalisation of the Table IV scores gives 0.140 for Sulfate (not 0.151).
    shap = [0.1378, 0.1293, 0.1187, 0.1139, 0.1076, 0.1032, 0.0974, 0.0931, 0.0819]
    w = normalize_importance(pd.Series(shap))
    assert w.sum() == pytest.approx(1.0)
    assert w.iloc[0] == pytest.approx(0.1402, abs=0.0005)
    assert w.is_monotonic_decreasing


def test_shap_values_array_shapes():
    assert shap_values_array(np.zeros((4, 3))).shape == (4, 3, 1)
    assert shap_values_array(np.zeros((4, 3, 2))).shape == (4, 3, 2)
    assert shap_values_array([np.zeros((4, 3)), np.zeros((4, 3))]).shape == (4, 3, 2)


def test_scale_first_layer_scales_rows_by_n_times_weight():
    kernel = np.ones((3, 5))
    out = scale_first_layer(kernel, [0.5, 0.25, 0.25])
    assert out[:, 0] == pytest.approx([1.5, 0.75, 0.75])
    with pytest.raises(ValueError):
        scale_first_layer(kernel, [0.5, 0.5])


def test_shap_init_changes_only_first_layer_rows_and_keeps_architecture():
    w = normalize_importance(pd.Series(np.arange(1, 10, dtype=float), index=FEATURES)).values
    xavier = build_ann(9, 2, seed=1)
    shapnn = build_ann(9, 2, seed=1, shap_weights=w)
    assert [l.units for l in shapnn.layers if hasattr(l, "units")] == [16, 8, 1]
    k_x, k_s = xavier.layers[0].get_weights()[0], shapnn.layers[0].get_weights()[0]
    assert k_s / k_x == pytest.approx(np.tile((w * 9)[:, None], (1, 16)), rel=1e-4)
    for a, b in zip(xavier.get_weights()[2:], shapnn.get_weights()[2:]):
        assert np.allclose(a, b)  # later layers identical -> clean ablation


def test_multiclass_ann_outputs_probabilities():
    model = build_ann(9, 4, seed=0)
    p = predict_proba(model, np.random.default_rng(0).random((5, 9)).astype("float32"))
    assert p.shape == (5, 4) and p.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
