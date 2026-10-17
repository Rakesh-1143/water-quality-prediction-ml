import json
import os
import subprocess
import sys

import numpy as np
import pytest

from config import FEATURES
from src.artifacts import BinaryBundle, MulticlassBundle, load_results, save_results
from src.pipeline import perturbation_test, run_sensitivity

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae"}


def test_binary_results_structure_and_bounds(binary_run):
    r = binary_run["results"]
    assert set(r["comparison"]) == {"Random Forest", "SVR", "ANN (Xavier Init)",
                                    "XGBoost (Standalone)", "Proposed Hybrid Model"}
    assert set(r["ablation"]) == {"ANN (Xavier Init)", "ANN (SHAP Init)",
                                  "ANN (Xavier Init + Boosting)", "Proposed Hybrid Model (Final)"}
    for table in (r["comparison"], r["ablation"], {"avg": r["hybrid_average"], "stack": r["hybrid_stacking"]}):
        for m in table.values():
            assert set(m) == METRICS
            assert all(0.0 <= v <= 1.0 for v in m.values())
    assert r["alpha"] in (0.3, 0.4, 0.5, 0.6, 0.7)
    assert sum(r["shap_weights"].values()) == pytest.approx(1.0)
    assert r["split_sizes"]["train"] + r["split_sizes"]["val"] + r["split_sizes"]["test"] == 3276
    assert len(r["loss_curve"]["loss"]) == len(r["loss_curve"]["val_loss"]) >= 1
    assert set(r["roc"]) == set(r["comparison"])


def test_hybrid_equals_weighted_average_of_components(binary_run):
    bundle, splits = binary_run["bundle"], binary_run["splits"]
    X = splits.X_test.head(20)
    p_nn, p_xgb = bundle.components(X)
    assert bundle.predict_proba(X) == pytest.approx(bundle.alpha * p_nn + (1 - bundle.alpha) * p_xgb)
    stack = bundle.predict_proba(X, mode="stacking")
    assert stack.shape == (20,) and ((stack >= 0) & (stack <= 1)).all()


def test_bundle_handles_missing_values_and_extreme_inputs(binary_run):
    bundle = binary_run["bundle"]
    X = binary_run["splits"].X_test.head(3).copy()
    X.iloc[0, 0] = np.nan          # missing pH
    X.iloc[1, 2] = 1e9             # absurd solids -> clipped, must not crash
    p = bundle.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_bundle_save_load_roundtrip(binary_run, tmp_path):
    bundle = binary_run["bundle"]
    assert not BinaryBundle.exists(str(tmp_path))
    bundle.save(str(tmp_path))
    assert BinaryBundle.exists(str(tmp_path))
    loaded = BinaryBundle.load(str(tmp_path))
    X = binary_run["splits"].X_test
    assert loaded.alpha == bundle.alpha
    assert loaded.predict_proba(X) == pytest.approx(bundle.predict_proba(X), abs=1e-5)
    assert loaded.predict_proba(X, "stacking") == pytest.approx(bundle.predict_proba(X, "stacking"), abs=1e-5)


def test_shap_explanation_for_single_sample(binary_run):
    bundle = binary_run["bundle"]
    X = binary_run["splits"].X_test.head(1)
    values = bundle.explain(X)
    assert values.shape == (1, len(FEATURES), 1)
    # SHAP additivity: base value + sum(shap) == XGBoost margin
    margin = bundle.xgb.predict(bundle.pre.transform(X), output_margin=True)[0]
    assert bundle.expected_value[0] + values[0, :, 0].sum() == pytest.approx(margin, abs=1e-3)


def test_perturbation_test(binary_run):
    s = binary_run["splits"]
    out = perturbation_test(binary_run["bundle"], s.X_test, s.y_test.values, FEATURES[:3])
    assert {"baseline_accuracy", "decrease", "increase"} <= set(out)
    assert 0 <= out["decrease"]["accuracy"] <= 1


def test_sensitivity_study_runs(df):
    out = run_sensitivity(df, seeds=(0, 1), epochs=2)
    assert len(out["accuracy"]["values"]) == 2 and out["accuracy"]["std"] >= 0
    assert 0 < out["accuracy"]["mean"] < 1


def test_multiclass_results_and_bundle(multiclass_run, tmp_path):
    r, bundle, splits = multiclass_run["results"], multiclass_run["bundle"], multiclass_run["splits"]
    assert r["classes"] == ["Excellent", "Good", "Poor", "Very Poor"]
    assert sum(r["class_counts"].values()) == 3276
    assert r["hybrid"]["accuracy"] > 0.5  # 4 balanced classes -> chance is 0.25
    X = splits.X_test.head(10)
    p = bundle.predict_proba(X)
    assert p.shape == (10, 4) and p.sum(axis=1) == pytest.approx(1.0, abs=1e-4)
    assert bundle.explain(X.head(1)).shape == (1, len(FEATURES), 4)
    bundle.save(str(tmp_path))
    loaded = MulticlassBundle.load(str(tmp_path))
    assert loaded.cutoffs == bundle.cutoffs
    assert loaded.predict_proba(X) == pytest.approx(p, abs=1e-5)
    assert np.array(r["hybrid"]["confusion_matrix"]).sum() == len(splits.y_test)


def test_results_json_roundtrip(tmp_path):
    save_results("x", {"a": np.float32(0.5), "b": [1, 2]}, str(tmp_path))
    assert load_results("x", str(tmp_path)) == {"a": 0.5, "b": [1, 2]}
    assert load_results("missing", str(tmp_path)) is None


@pytest.mark.parametrize("script,extra,expected", [
    ("train_model.py", ["--skip-sensitivity"], ["binary_nn.keras", "results_binary.json"]),
    ("train_multiclass.py", [], ["wqi_nn.keras", "results_wqi.json"]),
    ("train_lstm.py", [], ["lstm.keras", "results_lstm.json"]),
])
def test_training_scripts_run_end_to_end(script, extra, expected, tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "model", script), "--epochs", "2",
           "--out-dir", str(tmp_path)] + extra
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path),
                          env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3"))
    assert done.returncode == 0, done.stderr[-2000:]
    for name in expected:
        assert (tmp_path / name).exists(), name
    json.load(open(tmp_path / expected[-1]))
