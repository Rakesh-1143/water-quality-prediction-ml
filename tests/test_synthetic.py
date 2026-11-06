import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from src.synthetic import FEATURE_SPEC, generate, risk_score, save

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_shape_columns_ranges_and_no_missing_values():
    df, meta = generate(n=2000, seed=1)
    assert list(df.columns) == FEATURES + ["Potability"] and len(df) == 2000
    assert not df.isna().any().any() and set(df["Potability"]) == {0, 1}
    for name, (_, _, lo, hi) in FEATURE_SPEC.items():
        assert df[name].between(lo, hi).all()
    assert "SYNTHETIC" in meta["note"]


def test_feature_statistics_match_the_real_dataset(df):
    syn, _ = generate(n=5000, seed=2)
    for name in FEATURES:
        assert syn[name].mean() == pytest.approx(df[name].mean(), rel=0.05)
        assert syn[name].std() == pytest.approx(df[name].std(), rel=0.1)


def test_generation_is_reproducible_and_seed_dependent():
    a, _ = generate(n=300, seed=5)
    b, _ = generate(n=300, seed=5)
    c, _ = generate(n=300, seed=6)
    assert a.equals(b) and not a.equals(c)


def test_label_noise_fixes_the_accuracy_ceiling():
    clean, m0 = generate(n=4000, seed=3, label_noise=0.0)
    noisy, m1 = generate(n=4000, seed=3, label_noise=0.15)
    risk = risk_score(clean[FEATURES])
    rule = (risk <= np.quantile(risk, 0.39)).astype(int)
    assert (clean["Potability"] == rule).all() and m0["max_possible_accuracy"] == 1.0
    disagree = (noisy["Potability"] != rule).mean()                      # the rule is the best possible predictor
    assert disagree == pytest.approx(0.15, abs=0.02) and m1["max_possible_accuracy"] == pytest.approx(1 - disagree, abs=0.01)


def test_label_depends_on_the_features_in_the_documented_direction():
    df, _ = generate(n=4000, seed=4, label_noise=0.0)
    assert df.loc[df["Potability"] == 1, "Organic_carbon"].mean() < df.loc[df["Potability"] == 0, "Organic_carbon"].mean()
    assert df.loc[df["Potability"] == 1, "Turbidity"].mean() < df.loc[df["Potability"] == 0, "Turbidity"].mean()
    assert abs(df.loc[df["Potability"] == 1, "ph"].mean() - 7.5) < abs(df.loc[df["Potability"] == 0, "ph"].mean() - 7.5)
    assert df["Potability"].mean() == pytest.approx(0.39, abs=0.01)


@pytest.mark.parametrize("kwargs", [dict(label_noise=0.5), dict(label_noise=-0.1), dict(potable_rate=0.0), dict(potable_rate=1.0)])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        generate(n=100, **kwargs)


def test_save_writes_csv_and_metadata(tmp_path):
    df, meta = generate(n=50, seed=1)
    path = str(tmp_path / "s.csv")
    save(df, meta, path)
    assert pd.read_csv(path).shape == (50, 10)
    assert json.load(open(str(tmp_path / "s.meta.json")))["seed"] == 1


def test_cli_and_training_scripts_accept_a_data_file(tmp_path):
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    csv = tmp_path / "syn.csv"
    done = subprocess.run([sys.executable, os.path.join(ROOT, "model", "make_synthetic_dataset.py"),
                           "--n", "600", "--out", str(csv)], capture_output=True, text=True, env=env)
    assert done.returncode == 0 and "maximum possible accuracy" in done.stdout
    for script, extra, expected in (("train_model.py", ["--skip-sensitivity"], "results_binary.json"),
                                    ("train_multiclass.py", [], "results_wqi.json")):
        out = tmp_path / script
        done = subprocess.run([sys.executable, os.path.join(ROOT, "model", script), "--data", str(csv),
                               "--epochs", "2", "--out-dir", str(out)] + extra,
                              capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert done.returncode == 0, done.stderr[-1500:]
        assert (out / expected).exists()
