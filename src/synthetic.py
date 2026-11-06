"""Synthetic water-quality dataset generator (for demonstration only, NOT real measurements).

Features are sampled from distributions that match the mean / spread of the real Kaggle
dataset. The label is a documented rule plus label noise:

    risk = weighted sum of normalised guideline exceedances (pH distance from neutral,
           hardness, solids, chloramines, sulfate, conductivity, organic carbon,
           trihalomethanes, turbidity)
    clean label  = potable (1) for the lowest-risk `potable_rate` share, else 0
    final label  = clean label, flipped at random with probability `label_noise`

`label_noise` is the share of samples whose label is wrong (measurement / labelling error).
With symmetric noise no model can beat 1 - label_noise accuracy, so the noise level fixes the
ceiling: it is chosen up front and stated, never tuned to reach a target score.
"""
import json

import numpy as np
import pandas as pd

from config import FEATURES

# mean, std, lower bound, upper bound  (taken from the real dataset's summary statistics)
FEATURE_SPEC = {
    "ph": (7.08, 1.59, 0.0, 14.0),
    "Hardness": (196.4, 32.9, 47.0, 324.0),
    "Solids": (22014.0, 8768.0, 320.0, 61228.0),
    "Chloramines": (7.12, 1.58, 0.35, 13.2),
    "Sulfate": (333.8, 41.4, 129.0, 481.0),
    "Conductivity": (426.2, 80.8, 181.0, 754.0),
    "Organic_carbon": (14.28, 3.31, 2.2, 28.4),
    "Trihalomethanes": (66.4, 16.2, 0.7, 124.0),
    "Turbidity": (3.97, 0.78, 1.45, 6.74),
}
# guideline reference (value that counts as "1.0 unit of risk") and weight of each feature
RISK = {
    "ph": ("distance", 7.5, 1.0, 1.5), "Hardness": ("above", 150.0, 0.8, 80.0),
    "Solids": ("above", 15000.0, 1.2, 10000.0), "Chloramines": ("above", 4.0, 1.0, 3.0),
    "Sulfate": ("above", 250.0, 0.8, 100.0), "Conductivity": ("above", 400.0, 0.6, 150.0),
    "Organic_carbon": ("above", 10.0, 1.0, 5.0), "Trihalomethanes": ("above", 40.0, 1.0, 30.0),
    "Turbidity": ("above", 3.0, 0.8, 1.5),
}
DEFAULTS = dict(n=5000, seed=7, potable_rate=0.39, label_noise=0.10)


def sample_features(n, rng):
    data = {}
    for name in FEATURES:
        mean, std, lo, hi = FEATURE_SPEC[name]
        data[name] = np.clip(rng.normal(mean, std, n), lo, hi)
    return pd.DataFrame(data)


def risk_score(X):
    """Higher = worse water. Sum of weighted, scaled exceedances of guideline values."""
    total = np.zeros(len(X))
    for name, (kind, ref, weight, scale) in RISK.items():
        x = X[name].to_numpy()
        dev = np.abs(x - ref) if kind == "distance" else np.maximum(x - ref, 0.0)
        total += weight * dev / scale
    return total


def generate(n=DEFAULTS["n"], seed=DEFAULTS["seed"], potable_rate=DEFAULTS["potable_rate"],
             label_noise=DEFAULTS["label_noise"]):
    """Return (DataFrame with the 9 features + Potability, dict of generator settings)."""
    if not 0.0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5)")
    if not 0.0 < potable_rate < 1.0:
        raise ValueError("potable_rate must be in (0, 1)")
    rng = np.random.default_rng(seed)
    X = sample_features(n, rng)
    risk = risk_score(X)
    clean = (risk <= np.quantile(risk, potable_rate)).astype(int)
    flip = rng.random(n) < label_noise
    X["Potability"] = np.where(flip, 1 - clean, clean)
    meta = dict(n=n, seed=seed, potable_rate=potable_rate, label_noise=label_noise,
                measured_noise=float(flip.mean()), max_possible_accuracy=float(1 - flip.mean()),
                note="SYNTHETIC data for demonstration. Not real water measurements.")
    return X, meta


def save(df, meta, csv_path):
    df.to_csv(csv_path, index=False)
    with open(csv_path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
