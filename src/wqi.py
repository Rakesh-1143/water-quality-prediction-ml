"""Weighted-arithmetic Water Quality Index (WQI) and multiclass WQI levels.

The Kaggle dataset has no WQI column, so WQI is derived from the 9 features:
    q_i = |V_i - V_ideal| / (S_i - V_ideal) * 100,   w_i = (1/S_i) / sum(1/S),   WQI = sum(w_i * q_i)
Lower WQI = better water. Because almost every sample exceeds the guideline limits,
fixed cut-offs (50/100/200/300) would put all samples in one class, so the four levels
are quartiles of the WQI distribution (relative quality levels).
"""
import numpy as np
import pandas as pd

from config import FEATURES, WQI_CLASSES

# Guideline limits S_i (WHO / EPA style). Limit for pH is the upper bound, ideal is 7.
STANDARDS = {
    "ph": 8.5, "Hardness": 300.0, "Solids": 1000.0, "Chloramines": 4.0,
    "Sulfate": 250.0, "Conductivity": 400.0, "Organic_carbon": 4.0,
    "Trihalomethanes": 80.0, "Turbidity": 5.0,
}
IDEAL = {f: 0.0 for f in FEATURES}
IDEAL["ph"] = 7.0

_inv = np.array([1.0 / STANDARDS[f] for f in FEATURES])
WEIGHTS = dict(zip(FEATURES, _inv / _inv.sum()))


class WQIProfile:
    """Weighted-arithmetic WQI for a set of parameters: limits S_i and ideal values V_i."""

    def __init__(self, name, standards, ideal):
        self.name = name
        self.standards = dict(standards)
        self.ideal = dict(ideal)
        self.features = list(self.standards)
        inv = np.array([1.0 / self.standards[f] for f in self.features])  # w_i proportional to 1/S_i
        self.weights = dict(zip(self.features, inv / inv.sum()))

    def compute(self, X):
        """WQI for a DataFrame of raw (imputed) parameters. Returns a float Series."""
        if X[self.features].isna().any().any():
            raise ValueError("compute_wqi needs imputed data (no NaN)")
        total = 0.0
        for f in self.features:
            q = (X[f] - self.ideal[f]).abs() / abs(self.standards[f] - self.ideal[f]) * 100.0
            total = total + self.weights[f] * q
        return total.astype(float)


POTABILITY = WQIProfile("potability", STANDARDS, IDEAL)

# Estuarine river buoy (Brisbane dataset). Limits are approximate ANZECC-style trigger values for
# south-east Australian estuaries (assumption, adjust if your guideline differs):
# pH <= 8.5 (ideal 7), dissolved oxygen >= 5 mg/L (ideal = saturation ~ 14.6 mg/L at 0 C scale),
# turbidity <= 10 NTU, chlorophyll-a <= 4 ug/L.
BRISBANE = WQIProfile(
    "brisbane",
    {"ph": 8.5, "dissolved_oxygen": 5.0, "turbidity": 10.0, "chlorophyll": 4.0},
    {"ph": 7.0, "dissolved_oxygen": 14.6, "turbidity": 0.0, "chlorophyll": 0.0},
)
PROFILES = {"potability": POTABILITY, "brisbane": BRISBANE}


def compute_wqi(X, profile=POTABILITY):
    """WQI for a DataFrame of raw (imputed) parameters. Returns a float Series."""
    return profile.compute(X)


def quartile_cutoffs(wqi):
    """Three cut-offs that split WQI into four equally sized levels."""
    return [float(c) for c in np.quantile(wqi, [0.25, 0.5, 0.75])]


def classify_wqi(wqi, cutoffs):
    """Map WQI value(s) to class index 0..3 (0 = Excellent). Accepts scalar or array."""
    idx = np.searchsorted(np.asarray(cutoffs), np.asarray(wqi), side="right")
    return idx if np.ndim(wqi) else int(idx)


def class_name(index):
    return WQI_CLASSES[int(index)]
