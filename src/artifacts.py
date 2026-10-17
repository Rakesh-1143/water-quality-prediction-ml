"""Save / load trained models so the Streamlit app and tests use exactly what was trained."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from tensorflow import keras
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, FEATURES, WQI_CLASSES
from src.hybrid import fuse_average, stack_predict
from src.nn_models import predict_proba
from src.shap_init import shap_values_array


def _paths(directory, prefix):
    j = lambda name: os.path.join(directory, f"{prefix}{name}")
    return dict(pre=j("preprocessor.pkl"), nn=j("nn.keras"), xgb=j("xgb.json"),
                stacker=j("stacker.pkl"), meta=j("meta.json"))


class _Bundle:
    prefix = ""
    n_classes = 2

    def __init__(self, pre, nn, xgb, alpha, meta, stacker=None):
        self.pre, self.nn, self.xgb = pre, nn, xgb
        self.alpha, self.meta, self.stacker = alpha, meta, stacker
        self._explainer = None

    # ---- inference -------------------------------------------------
    def components(self, X_raw):
        """Return (p_nn, p_xgb) for a DataFrame of raw features."""
        Xs = self.pre.transform(X_raw)
        p_nn = predict_proba(self.nn, Xs)
        p_xgb = self.xgb.predict_proba(Xs)
        return p_nn, (p_xgb[:, 1] if self.n_classes == 2 else p_xgb)

    def predict_proba(self, X_raw, mode="average"):
        p_nn, p_xgb = self.components(X_raw)
        if mode == "stacking":
            if self.stacker is None:
                raise ValueError("No stacking model saved")
            return stack_predict(self.stacker, p_nn, p_xgb)
        return fuse_average(p_nn, p_xgb, self.alpha)

    def explain(self, X_raw):
        """SHAP values of the XGBoost component: (n_samples, n_features, n_outputs)."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.xgb)
        return shap_values_array(self._explainer.shap_values(self.pre.transform(X_raw)))

    @property
    def expected_value(self):
        self.explain(pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES))
        return np.atleast_1d(self._explainer.expected_value)

    # ---- persistence -----------------------------------------------
    def save(self, directory=ARTIFACT_DIR):
        os.makedirs(directory, exist_ok=True)
        p = _paths(directory, self.prefix)
        joblib.dump(self.pre, p["pre"])
        self.nn.save(p["nn"])
        self.xgb.save_model(p["xgb"])
        if self.stacker is not None:
            joblib.dump(self.stacker, p["stacker"])
        meta = dict(self.meta, alpha=self.alpha, features=FEATURES)
        with open(p["meta"], "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory=ARTIFACT_DIR):
        p = _paths(directory, cls.prefix)
        with open(p["meta"]) as f:
            meta = json.load(f)
        xgb = XGBClassifier()
        xgb.load_model(p["xgb"])
        stacker = joblib.load(p["stacker"]) if os.path.exists(p["stacker"]) else None
        return cls(joblib.load(p["pre"]), keras.models.load_model(p["nn"]), xgb,
                   meta["alpha"], meta, stacker)

    @classmethod
    def exists(cls, directory=ARTIFACT_DIR):
        return all(os.path.exists(_paths(directory, cls.prefix)[k]) for k in ("pre", "nn", "xgb", "meta"))


class BinaryBundle(_Bundle):
    """SHAP-initialised ANN + XGBoost for potable / not potable."""
    prefix = "binary_"
    n_classes = 2


class MulticlassBundle(_Bundle):
    """SHAP-initialised ANN + XGBoost for the 4 WQI levels."""
    prefix = "wqi_"
    n_classes = len(WQI_CLASSES)

    @property
    def cutoffs(self):
        return self.meta["cutoffs"]


def save_results(name, results, directory=ARTIFACT_DIR):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"results_{name}.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


def load_results(name, directory=ARTIFACT_DIR):
    path = os.path.join(directory, f"results_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
