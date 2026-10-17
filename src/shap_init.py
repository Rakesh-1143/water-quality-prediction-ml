"""SHAP feature importance and feature-guided weight initialisation (IEEE paper, Sec. III-C/D)."""
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from config import FEATURES, RANDOM_STATE


def shap_values_array(shap_values):
    """Return SHAP values as (n_samples, n_features, n_outputs) for any shap/xgboost output shape."""
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, None]
    return shap_values


def compute_shap_importance(X_scaled, y, features=FEATURES, random_state=RANDOM_STATE):
    """Train a default XGBoost model and return mean |SHAP| per feature (Series).

    Works for binary and multiclass targets (multiclass: averaged over classes).
    """
    model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    model.fit(X_scaled, y)
    values = shap_values_array(shap.TreeExplainer(model).shap_values(X_scaled))
    return pd.Series(np.abs(values).mean(axis=(0, 2)), index=list(features), name="shap_importance")


def normalize_importance(importance):
    """W_i = FI_i / sum(FI)  (paper Eq. 3). Result sums to 1."""
    importance = pd.Series(importance, dtype=float)
    total = importance.sum()
    if total <= 0:
        raise ValueError("SHAP importances sum to zero")
    return importance / total


def scale_first_layer(kernel, weights):
    """Scale row i (input feature i) of a Glorot-initialised kernel by n * W_i.

    Multiplying by n keeps the average scale at 1, so only the *relative* importance
    changes: important features start with larger weights. Using a different random
    kernel per hidden neuron (instead of copying W_i) avoids symmetric neurons.
    """
    weights = np.asarray(weights, dtype=float)
    if kernel.shape[0] != len(weights):
        raise ValueError("kernel rows must match number of features")
    return kernel * (weights * len(weights))[:, None]
