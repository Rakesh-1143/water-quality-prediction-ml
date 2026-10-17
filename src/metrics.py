"""Classification and error metrics (IEEE paper, Tables VI and VII)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)


def binary_metrics(y_true, p, threshold=0.5):
    """p = probability / score of the positive class (clipped to [0, 1])."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p),
        "rmse": float(np.sqrt(mean_squared_error(y_true, p))),
        "mae": mean_absolute_error(y_true, p),
    }


def roc_points(y_true, p, max_points=200):
    fpr, tpr, _ = roc_curve(y_true, np.clip(p, 0, 1))
    if len(fpr) > max_points:
        keep = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def multiclass_metrics(y_true, proba, n_classes):
    y_true = np.asarray(y_true)
    pred = np.asarray(proba).argmax(axis=1)
    labels = list(range(n_classes))
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", labels=labels),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
