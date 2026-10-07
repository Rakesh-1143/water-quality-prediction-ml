"""Data preparation, initialization, and evaluation without heavy ML imports."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix)
from config import FEATURES


def load_data(path):
    df = pd.read_csv(path)
    missing = set(FEATURES + ['Potability']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    X = df[FEATURES].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(df['Potability'], errors='raise')
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('Potability must contain both binary classes without missing values')
    if np.isinf(X.to_numpy()).any() or X.isna().all().any():
        raise ValueError('Features must have finite observations in every column')
    return X, y.astype(int)


def split_data(X, y, seed=42):
    """70/15/15 stratified split before fitting any data-dependent transform."""
    Xtr, Xrest, ytr, yrest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    Xv, Xt, yv, yt = train_test_split(
        Xrest, yrest, test_size=0.50, random_state=seed, stratify=yrest)
    return Xtr, Xv, Xt, ytr, yv, yt


def preprocessor():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', MinMaxScaler())])


def shap_importance(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # samples x features x outputs
        scores = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        scores = np.abs(values).mean(axis=0)
    else:
        raise ValueError('Expected SHAP samples x features [x outputs]')
    if scores.shape != (len(FEATURES),):
        raise ValueError('SHAP importance must cover the nine original features')
    if not np.isfinite(scores).all() or scores.sum() <= 0:
        raise ValueError('SHAP importance must be finite and have positive total')
    return scores, scores / scores.sum()


def input_kernel(weights, units=16):
    """Algorithm 1: repeat normalized feature weights across first-layer units.

    The paper specifies a vector, not a complete 9x16 matrix. This explicit
    deterministic expansion is documented rather than claimed as authors' code.
    """
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (9,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Expected nine finite nonnegative SHAP weights')
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError('SHAP weights must sum to one')
    return np.repeat(weights[:, None], units, axis=1)


def average(pnn, pxgb, alpha=0.5):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be within [0,1]')
    a, b = np.asarray(pnn), np.asarray(pxgb)
    if a.shape != b.shape:
        raise ValueError('Probability shapes must match')
    return alpha * a + (1 - alpha) * b


def metrics(y, probabilities):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(y)
    if y.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected one finite [0,1] probability per label')
    labels = (p >= 0.5).astype(int)
    return dict(accuracy=float(accuracy_score(y, labels)),
        precision=float(precision_score(y, labels, zero_division=0)),
        recall=float(recall_score(y, labels, zero_division=0)),
        f1=float(f1_score(y, labels, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)), r2=float(r2_score(y, p)),
        confusion_matrix=confusion_matrix(y, labels, labels=[0, 1]).tolist())


def dataset_profile(X, y):
    from scipy.stats import shapiro
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    outliers = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
    return dict(rows=len(X), missing=X.isna().sum().astype(int).to_dict(),
        classes=y.value_counts().sort_index().astype(int).to_dict(),
        iqr_outliers=outliers.astype(int).to_dict(),
        excluded_outliers=0,
        shapiro_p={c:float(shapiro(X[c].dropna()).pvalue) for c in FEATURES},
        descriptive=X.describe().to_dict())
