"""Late-fusion of the SHAP-initialised ANN and XGBoost (IEEE paper, Eq. 6-7, Algorithm 2)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import ALPHA_GRID


def fuse_average(p_nn, p_xgb, alpha):
    """y = alpha * y_NN + (1 - alpha) * y_XGB. Works for 1-D and (n, k) probabilities."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * np.asarray(p_nn) + (1.0 - alpha) * np.asarray(p_xgb)


def select_alpha(p_nn_val, p_xgb_val, y_val, grid=ALPHA_GRID, score=roc_auc_score):
    """Pick alpha with the best validation score; ties go to the value closest to 0.5."""
    scored = [(score(y_val, fuse_average(p_nn_val, p_xgb_val, a)), a) for a in grid]
    best = max(s for s, _ in scored)
    candidates = [a for s, a in scored if np.isclose(s, best)]
    return float(min(candidates, key=lambda a: abs(a - 0.5))), {float(a): float(s) for s, a in scored}


def fit_stacker(p_nn_val, p_xgb_val, y_val):
    """Logistic-regression meta-learner on [p_NN, p_XGB], fitted on the validation set (Eq. 7)."""
    return LogisticRegression().fit(np.column_stack([p_nn_val, p_xgb_val]), y_val)


def stack_predict(stacker, p_nn, p_xgb):
    return stacker.predict_proba(np.column_stack([p_nn, p_xgb]))[:, 1]
