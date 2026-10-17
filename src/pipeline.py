"""End-to-end training pipelines: binary potability (IEEE paper) and multiclass WQI (abstract)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVR
from xgboost import XGBClassifier

from config import FEATURES, RANDOM_STATE, RF_PARAMS, WQI_CLASSES, XGB_PARAMS
from src.artifacts import BinaryBundle, MulticlassBundle, WqiBinaryBundle
from src.data import load_dataset, prepare
from src.hybrid import fit_stacker, fuse_average, select_alpha, stack_predict
from src.metrics import binary_metrics, multiclass_metrics, roc_points
from src.nn_models import build_ann, predict_proba, train_ann
from src.shap_init import compute_shap_importance, normalize_importance
from src.wqi import classify_wqi, compute_wqi, quartile_cutoffs


def _xgb(seed):
    return XGBClassifier(**XGB_PARAMS, random_state=seed, eval_metric="logloss")


def fit_hybrid(splits, scaled, seed=RANDOM_STATE, epochs=None, n_classes=2):
    """Fit SHAP importance, SHAP-initialised ANN and XGBoost. Returns dict of parts."""
    importance = compute_shap_importance(scaled["train"], splits.y_train, random_state=seed)
    weights = normalize_importance(importance)
    ann = build_ann(len(FEATURES), n_classes, seed, shap_weights=weights.values)
    history = train_ann(ann, scaled["train"], splits.y_train, scaled["val"], splits.y_val, epochs=epochs)
    xgb = _xgb(seed).fit(scaled["train"], splits.y_train)
    return dict(importance=importance, weights=weights, ann=ann, history=history.history, xgb=xgb)


def _binary_probs(parts, scaled):
    out = {}
    for k in ("val", "test"):
        out[k] = (predict_proba(parts["ann"], scaled[k]), parts["xgb"].predict_proba(scaled[k])[:, 1])
    return out


def make_wqi_binary_labels(df):
    """1 = good (WQI at or below the median), 0 = poor. Returns (labels, cutoff).

    A formula of the 9 features, NOT the Kaggle potability label (their correlation is ~ -0.04).
    """
    imputed = df[FEATURES].fillna(df[FEATURES].median())
    wqi = compute_wqi(imputed)
    cutoff = float(wqi.median())
    return (wqi <= cutoff).astype(int), cutoff


def run_binary_pipeline(df=None, seed=RANDOM_STATE, epochs=None, baselines=True, task="potability"):
    """Train the proposed hybrid model and (optionally) the baselines/ablations of the paper.

    task="potability": Kaggle Potability label. task="wqi": WQI-derived good / poor label.
    """
    df = load_dataset() if df is None else df
    extra_meta, bundle_cls = {}, BinaryBundle
    if task == "wqi":
        labels, cutoff = make_wqi_binary_labels(df)
        df = df.assign(Potability=labels)
        extra_meta, bundle_cls = dict(wqi_cutoff=cutoff, task="wqi"), WqiBinaryBundle
    elif task != "potability":
        raise ValueError(f"Unknown task: {task}")
    splits, pre, scaled = prepare(df, seed)
    y_val, y_test = splits.y_val.values, splits.y_test.values

    parts = fit_hybrid(splits, scaled, seed, epochs)
    probs = _binary_probs(parts, scaled)
    alpha, alpha_scores = select_alpha(*probs["val"], y_val)
    stacker = fit_stacker(*probs["val"], y_val)

    p_nn, p_xgb = probs["test"]
    p_hybrid = fuse_average(p_nn, p_xgb, alpha)
    p_stack = stack_predict(stacker, p_nn, p_xgb)

    meta = dict(seed=seed, shap_importance=parts["importance"].to_dict(),
                shap_weights=parts["weights"].to_dict(), alpha_scores=alpha_scores,
                n_train=len(splits.y_train), n_val=len(y_val), n_test=len(y_test), **extra_meta)
    bundle = bundle_cls(pre, parts["ann"], parts["xgb"], alpha, meta, stacker)

    results = dict(
        alpha=alpha, alpha_val_auc=alpha_scores,
        shap_importance=parts["importance"].sort_values(ascending=False).to_dict(),
        shap_weights=parts["weights"].sort_values(ascending=False).to_dict(),
        hybrid_average=binary_metrics(y_test, p_hybrid),
        hybrid_stacking=binary_metrics(y_test, p_stack),
        loss_curve={k: [float(v) for v in parts["history"][k]] for k in ("loss", "val_loss")},
        roc={"Proposed Hybrid Model": roc_points(y_test, p_hybrid)},
        split_sizes=dict(train=len(splits.y_train), val=len(y_val), test=len(y_test)),
        positive_rate=float(np.mean(df["Potability"])), task=task,
        **{k: v for k, v in extra_meta.items() if k != "task"},
    )
    if baselines:
        results.update(_baselines(splits, scaled, parts, probs, p_hybrid, seed, epochs, y_val, y_test))
    return dict(bundle=bundle, results=results, splits=splits, scaled=scaled, parts=parts)


def _baselines(splits, scaled, parts, probs, p_hybrid, seed, epochs, y_val, y_test):
    """Table IX (comparison) and Table XIII (ablation) of the paper."""
    rf = RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=-1)
    rf.fit(scaled["train"], splits.y_train)
    svr = SVR(kernel="rbf").fit(scaled["train"], splits.y_train)
    ann_x = build_ann(len(FEATURES), 2, seed)
    train_ann(ann_x, scaled["train"], splits.y_train, scaled["val"], splits.y_val, epochs=epochs)

    p_rf = rf.predict_proba(scaled["test"])[:, 1]
    p_svr = np.clip(svr.predict(scaled["test"]), 0, 1)
    p_annx = predict_proba(ann_x, scaled["test"])
    p_annx_val = predict_proba(ann_x, scaled["val"])
    p_ann_s, p_xgb = probs["test"]
    alpha_x, _ = select_alpha(p_annx_val, probs["val"][1], y_val)
    p_annx_boost = fuse_average(p_annx, p_xgb, alpha_x)

    probs_by_model = {
        "Random Forest": p_rf, "SVR": p_svr, "ANN (Xavier Init)": p_annx,
        "XGBoost (Standalone)": p_xgb, "Proposed Hybrid Model": p_hybrid,
    }
    ablation_probs = {
        "ANN (Xavier Init)": p_annx, "ANN (SHAP Init)": p_ann_s,
        "ANN (Xavier Init + Boosting)": p_annx_boost, "Proposed Hybrid Model (Final)": p_hybrid,
    }
    comparison = {k: binary_metrics(y_test, v) for k, v in probs_by_model.items()}
    return dict(
        comparison=comparison,
        ablation={k: binary_metrics(y_test, v) for k, v in ablation_probs.items()},
        roc={k: roc_points(y_test, v) for k, v in probs_by_model.items()},
    )


def run_sensitivity(df=None, seeds=(0, 1, 2, 3, 4), epochs=None, task="potability"):
    """Paper Sec. IV-D: repeat random 70/15/15 splits and report mean +- std."""
    df = load_dataset() if df is None else df
    if task == "wqi":
        df = df.assign(Potability=make_wqi_binary_labels(df)[0])
    accs, f1s, aucs = [], [], []
    for s in seeds:
        splits, _, scaled = prepare(df, s)
        parts = fit_hybrid(splits, scaled, s, epochs)
        probs = _binary_probs(parts, scaled)
        alpha, _ = select_alpha(*probs["val"], splits.y_val.values)
        m = binary_metrics(splits.y_test.values, fuse_average(*probs["test"], alpha))
        accs.append(m["accuracy"]); f1s.append(m["f1"]); aucs.append(m["roc_auc"])
    stat = lambda v: dict(mean=float(np.mean(v)), std=float(np.std(v)), values=[float(x) for x in v])
    return dict(seeds=list(seeds), accuracy=stat(accs), f1=stat(f1s), roc_auc=stat(aucs))


def perturbation_test(bundle, X_test, y_test, top_features, fraction=0.10):
    """Paper Sec. IV-D: change the top features by -10% / +10% and measure the accuracy change."""
    base = accuracy_score(y_test, bundle.predict_proba(X_test) >= 0.5)
    imputed = bundle.pre.impute(X_test)
    out = {"baseline_accuracy": float(base), "fraction": fraction, "features": list(top_features)}
    for name, factor in (("decrease", 1 - fraction), ("increase", 1 + fraction)):
        changed = imputed.copy()
        changed[list(top_features)] = changed[list(top_features)] * factor
        acc = accuracy_score(y_test, bundle.predict_proba(changed) >= 0.5)
        out[name] = dict(accuracy=float(acc), drop=float(base - acc))
    return out


# --------------------------------------------------------------------- multiclass
def make_wqi_labels(df):
    """Derived WQI value and class (0..3) for every row (labels use full-data medians/quartiles)."""
    imputed = df[FEATURES].fillna(df[FEATURES].median())
    wqi = compute_wqi(imputed)
    cutoffs = quartile_cutoffs(wqi)
    return wqi, classify_wqi(wqi.values, cutoffs), cutoffs


def run_multiclass_pipeline(df=None, seed=RANDOM_STATE, epochs=None):
    """SHAP-initialised ANN + XGBoost for the four WQI levels (softmax / multi:softprob)."""
    import pandas as pd
    df = load_dataset() if df is None else df
    wqi, labels, cutoffs = make_wqi_labels(df)
    y = pd.Series(labels, index=df.index, name="wqi_class")
    splits, pre, scaled = prepare(df, seed, y=y)
    n_classes = len(WQI_CLASSES)

    parts = fit_hybrid(splits, scaled, seed, epochs, n_classes=n_classes)
    p_val = (predict_proba(parts["ann"], scaled["val"]), parts["xgb"].predict_proba(scaled["val"]))
    p_test = (predict_proba(parts["ann"], scaled["test"]), parts["xgb"].predict_proba(scaled["test"]))
    acc = lambda y_true, p: accuracy_score(y_true, p.argmax(axis=1))
    alpha, alpha_scores = select_alpha(*p_val, splits.y_val.values, score=acc)

    y_test = splits.y_test.values
    p_hybrid = fuse_average(*p_test, alpha)
    meta = dict(seed=seed, cutoffs=cutoffs, classes=WQI_CLASSES,
                shap_importance=parts["importance"].to_dict(), shap_weights=parts["weights"].to_dict(),
                alpha_scores=alpha_scores)
    bundle = MulticlassBundle(pre, parts["ann"], parts["xgb"], alpha, meta)
    results = dict(
        alpha=alpha, cutoffs=cutoffs, classes=WQI_CLASSES,
        class_counts={WQI_CLASSES[i]: int((labels == i).sum()) for i in range(n_classes)},
        hybrid=multiclass_metrics(y_test, p_hybrid, n_classes),
        ann_shap_only=multiclass_metrics(y_test, p_test[0], n_classes),
        xgb_only=multiclass_metrics(y_test, p_test[1], n_classes),
        shap_importance=parts["importance"].sort_values(ascending=False).to_dict(),
        wqi_summary=dict(min=float(wqi.min()), median=float(wqi.median()), max=float(wqi.max())),
    )
    return dict(bundle=bundle, results=results, splits=splits, scaled=scaled)
