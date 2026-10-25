"""Dashboard tool: performance."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import theme
from src.ui import require, results


def render():
    binary = require(results("binary"), "train_model.py")
    NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
             "roc_auc": "ROC-AUC", "rmse": "RMSE", "mae": "MAE"}


    def table(d):
        return pd.DataFrame(d).T.rename(columns=NAMES).round(3)


    st.caption("All numbers are test-set results (15% hold-out) from `model/train_model.py`.")
    t1, t2, t3, t4, t5 = st.tabs(["Comparison", "Ablation", "ROC & Loss", "SHAP", "Stability"])

    with t1:
        st.subheader("Proposed hybrid vs baselines")
        theme.leaderboard({k: v["accuracy"] for k, v in binary["comparison"].items()})
        st.dataframe(table(binary["comparison"]))
        st.write(f"Weighted-average α selected on validation set: **{binary['alpha']}**")
        st.subheader("Hybrid fusion: weighted average vs stacking")
        st.dataframe(table({"Weighted average": binary["hybrid_average"], "Stacking": binary["hybrid_stacking"]}))

    with t2:
        st.subheader("Effect of SHAP initialisation and boosting")
        st.dataframe(table(binary["ablation"]))

    with t3:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, pts in binary["roc"].items():
            ax.plot(pts["fpr"], pts["tpr"], label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(binary["loss_curve"]["loss"], label="Training loss")
        ax.plot(binary["loss_curve"]["val_loss"], label="Validation loss")
        ax.set_xlabel("Epoch"); ax.legend()
        st.pyplot(fig)

    with t4:
        imp = pd.DataFrame({"Mean |SHAP|": binary["shap_importance"], "Initial weight W_i": binary["shap_weights"]})
        st.dataframe(imp.round(4))
        st.bar_chart(imp["Mean |SHAP|"], color=theme.colors()[0])

    with t5:
        if "sensitivity" in binary:
            s = binary["sensitivity"]
            st.write(f"5 random 70/15/15 splits: accuracy **{s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f}**, "
                     f"F1 **{s['f1']['mean']:.3f} ± {s['f1']['std']:.3f}**, "
                     f"AUC **{s['roc_auc']['mean']:.3f} ± {s['roc_auc']['std']:.3f}**")
        if "perturbation" in binary:
            p = binary["perturbation"]
            st.write(f"±10% change of {', '.join(p['features'])}: accuracy drop "
                     f"{p['decrease']['drop']:.3f} (−10%), {p['increase']['drop']:.3f} (+10%) "
                     f"from baseline {p['baseline_accuracy']:.3f}")

    wb = results("wqibin")
    if wb:
        st.divider()
        st.subheader("WQI-based good / poor task")
        st.caption("Label = WQI at or below the median. It is a formula of the 9 inputs and is unrelated to the "
                   "Kaggle potability label (correlation about -0.04), so these accuracies are not comparable "
                   "to the potability results above.")
        st.dataframe(table(wb["comparison"]))
        st.dataframe(table(wb["ablation"]))

    wqi = results("wqi")
    if wqi:
        st.divider()
        st.subheader("Multiclass WQI")
        st.dataframe(pd.DataFrame({k: {"Accuracy": wqi[k]["accuracy"], "Macro F1": wqi[k]["f1_macro"],
                                       "AUC (OvR)": wqi[k]["roc_auc_ovr"]}
                                   for k in ("ann_shap_only", "xgb_only", "hybrid")}).T.round(3))
        st.write("Confusion matrix (hybrid, rows = actual):")
        st.dataframe(pd.DataFrame(wqi["hybrid"]["confusion_matrix"], index=wqi["classes"], columns=wqi["classes"]))
        st.caption("WQI levels are derived from the same 9 features, so high accuracy is expected here; "
                   "this is not comparable to the potability task.")
