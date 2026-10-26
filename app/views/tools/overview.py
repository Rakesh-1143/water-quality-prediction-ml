"""Dashboard tool: overview with the main results at a glance."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import theme
from src.timeseries import break_at_gaps
from src.ui import results


def render():
    binary, lstm = results("binary"), results("lstm")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Model comparison (accuracy)")
        if binary:
            theme.leaderboard({k: v["accuracy"] for k, v in binary["comparison"].items()})
            st.caption("Kaggle potability label, test set. Bars start at zero; the best model is highlighted.")
    with right:
        st.subheader("What drives the prediction (SHAP)")
        if binary:
            imp = pd.Series(binary["shap_importance"]).sort_values()
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.barh(imp.index, imp.values, color=theme.colors()[0])
            ax.set_xlabel("Mean |SHAP|")
            fig.tight_layout()
            st.pyplot(fig)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("ROC curves")
        if binary:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            for name, pts in binary["roc"].items():
                ax.plot(pts["fpr"], pts["tpr"], label=name, lw=2.2 if name.startswith("Proposed") else 1.2)
            ax.plot([0, 1], [0, 1], "k--", lw=.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
    with right:
        st.subheader("Water quality forecast (last 300 hours)")
        if lstm and lstm.get("test_dates"):
            x, (true, pred) = break_at_gaps(lstm["test_dates"][-300:], lstm["test_true"][-300:], lstm["test_pred"][-300:])
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(x, true, label="Actual", lw=1.1)
            ax.plot(x, pred, label="LSTM", lw=1.1, alpha=.85)
            ax.set_ylabel("WQI (lower = better)")
            ax.legend(fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
    theme.note("Use the tool switcher above to explore the data, make predictions, classify quality levels, forecast "
               "and compare models in detail.")
