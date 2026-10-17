import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.ui import results

st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Quality Prediction System")
st.write("---")

st.markdown("""
### Welcome!
A hybrid machine-learning framework for water quality: a **SHAP-initialised neural network**
combined with **XGBoost**, plus multiclass WQI classification and LSTM time-series forecasting.

**Navigate using the sidebar:**
- 📊 **EDA** — Explore the dataset
- 🔬 **Prediction** — Potable / not potable (hybrid model + SHAP explanation)
- 🏷️ **Multiclass WQI** — Water quality level (Excellent / Good / Poor / Very Poor)
- 📈 **Time Series** — LSTM next-step WQI forecast
- 🏆 **Model Performance** — Comparison, ablation, ROC, SHAP ranking
""")

binary, wqi, lstm = results("binary"), results("wqi"), results("lstm")
c1, c2, c3 = st.columns(3)
if binary:
    m = binary["hybrid_average"]
    c1.metric("Hybrid (binary) accuracy", f"{m['accuracy']:.1%}", f"AUC {m['roc_auc']:.3f}", delta_color="off")
if wqi:
    c2.metric("Multiclass WQI accuracy", f"{wqi['hybrid']['accuracy']:.1%}",
              f"macro-F1 {wqi['hybrid']['f1_macro']:.3f}", delta_color="off")
if lstm:
    c3.metric("LSTM forecast R²", f"{lstm['lstm']['r2']:.3f}", f"RMSE {lstm['lstm']['rmse']:.2f}", delta_color="off")

st.info("Dataset: Kaggle Water Potability (3276 samples, 9 features). "
        "Test-set metrics from a stratified 70/15/15 split.")
