import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src import theme
from src.ui import results

st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide"
)

theme.apply()
theme.hero("Hybrid machine learning · water quality",
           "Predict water quality with explainable AI",
           "A SHAP-initialised neural network fused with XGBoost, plus multiclass WQI classification "
           "and LSTM forecasting on real river-sensor data.",
           stats={"Input features": 9, "Prediction tasks": 4, "Real sensor readings": "30,894"}, height=420)

st.subheader("Explore the system")
theme.feature_cards([
    ("📊", "EDA", "Class balance, missing values, outliers and correlations of the dataset."),
    ("🔬", "Prediction", "Potable or not, with confidence and a SHAP explanation of every decision."),
    ("🏷️", "Multiclass WQI", "Four quality levels from Excellent to Very Poor, with class probabilities."),
    ("📈", "Time Series", "LSTM next-hour WQI forecast, tested on a real Brisbane River buoy."),
    ("🏆", "Model Performance", "Baselines, ablation, ROC curves, SHAP ranking and stability."),
])
pages = st.columns(5)
for col, (path, label) in zip(pages, [("pages/1_EDA.py", "Open EDA"), ("pages/2_Prediction.py", "Open Prediction"),
                                      ("pages/3_Multiclass_WQI.py", "Open WQI"), ("pages/4_Time_Series.py", "Open Time Series"),
                                      ("pages/5_Model_Performance.py", "Open Performance")]):
    col.page_link(path, label=label, icon="➡️")

st.subheader("How it works")
theme.timeline([("Split", "Stratified 70/15/15"), ("Preprocess", "Median impute + min-max"),
                ("SHAP", "Feature importance"), ("Init ANN", "Importance-scaled weights"),
                ("Boost", "XGBoost fusion"), ("Evaluate", "Hold-out test set")])

st.subheader("Results at a glance")
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
