"""Home: landing page."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from config import APP_NAME, GITHUB_URL
from src import theme
from src.ui import results

TOOLS = [("📊", "Explore data", "Class balance, missing values, outliers and correlations."),
         ("🔬", "Predict", "Potable or not, with confidence and a SHAP explanation."),
         ("🏷️", "WQI levels", "Excellent to Very Poor, with class probabilities."),
         ("📈", "Forecast", "LSTM next-hour WQI on a real Brisbane River buoy."),
         ("🏆", "Performance", "Baselines, ablation, ROC curves and stability.")]


def open_tool(tool):
    st.session_state["_goto_tool"] = tool
    st.switch_page("views/dashboard.py")


theme.hero("Explainable AI · water quality", "Know your water quality, and why",
           "A SHAP-guided neural network fused with XGBoost predicts potability and water quality levels, "
           "and an LSTM forecasts the index on real river-sensor data.",
           stats={"Input features": 9, "Prediction tasks": 4, "Sensor readings": "30,894"}, height=470)

c1, c2, _ = st.columns([1.3, 1.3, 5])
if c1.button("Open dashboard →", type="primary", key="home_cta_dash"):
    st.switch_page("views/dashboard.py")
if c2.button("Learn more", key="home_cta_about"):
    st.switch_page("views/about.py")

theme.stat_strip([(3276, "Water samples", {}), (30894, "Sensor readings", {}),
                  (4, "Prediction tasks", {}), (5, "Models compared", {})])

theme.section("Dashboard", "Everything in one place",
              "Five tools, one click away. Pick one to jump straight into the dashboard.")
for col, (icon, name, text) in zip(st.columns(5), TOOLS):
    with col:
        theme.card(icon, name, text, min_height=238)
        if st.button("Open →", key=f"home_open_{name}", use_container_width=True):
            open_tool(name)

theme.section("Method", "How it works", "Leakage-free from the first split to the final test.")
theme.timeline([("Split", "Stratified 70/15/15"), ("Preprocess", "Median impute + min-max"),
                ("SHAP", "Feature importance"), ("Init ANN", "Importance-scaled weights"),
                ("Boost", "XGBoost fusion"), ("Evaluate", "Hold-out test set")])

theme.section("Results", "Results at a glance", "Measured on data the models never saw during training or tuning.")
binary, wqi, lstm = results("binary"), results("wqi"), results("lstm")
m1, m2, m3 = st.columns(3)
if binary:
    m = binary["hybrid_average"]
    m1.metric("Potability accuracy", f"{m['accuracy']:.1%}", f"AUC {m['roc_auc']:.3f}", delta_color="off")
if wqi:
    m2.metric("Multiclass WQI accuracy", f"{wqi['hybrid']['accuracy']:.1%}",
              f"macro-F1 {wqi['hybrid']['f1_macro']:.3f}", delta_color="off")
if lstm:
    m3.metric("LSTM forecast R²", f"{lstm['lstm']['r2']:.3f}", f"RMSE {lstm['lstm']['rmse']:.2f}", delta_color="off")
theme.note("Honest numbers: on the Kaggle potability label about 68% is the practical ceiling, and a leaky "
           "protocol would only inflate it. See About for the full story.")

theme.section("Stack", "Built with")
theme.chips(["Python", "TensorFlow / Keras", "XGBoost", "SHAP", "scikit-learn", "Streamlit", "pandas", "Matplotlib"])

theme.cta("Ready to explore?", "Open the dashboard to predict, compare models and forecast water quality.")
_, middle, _ = st.columns([2, 1.4, 2])
if middle.button("Go to the dashboard →", type="primary", key="home_cta_bottom", use_container_width=True):
    st.switch_page("views/dashboard.py")
theme.footer(APP_NAME, GITHUB_URL)
