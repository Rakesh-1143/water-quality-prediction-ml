"""Dashboard: overview and the five analysis tools."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import streamlit as st

from config import APP_NAME, GITHUB_URL
from src import theme
from src.ui import results
from views.tools import explore, forecast, overview, performance, predict, wqi_levels

TOOLS = {"Overview": overview, "Explore data": explore, "Predict": predict,
         "WQI levels": wqi_levels, "Forecast": forecast, "Performance": performance}

theme.hero("Dashboard", "Water quality dashboard",
           "Pick a tool: explore the data, predict, classify, forecast or compare models.", height=230)

goto = st.session_state.pop("_goto_tool", None)
if goto in TOOLS:
    st.session_state["dash_tool"] = goto

binary, wqi, lstm = results("binary"), results("wqi"), results("lstm")
k1, k2, k3, k4 = st.columns(4)
if binary:
    k1.metric("Potability accuracy", f"{binary['hybrid_average']['accuracy']:.1%}",
              f"AUC {binary['hybrid_average']['roc_auc']:.3f}", delta_color="off")
wqibin = results("wqibin")
if wqibin:
    k2.metric("WQI good/poor accuracy", f"{wqibin['hybrid_average']['accuracy']:.1%}",
              f"F1 {wqibin['hybrid_average']['f1']:.3f}", delta_color="off")
if wqi:
    k3.metric("Multiclass WQI accuracy", f"{wqi['hybrid']['accuracy']:.1%}",
              f"macro-F1 {wqi['hybrid']['f1_macro']:.3f}", delta_color="off")
if lstm:
    k4.metric("LSTM forecast R²", f"{lstm['lstm']['r2']:.3f}", f"RMSE {lstm['lstm']['rmse']:.2f}", delta_color="off")

tool = st.segmented_control("Tool", list(TOOLS), default="Overview", key="dash_tool",
                            selection_mode="single", label_visibility="collapsed") or "Overview"
st.write("")
TOOLS[tool].render()
theme.footer(APP_NAME, GITHUB_URL)
