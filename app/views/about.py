"""About: project, research basis, data and honest limitations."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from config import APP_NAME, GITHUB_URL
from src import theme

theme.hero("About the project", "Hybrid machine learning for water quality",
           "Research-grade, reproducible and explainable: every number on this site comes from a held-out test set.",
           height=300, scene="waves")

theme.section("The project", "What it does",
              "The system predicts whether water is drinkable from nine physicochemical measurements, classifies it into "
              "four quality levels, and forecasts a water quality index from real sensor data.")
a, b, c = st.columns(3)
with a:
    theme.card("🧠", "SHAP-guided neural network", "XGBoost + SHAP rank the inputs; the network's first-layer weights start "
               "larger for the more important features.", min_height=190)
with b:
    theme.card("🌲", "Boosting fusion", "A weighted average of the network and XGBoost, with the weight chosen on a "
               "validation set; stacking is also available.")
with c:
    theme.card("📈", "LSTM forecasting", "A sequence model predicts the next-hour index and is compared with a "
               "persistence baseline.")

theme.section("Research basis", "Built on published work")
d, e = st.columns(2)
with d:
    theme.card("📄", "IEEE JSTARS 2026", "Al Bataineh, Vamsi & Smith, “A Hybrid Machine Learning Framework for Water "
               "Quality Index Prediction Using Feature-Based Neural Network Initialization”, vol. 19, pp. 4887–4905. "
               "doi:10.1109/JSTARS.2026.3654017")
with e:
    theme.card("🎓", "M.Tech extension", "“A Multiclass and Time-Series Water Quality Prediction Using SHAP and LSTM”: "
               "multiclass WQI classification and LSTM time-series forecasting.")

theme.section("Data", "Two datasets")
f, g = st.columns(2)
with f:
    theme.card("💧", "Kaggle Water Potability", "3,276 samples, 9 features (pH, hardness, solids, chloramines, sulfate, "
               "conductivity, organic carbon, trihalomethanes, turbidity), binary potability label.")
with g:
    theme.card("🛰️", "Brisbane River buoy", "30,894 sensor readings (Aug 2023 – Jun 2024) from the Queensland Government "
               "Open Data Portal, Apache-2.0. Used for time-series forecasting.")

theme.section("Honesty", "Limitations, stated plainly")
theme.note("The paper reports 86.9% accuracy on the Kaggle label. With a leakage-free split this project reaches 67.7% "
           "(AUC 0.684), which appears to be the practical ceiling for these nine features; even a leaky protocol "
           "only reached about 74% in our tests.")
theme.note("The WQI-based good/poor task and the four WQI levels use labels computed from the same nine inputs by a "
           "formula, so their high accuracy is expected and not comparable to the potability task.")
theme.note("The Brisbane site is a tidal estuary and its WQI limits are approximate guideline values; the LSTM beats the "
           "“next hour equals this hour” baseline only modestly.")

theme.section("Quality", "Tested end to end",
              "Automated tests cover the data split, preprocessing, WQI formulas, model training, saved-model "
              "reproduction, every page and the animations.")
theme.chips(["No data leakage", "Reproducible seeds", "SHAP explanations", "Automated tests", "Open source (MIT)"])

theme.cta("Questions or ideas?", "Get in touch or open an issue on GitHub.")
_, middle, _ = st.columns([2, 1.4, 2])
if middle.button("Contact us →", type="primary", key="about_contact", use_container_width=True):
    st.switch_page("views/contact.py")
theme.footer(APP_NAME, GITHUB_URL)
