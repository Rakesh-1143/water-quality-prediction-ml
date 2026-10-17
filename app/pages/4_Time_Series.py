import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import FEATURES
from src.timeseries import clean_series, forecast_next
from src.ui import lstm_bundle, require, results
from src.wqi import classify_wqi, class_name

st.set_page_config(page_title="Time Series", page_icon="📈", layout="wide")
st.title("📈 Time-Series WQI Forecast (LSTM)")

bundle = require(lstm_bundle(), "train_lstm.py")
res = results("lstm")

if res and res["simulated"]:
    st.warning("The bundled model was trained on a **simulated** daily series (the Kaggle dataset has no "
               "time column). Use it to test the pipeline; retrain on real monitoring data for real use.")

if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM RMSE", f"{res['lstm']['rmse']:.2f}", f"persistence {res['persistence']['rmse']:.2f}", delta_color="off")
    c2.metric("LSTM R²", f"{res['lstm']['r2']:.3f}", f"persistence {res['persistence']['r2']:.3f}", delta_color="off")
    c3.metric("Level accuracy", f"{res['level_accuracy']:.1%}", f"persistence {res['level_accuracy_persistence']:.1%}", delta_color="off")
    st.subheader("Test period: actual vs predicted WQI")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = pd.to_datetime(res["test_dates"]) if res["test_dates"] else range(len(res["test_true"]))
    ax.plot(x, res["test_true"], label="Actual")
    ax.plot(x, res["test_pred"], label="LSTM", alpha=0.8)
    ax.set_ylabel("WQI")
    ax.legend()
    st.pyplot(fig)

st.subheader("Forecast from your own data")
st.write(f"Upload a CSV with a `date` column and the 9 features (at least {bundle['window']} rows).")
file = st.file_uploader("CSV file", type="csv")
if file is not None:
    try:
        ts = clean_series(pd.read_csv(file))
        value = forecast_next(bundle["model"], bundle["scaler"], bundle["cols"], ts[FEATURES], bundle["window"])
        level = class_name(classify_wqi(value, bundle["cutoffs"]))
        st.metric("Forecast WQI (next step)", f"{value:.1f}")
        st.success(f"Expected quality level: {level}")
    except ValueError as e:
        st.error(str(e))
