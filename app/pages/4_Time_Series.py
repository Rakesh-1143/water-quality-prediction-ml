import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import theme
from src.timeseries import break_at_gaps, clean_series, forecast_next
from src.ui import lstm_bundle, require, results
from src.wqi import PROFILES, class_name, classify_wqi

st.set_page_config(page_title="Time Series", page_icon="📈", layout="wide")
theme.apply()
theme.hero("LSTM · real sensor data", "Time-Series WQI Forecast",
           "One-hour-ahead water quality index forecast on the Brisbane River buoy record.", height=250)

bundle = require(lstm_bundle(), "train_lstm.py")
res = results("lstm")
features = bundle["features"]
unit = "hour" if bundle["freq"] == "h" else "day"

if bundle["simulated"]:
    st.warning("This model was trained on a **simulated** series (for testing only). "
               "Retrain on real data with `python model/train_lstm.py`.")
else:
    st.info("Trained on **real** sensor data (Brisbane River buoy, Queensland Government Open Data, "
            f"Apache-2.0). Inputs: {', '.join(features)}. WQI is computed from the "
            f"`{bundle['profile']}` parameters and forecast one {unit} ahead.")

if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM RMSE", f"{res['lstm']['rmse']:.2f}", f"persistence {res['persistence']['rmse']:.2f}", delta_color="off")
    c2.metric("LSTM R²", f"{res['lstm']['r2']:.3f}", f"persistence {res['persistence']['r2']:.3f}", delta_color="off")
    c3.metric("Level accuracy", f"{res['level_accuracy']:.1%}", f"persistence {res['level_accuracy_persistence']:.1%}", delta_color="off")
    st.subheader(f"Test period: actual vs predicted WQI (next {unit})")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x, (true, pred) = break_at_gaps(res["test_dates"], res["test_true"], res["test_pred"])
    ax.plot(x, true, label="Actual", lw=1)
    ax.plot(x, pred, label="LSTM", alpha=0.8, lw=1)
    ax.set_ylabel("WQI (lower = better)")
    ax.legend()
    st.pyplot(fig)
    st.caption(f"{res['n_test']} test steps, never used for training or tuning. "
               "'Persistence' predicts that the next WQI equals the current one. "
               "Windows never span a gap in the sensor record, and lines are broken at gaps.")

st.subheader("Forecast from your own data")
st.write(f"Upload a CSV with a `date` column and these columns: `{'`, `'.join(features)}` "
         f"(at least {bundle['window']} consecutive {unit}ly rows).")
file = st.file_uploader("CSV file", type="csv")
if file is not None:
    try:
        ts = clean_series(pd.read_csv(file), features)
        value = forecast_next(bundle["model"], bundle["scaler"], bundle["cols"], ts[features],
                              bundle["window"], PROFILES[bundle["profile"]], dates=ts["date"])
        level = class_name(classify_wqi(value, bundle["cutoffs"]))
        st.metric(f"Forecast WQI (next {unit})", f"{value:.1f}")
        st.success(f"Expected quality level: {level}")
    except ValueError as e:
        st.error(str(e))
