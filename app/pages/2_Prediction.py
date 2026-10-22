import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from src import theme
from src.ui import binary_bundle, feature_sliders, require, shap_bar, wqibin_bundle
from src.wqi import compute_wqi

st.set_page_config(page_title="Prediction", page_icon="🔬", layout="wide")
theme.apply()
theme.hero("Hybrid SHAP-ANN + XGBoost", "Water Quality Prediction",
           "Adjust the nine water parameters in the sidebar and get an explained verdict.", height=250)

TASKS = ["Kaggle potability label", "WQI-based good / poor"]
task = st.sidebar.radio("Prediction task", TASKS, key="task")
wqi_task = task == TASKS[1]
bundle = require(wqibin_bundle() if wqi_task else binary_bundle(),
                 "train_model.py --task wqi" if wqi_task else "train_model.py")
if wqi_task:
    st.caption("Good = WQI at or below the dataset median. This label is a formula of the 9 inputs "
               "(not the Kaggle potability label), so accuracy is very high.")
X = feature_sliders("pred")

with st.expander("What do these parameters mean?"):
    st.write("""
    - **pH**: Acidity level (WHO safe range: 6.5–8.5)
    - **Hardness**: Calcium/magnesium content (mg/L)
    - **Solids**: Total dissolved solids (ppm)
    - **Chloramines**: Disinfectant level (ppm)
    - **Sulfate**: Naturally occurring mineral (mg/L)
    - **Conductivity**: Electrical conductivity (μS/cm)
    - **Organic Carbon**: Carbon from organic matter (ppm)
    - **Trihalomethanes**: Chlorination byproduct (μg/L)
    - **Turbidity**: Water cloudiness (NTU)
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Values")
    st.dataframe(X.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("Prediction Result")
    if st.button("Predict", use_container_width=True):
        p_nn, p_xgb = bundle.components(X)
        p = float(bundle.predict_proba(X)[0])

        good, bad = ("Water quality is GOOD (low WQI)", "Water quality is POOR (high WQI)") if wqi_task \
            else ("Water is Safe for Drinking", "Water is NOT Safe for Drinking")
        theme.gauge(max(p, 1 - p), "model confidence")
        if p >= 0.5:
            st.success(good)
        else:
            st.error(bad)
        if wqi_task:
            st.write(f"WQI by formula: **{float(compute_wqi(X).iloc[0]):.1f}** "
                     f"(good if ≤ {bundle.wqi_cutoff:.1f})")

        confidence = max(p, 1 - p)
        st.info(f"Model Confidence: {confidence:.1%}  (P({'good' if wqi_task else 'potable'}) = {p:.3f})")
        st.progress(float(confidence))
        st.caption(f"SHAP-initialised ANN: {float(p_nn[0]):.3f} · XGBoost: {float(p_xgb[0]):.3f} · "
                   f"weight α = {bundle.alpha}")

        st.subheader("Why this prediction? (SHAP)")
        st.pyplot(shap_bar(bundle.explain(X)[0, :, 0]))
        st.caption("Blue pushes toward the positive class (potable / good), orange toward the other (XGBoost component).")
