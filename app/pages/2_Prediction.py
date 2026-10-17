import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from src.ui import binary_bundle, feature_sliders, require, shap_bar

st.set_page_config(page_title="Prediction", page_icon="🔬", layout="wide")
st.title("Water Quality Prediction")

bundle = require(binary_bundle(), "train_model.py")
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

        if p >= 0.5:
            st.success("Water is Safe for Drinking")
        else:
            st.error("Water is NOT Safe for Drinking")

        confidence = max(p, 1 - p)
        st.info(f"Model Confidence: {confidence:.1%}  (P(potable) = {p:.3f})")
        st.progress(float(confidence))
        st.caption(f"SHAP-initialised ANN: {float(p_nn[0]):.3f} · XGBoost: {float(p_xgb[0]):.3f} · "
                   f"weight α = {bundle.alpha}")

        st.subheader("Why this prediction? (SHAP)")
        st.pyplot(shap_bar(bundle.explain(X)[0, :, 0]))
        st.caption("Blue pushes toward potable, orange toward not potable (XGBoost component).")
