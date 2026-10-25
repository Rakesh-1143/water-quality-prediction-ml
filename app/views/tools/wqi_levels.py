"""Dashboard tool: wqi levels."""
import pandas as pd
import streamlit as st

from config import WQI_CLASSES
from src import theme
from src.ui import feature_sliders, multiclass_bundle, require, shap_bar
from src.wqi import STANDARDS, classify_wqi, compute_wqi


def render():
    st.caption("Four relative quality levels from the weighted-arithmetic WQI (lower WQI = better water). "
               "The Kaggle data has no WQI label, so levels are WQI quartiles.")

    bundle = require(multiclass_bundle(), "train_multiclass.py")
    X = feature_sliders("wqi")

    wqi_value = float(compute_wqi(X).iloc[0])
    formula_level = classify_wqi(wqi_value, bundle.cutoffs)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("WQI (formula)")
        st.metric("WQI value", f"{wqi_value:.1f}")
        st.write(f"Level by formula: **{WQI_CLASSES[formula_level]}**")
        st.caption("Level cut-offs: " + " / ".join(f"{c:.1f}" for c in bundle.cutoffs))
        with st.expander("Guideline limits used"):
            st.dataframe(pd.Series(STANDARDS, name="Limit"))

    with col2:
        st.subheader("Model prediction")
        if st.button("Predict level", use_container_width=True):
            proba = bundle.predict_proba(X)[0]
            level = int(proba.argmax())
            theme.ladder(WQI_CLASSES, level)
            st.success(f"Predicted level: {WQI_CLASSES[level]}  ({proba[level]:.1%})")
            st.bar_chart(pd.Series(proba, index=WQI_CLASSES, name="Probability"), color=theme.colors()[0])
            st.pyplot(shap_bar(bundle.explain(X)[0, :, level],
                               f"SHAP for class '{WQI_CLASSES[level]}'"))
