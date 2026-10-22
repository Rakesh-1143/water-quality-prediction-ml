"""Shared Streamlit helpers (sliders, cached model loading, SHAP bar chart)."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import FEATURES
from src.artifacts import BinaryBundle, MulticlassBundle, WqiBinaryBundle, load_results
from src import theme
from src.pipeline_ts import load_lstm

# label, min, max, default
SLIDERS = {
    "ph": ("pH", 0.0, 14.0, 7.0),
    "Hardness": ("Hardness (mg/L)", 0.0, 500.0, 200.0),
    "Solids": ("Solids (ppm)", 0.0, 65000.0, 20000.0),
    "Chloramines": ("Chloramines (ppm)", 0.0, 15.0, 7.0),
    "Sulfate": ("Sulfate (mg/L)", 0.0, 500.0, 330.0),
    "Conductivity": ("Conductivity (μS/cm)", 0.0, 800.0, 420.0),
    "Organic_carbon": ("Organic Carbon (ppm)", 0.0, 30.0, 14.0),
    "Trihalomethanes": ("Trihalomethanes (μg/L)", 0.0, 150.0, 66.0),
    "Turbidity": ("Turbidity (NTU)", 0.0, 10.0, 4.0),
}


def feature_sliders(prefix):
    """Sidebar sliders for the 9 features -> one-row DataFrame of raw values."""
    st.sidebar.header("Input Parameters")
    values = {f: st.sidebar.slider(SLIDERS[f][0], SLIDERS[f][1], SLIDERS[f][2], SLIDERS[f][3],
                                   key=f"{prefix}_{f}") for f in FEATURES}
    return pd.DataFrame([values])[FEATURES]


@st.cache_resource
def binary_bundle():
    return BinaryBundle.load() if BinaryBundle.exists() else None


@st.cache_resource
def wqibin_bundle():
    return WqiBinaryBundle.load() if WqiBinaryBundle.exists() else None


@st.cache_resource
def multiclass_bundle():
    return MulticlassBundle.load() if MulticlassBundle.exists() else None


@st.cache_resource
def lstm_bundle():
    try:
        return load_lstm()
    except (FileNotFoundError, OSError):
        return None


@st.cache_data
def results(name):
    return load_results(name)


def require(obj, script):
    """Stop the page with a helpful message when trained files are missing."""
    if obj is None:
        st.error(f"Trained files not found. Please run `python model/{script}` first.")
        st.stop()
    return obj


def shap_bar(values, title="Feature contribution (SHAP)"):
    """Horizontal bar chart of per-feature SHAP values for one sample."""
    s = pd.Series(values, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pos, neg, _ = theme.colors()
    ax.barh(s.index, s.values, color=[neg if v < 0 else pos for v in s.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig
