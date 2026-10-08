"""Select the same experiment snapshot across the Streamlit pages."""
from config import ARTIFACT_DIR, REPORT_DIR, ABSTRACT_ARTIFACT_DIR, ABSTRACT_REPORT_DIR


def experiment_paths():
    import streamlit as st
    mode = st.sidebar.selectbox('Experiment snapshot',
        ['Abstract implementation', 'Paper reproduction attempt'], key='experiment_snapshot')
    if mode == 'Abstract implementation':
        return ABSTRACT_ARTIFACT_DIR, ABSTRACT_REPORT_DIR
    return ARTIFACT_DIR, REPORT_DIR
