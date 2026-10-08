import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import streamlit as st
from config import ARTIFACT_DIR
from wqi.ui import experiment_paths
st.set_page_config(page_title='Hybrid Water Quality Predictor',page_icon='💧',layout='wide')
st.title('💧 Hybrid Water Quality Prediction')
ARTIFACT_DIR, _ = experiment_paths()
st.markdown('SHAP-informed neural network initialization combined with XGBoost. '
            'Explore the dataset, predict potability, and inspect measured experiment results using the sidebar.')
st.caption('Reference: Al Bataineh, Vamsi & Smith — DOI: 10.1109/JSTARS.2026.3654017')
if (ARTIFACT_DIR/'metadata.json').exists():
    metadata=json.loads((ARTIFACT_DIR/'metadata.json').read_text())
    st.metric('Measured holdout accuracy',f"{metadata['test_metrics']['accuracy']:.1%}")
    st.write('Selected fusion:',metadata['fusion'])
    st.write('Feature initialization:', metadata.get('initialization', 'repeat'))
else:
    st.info('Train the abstract model: python model/train_model.py --initialization feature_glorot --full')
st.warning('This educational classifier does not certify drinking-water safety. '
           'The paper’s reported accuracy is a reference result, not a measurement of this implementation.')
