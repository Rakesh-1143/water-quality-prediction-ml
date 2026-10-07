import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import streamlit as st
from config import ARTIFACT_DIR, FEATURES
from wqi.inference import Predictor
st.set_page_config(page_title='Prediction',page_icon='🔬',layout='wide')
st.title('Water Potability Prediction')

@st.cache_resource
def load_predictor(path, artifact_signature):
    return Predictor(path)

try:
    # Include file signatures so retraining invalidates cached resources.
    names=['metadata.json','preprocess.joblib','ann.keras','xgboost.json','combiner.joblib']
    signature=tuple((ARTIFACT_DIR/n).stat().st_mtime_ns for n in names)
    predictor=load_predictor(ARTIFACT_DIR,signature)
except FileNotFoundError:
    st.info('Train the paper model first: python model/train_model.py --tune --full')
    st.stop()
except Exception as exc:
    st.error(f'Cannot load the paper model: {exc}')
    st.stop()

specs=[('pH',0.,14.,7.),('Hardness (mg/L)',0.,500.,200.),
    ('Solids (ppm)',0.,70000.,20000.),('Chloramines (ppm)',0.,15.,7.),
    ('Sulfate (mg/L)',0.,500.,300.),('Conductivity (μS/cm)',0.,1000.,500.),
    ('Organic Carbon (ppm)',0.,30.,15.),('Trihalomethanes (μg/L)',0.,150.,80.),
    ('Turbidity (NTU)',0.,10.,4.)]
with st.form('prediction'):
    cols=st.columns(3); values=[]
    for i,(label,low,high,default) in enumerate(specs):
        values.append(cols[i%3].number_input(label,min_value=low,max_value=high,value=default))
    submitted=st.form_submit_button('Predict',use_container_width=True)
if submitted:
    frame=pd.DataFrame([values],columns=FEATURES)
    scaler=predictor.preprocess.named_steps['scaler']
    out=(np.array(values)<scaler.data_min_)|(np.array(values)>scaler.data_max_)
    if out.any(): st.warning('Some inputs fall outside training ranges: '+', '.join(np.array(FEATURES)[out]))
    result,pn,px=predictor.predict(frame); p=float(result[0])
    st.subheader('Predicted class: '+('Potable' if p>=.5 else 'Nonpotable'))
    st.metric('Model probability of potable class',f'{p:.1%}')
    st.write({'SHAP-initialized ANN probability':float(pn[0]),'XGBoost probability':float(px[0])})
    st.caption('Probability is a model output; calibration and actual safety are not established.')
st.warning('Use laboratory testing and applicable standards to determine drinking-water safety.')
