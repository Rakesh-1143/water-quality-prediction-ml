import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import streamlit as st
from wqi.validation import read_csv
from config import ARTIFACT_DIR, FEATURES
from wqi.inference import Predictor
from wqi.ui import experiment_paths
st.set_page_config(page_title='Prediction',page_icon='🔬',layout='wide')
st.title('Water Potability Prediction')
ARTIFACT_DIR, _ = experiment_paths()

@st.cache_resource
def load_predictor(path, artifact_signature):
    return Predictor(path)

try:
    # Include file signatures so retraining invalidates cached resources.
    names=['metadata.json','preprocess.joblib','ann.keras','xgboost.json','combiner.joblib']
    signature=tuple((ARTIFACT_DIR/n).stat().st_mtime_ns for n in names)
    predictor=load_predictor(ARTIFACT_DIR,signature)
except FileNotFoundError:
    st.info('Train the selected model first. See README for the two training commands.')
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
st.subheader('Predict a CSV of measurements')
st.write('Upload the nine measurement columns; a Potability label is not required. '
         'Blank measurements use the training medians.')
template = pd.DataFrame([item[3] for item in specs], index=FEATURES).T
st.download_button('Download CSV template', template.to_csv(index=False),
                   file_name='measurement-template.csv', mime='text/csv')
uploaded = st.file_uploader('Measurement CSV', type=['csv'])
if uploaded is not None:
    try:
        measurements = read_csv(uploaded)
        output = predictor.predict_frame(measurements)
        output.insert(0, 'row_number', range(1, len(output) + 1))
        st.dataframe(output, use_container_width=True)
        st.download_button('Download predictions', output.to_csv(index=False, float_format='%.17g'),
                           file_name='potability-predictions.csv', mime='text/csv')
    except (ValueError, KeyError) as exc:
        st.error(f'Cannot predict this CSV: {exc}')
