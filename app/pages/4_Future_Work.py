import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
import streamlit as st
from config import BASE_DIR
st.set_page_config(page_title='Future Work',layout='wide')
st.title('Experimental Future Work')
st.info('Training and prediction code is available. Genuine future-work datasets and real-data '
        'evaluation are still required. Synthetic integration tests are software checks only.')
st.table(pd.DataFrame([
    ['Multiclass hybrid','SHAP ANN + XGBoost','Nine features + verified WQI_class labels'],
    ['Forecasting','LSTM','One site: equally spaced timestamp + measured WQI'],
    ['Transfer learning','Fine-tuned source ANN','Target-region features + Potability'],
    ['Domain adaptation','DANN','Labeled source + unlabeled target; separate target test']
],columns=['Extension','Model','Data needed']))
with st.expander('Training and data instructions'):
    st.markdown((BASE_DIR/'docs/FUTURE_WORK.md').read_text())
st.subheader('Predict with a future-work model you trained')
snapshots=sorted((BASE_DIR/'experiments').glob('*/metadata.json'))
if not snapshots:
    st.info('No future-work models found. Train one into experiments/<run-name> using the documented CLI.')
    st.stop()
chosen=Path(st.selectbox('Saved experimental model',[str(p.parent) for p in snapshots]))
uploaded=st.file_uploader('Measurements (CSV)',type=['csv'])
if uploaded is not None:
    try:
        from future_work.core import FuturePredictor
        @st.cache_resource
        def load(directory, signature):
            return FuturePredictor(directory)
        signature=tuple((p.name,p.stat().st_mtime_ns) for p in sorted(chosen.iterdir()) if p.is_file())
        predictor=load(chosen,signature)
        st.write('Experimental model:',predictor.meta['kind'])
        result=predictor.predict(pd.read_csv(uploaded))
        st.dataframe(result,use_container_width=True)
        st.download_button('Download experimental predictions',result.to_csv(index=False),
                           file_name='future-predictions.csv',mime='text/csv')
    except (ValueError,OSError,KeyError) as exc:
        st.error(f'Cannot predict: {exc}')
