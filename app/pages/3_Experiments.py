import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import pandas as pd
import streamlit as st
from config import REPORT_DIR
st.set_page_config(page_title='Experiments',layout='wide')
st.title('Measured Experiments')
if not (REPORT_DIR/'evaluation.json').exists():
    st.info('Run python model/train_model.py --tune --full to generate experiment reports.')
    st.stop()
report=json.loads((REPORT_DIR/'evaluation.json').read_text())
st.write('Split sizes:',report['split_sizes'])
st.write('Fusion selected using validation:',report['selected_fusion'])
st.dataframe(pd.DataFrame(report['test']).T)
st.caption('Scores above come from this run. Reference paper: accuracy 0.869, F1 0.849, ROC-AUC 0.894. '
           'Equal results are not guaranteed.')
for title,name in [('Ablations','ablations.json'),('Feature perturbation','feature_sensitivity.json'),
    ('Five split repetitions','split_sensitivity.json'),('Hybrid feature importance','hybrid_shap_importance.json'),
    ('SHAP alignment','importance_alignment.json')]:
    with st.expander(title):
        p=REPORT_DIR/name
        if p.exists(): st.json(json.loads(p.read_text()))
        else: st.info('Not generated for this run; use --full for the full suite.')
for name in ['roc_curves.png','loss_curves.png','metrics.png','actual_vs_probability.png',
             'training_tree_shap.png','hybrid_shap.png']:
    if (REPORT_DIR/name).exists(): st.image(str(REPORT_DIR/name))
