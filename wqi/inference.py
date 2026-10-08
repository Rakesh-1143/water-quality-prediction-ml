"""Load only new paper artifacts; never silently use the legacy ensemble."""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from config import FEATURES
from wqi.core import average
from wqi.neural import probability


class Predictor:
    def __init__(self, directory):
        from tensorflow.keras.models import load_model
        from xgboost import XGBClassifier
        directory = Path(directory)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata.get('schema') != 1 or self.metadata.get('features') != FEATURES:
            raise ValueError('Artifact schema/features do not match the paper framework')
        if self.metadata.get('fusion') not in {'stacking', 'average'}:
            raise ValueError('Unknown artifact fusion method')
        if self.metadata['fusion'] == 'average' and not 0 <= self.metadata.get('alpha', -1) <= 1:
            raise ValueError('Artifact averaging alpha must be within [0,1]')
        self.preprocess = joblib.load(directory / 'preprocess.joblib')
        self.nn = load_model(directory / 'ann.keras', compile=False)
        self.xgb = XGBClassifier()
        self.xgb.load_model(directory / 'xgboost.json')
        self.combiner = joblib.load(directory / 'combiner.joblib')

    def predict(self, frame):
        if frame.empty:
            raise ValueError('Provide at least one measurement row')
        if not frame.columns.is_unique:
            raise ValueError('Measurement column names must be unique')
        if not set(FEATURES).issubset(frame.columns):
            raise ValueError('All nine features are required')
        values = frame[FEATURES].apply(pd.to_numeric, errors='raise')
        # Missing measurements use the training medians, as in the paper's
        # preprocessing pipeline. Infinite values remain invalid.
        if np.isinf(values.to_numpy()).any():
            raise ValueError('Prediction inputs must not contain infinite numbers')
        scaled = self.preprocess.transform(values)
        pn = probability(self.nn, scaled)
        px = self.xgb.predict_proba(scaled)[:, 1]
        if self.metadata['fusion'] == 'stacking':
            result = self.combiner.predict_proba(np.column_stack([pn, px]))[:, 1]
        else:
            result = average(pn, px, self.metadata['alpha'])
        return result, pn, px

    def predict_frame(self, frame, batch_size=512):
        """CSV-ready predictions in original row order using bounded batches."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if frame.empty:
            raise ValueError('Provide at least one measurement row')
        chunks = []
        for start in range(0, len(frame), batch_size):
            part = frame.iloc[start:start + batch_size]
            p, pn, px = self.predict(part)
            values = part[FEATURES].apply(pd.to_numeric, errors='raise')
            scaler = self.preprocess.named_steps['scaler']
            outside = (values.to_numpy() < scaler.data_min_) | (values.to_numpy() > scaler.data_max_)
            chunks.append(pd.DataFrame({
                'potable_probability': p, 'ann_probability': pn, 'xgboost_probability': px,
                'predicted_class': (p >= .5).astype(int),
                'predicted_label': np.where(p >= .5, 'Potable', 'Nonpotable'),
                'imputed_measurements': values.isna().sum(axis=1).to_numpy(),
                'outside_training_range': outside.sum(axis=1)}, index=part.index))
        return pd.concat(chunks)
