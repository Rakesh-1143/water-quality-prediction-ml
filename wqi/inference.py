"""Load only new paper artifacts; never silently use the legacy ensemble."""
import json
import joblib
import numpy as np
import pandas as pd
from config import FEATURES
from wqi.core import average
from wqi.neural import probability


class Predictor:
    def __init__(self, directory):
        from tensorflow.keras.models import load_model
        from xgboost import XGBClassifier
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata.get('schema') != 1 or self.metadata.get('features') != FEATURES:
            raise ValueError('Artifact schema/features do not match the paper framework')
        self.preprocess = joblib.load(directory / 'preprocess.joblib')
        self.nn = load_model(directory / 'ann.keras', compile=False)
        self.xgb = XGBClassifier()
        self.xgb.load_model(directory / 'xgboost.json')
        self.combiner = joblib.load(directory / 'combiner.joblib')

    def predict(self, frame):
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
