"""Shared validation, bounded TensorFlow training, and saved inference."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from config import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from wqi.core import preprocessor
from wqi.validation import model_inputs, probabilities


def features(frame):
    if frame.empty or not frame.columns.is_unique or not set(FEATURES).issubset(frame.columns):
        raise ValueError('Provide nonempty rows with all nine unique feature columns')
    X = frame[FEATURES].apply(pd.to_numeric, errors='raise').astype('float64')
    if np.isinf(X.to_numpy()).any():
        raise ValueError('Infinite measurements are invalid')
    return X


def labeled(frame, label='Potability', multiclass=False, evaluation=False):
    X = features(frame)
    if not evaluation and X.isna().all().any():
        raise ValueError('Every training feature needs observed values')
    if label not in frame or frame[label].isna().any():
        raise ValueError(f'Complete {label} labels are required')
    if multiclass:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(frame[label].astype(str))
        if len(encoder.classes_) < 3:
            raise ValueError('Multiclass WQI needs at least three genuine quality classes')
        y = encoder.transform(frame[label].astype(str))
        classes = encoder.classes_.tolist()
    else:
        y = pd.to_numeric(frame[label], errors='raise').to_numpy()
        if not set(np.unique(y)).issubset({0, 1}) or (not evaluation and set(np.unique(y)) != {0, 1}):
            raise ValueError('Potability must contain both binary classes 0 and 1')
        y = y.astype(int)
        classes = ['Nonpotable', 'Potable']
    if not evaluation and pd.Series(y).value_counts().min() < 8:
        raise ValueError('At least eight observations per class are needed for stratified partitions')
    return X, y, classes


def partitions(y, seed):
    ids = np.arange(len(y))
    train, rest = train_test_split(ids, test_size=.3, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed)
    return train, val, test


def seed_all(seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def dataset(X, y, seed=42, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((model_inputs(X), y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return ds.batch(32).with_options(options)


def fit(model, X, y, Xv, yv, epochs, seed):
    if epochs < 1:
        raise ValueError('Epochs must be positive')
    history = model.fit(dataset(X, y, seed, True), validation_data=dataset(Xv, yv),
        epochs=epochs, verbose=0, shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                    restore_best_weights=True)])
    return history.history


def scores(y, probabilities, classes):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), len(classes)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid class probability matrix')
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
    p = p / p.sum(axis=1, keepdims=True)  # float32 model export roundoff
    pred = p.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=np.arange(len(classes)), zero_division=0)),
        'log_loss': float(log_loss(y, p, labels=np.arange(len(classes)))),
        'confusion_matrix': confusion_matrix(y, pred, labels=np.arange(len(classes))).tolist()}


def binary_prob(model, X):
    p = np.asarray(model(model_inputs(X), training=False)).reshape(-1)
    return probabilities(np.column_stack([1-p, p]))


def fingerprint(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def reject_overlap(a, b):
    # Prevent identical observations being adaptation inputs and evaluation cases.
    ha = set(pd.util.hash_pandas_object(features(a).astype('float64'), index=False).tolist())
    hb = set(pd.util.hash_pandas_object(features(b).astype('float64'), index=False).tolist())
    if ha & hb:
        raise ValueError('Adaptation/source and evaluation datasets contain overlapping measurements')


def save(directory, model, prep, metadata, report, predictions):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Choose a new empty output directory; existing runs are preserved')
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory/'model.keras')
    joblib.dump(prep, directory/'preprocess.joblib')
    metadata = {'schema': 2, 'features': FEATURES, 'status': 'experimental', **metadata}
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (directory/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    predictions.to_csv(directory/'test_predictions.csv', index=False, float_format='%.17g')


def training_preprocessor(X):
    if X.isna().all().any():
        raise ValueError('Each training feature needs at least one observed value')
    return preprocessor().fit(X)


class FuturePredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory/'metadata.json').read_text())
        if self.meta.get('schema') != 2 or self.meta.get('features') != FEATURES:
            raise ValueError('Unsupported future-work model schema')
        if self.meta.get('kind') not in {'multiclass','timeseries','transfer','dann'}:
            raise ValueError('Unsupported future-work model type')
        self.model = tf.keras.models.load_model(directory/'model.keras', compile=False)
        self.prep = joblib.load(directory/'preprocess.joblib')
        if self.meta['kind'] == 'multiclass':
            from xgboost import XGBClassifier
            self.xgb = XGBClassifier()
            self.xgb.load_model(directory/'xgboost.json')

    def predict(self, frame):
        X = features(frame)
        scaled = model_inputs(self.prep.transform(X))
        if self.meta['kind'] == 'timeseries':
            length = self.meta['lookback']
            validate_time(frame, self.meta['timestamp'], self.meta['interval_ns'])
            if len(X) < length:
                raise ValueError(f'Provide at least {length} consecutive observations')
            # One future forecast from the last history window; no future rows required.
            raw = float(self.model(scaled[-length:][None].astype('float32'), training=False).numpy()[0,0])
            value = raw * self.meta['target_std'] + self.meta['target_mean']
            if not np.isfinite(value):
                raise ValueError('Model returned a nonfinite WQI forecast')
            stamp = pd.to_datetime(frame[self.meta['timestamp']], utc=True).iloc[-1]
            stamp += pd.Timedelta(self.meta['interval_ns'] * self.meta['horizon'], unit='ns')
            return pd.DataFrame({'forecast_timestamp': [stamp.isoformat()], 'predicted_WQI': [value]})
        if self.meta['kind'] == 'multiclass':
            pn = self.model(scaled.astype('float32'), training=False).numpy()
            p = self.meta['alpha'] * pn + (1-self.meta['alpha']) * self.xgb.predict_proba(scaled)
        else:
            p = binary_prob(self.model, scaled)
        p = probabilities(p)
        if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Class probabilities must sum to one')
        out = pd.DataFrame({f'probability_{c}': p[:,i] for i,c in enumerate(self.meta['classes'])})
        out.insert(0, 'predicted_class', np.array(self.meta['classes'])[p.argmax(axis=1)])
        return out


def validate_time(frame, timestamp='timestamp', interval_ns=None):
    if timestamp not in frame:
        raise ValueError('Timestamp column is required; row order is not time')
    stamps = pd.to_datetime(frame[timestamp], errors='raise', utc=True)
    if stamps.isna().any() or not stamps.is_monotonic_increasing or stamps.duplicated().any():
        raise ValueError('Use unique chronological timestamps for one monitoring site')
    steps = np.diff(stamps.astype('int64').to_numpy())
    if len(steps) == 0 or np.any(steps != steps[0]) or steps[0] <= 0:
        raise ValueError('Equally spaced consecutive observations are required')
    if interval_ns is not None and steps[0] != interval_ns:
        raise ValueError('Prediction sampling interval differs from training')
    return stamps, int(steps[0])
