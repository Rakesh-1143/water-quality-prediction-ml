"""Validate CSV headers and numeric model boundaries before inference."""
import csv
import io
from pathlib import Path
import numpy as np
import pandas as pd


def read_csv(source):
    if isinstance(source, (str, Path)):
        content = Path(source).read_bytes()
    elif hasattr(source, 'getvalue'):
        content = source.getvalue()
    else:
        position = source.tell()
        content = source.read()
        source.seek(position)
    if isinstance(content, bytes):
        content = content.decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(content)), [])
    if not header or any(not name.strip() for name in header):
        raise ValueError('CSV must have nonempty column names')
    if len(set(header)) != len(header):
        raise ValueError('CSV column names must be unique; duplicate headers are invalid')
    return pd.read_csv(io.StringIO(content))


def model_inputs(values):
    values = np.asarray(values, dtype='float64')
    if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
        raise ValueError('Preprocessed inputs must be finite and representable as float32')
    return values.astype('float32')


def probabilities(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Model returned invalid probabilities; prediction was rejected')
    return values
