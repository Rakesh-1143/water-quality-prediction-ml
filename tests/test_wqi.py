import numpy as np
import pandas as pd
import pytest

from config import FEATURES, WQI_CLASSES
from src.pipeline import make_wqi_labels
from src.wqi import (IDEAL, STANDARDS, WEIGHTS, class_name, classify_wqi, compute_wqi,
                     quartile_cutoffs)


def row(values):
    return pd.DataFrame([values])[FEATURES]


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_wqi_zero_at_ideal_and_100_at_standard():
    assert compute_wqi(row(IDEAL)).iloc[0] == pytest.approx(0.0)
    assert compute_wqi(row(STANDARDS)).iloc[0] == pytest.approx(100.0)


def test_wqi_increases_with_pollution():
    base = dict(IDEAL)
    low = compute_wqi(row(base)).iloc[0]
    base["Sulfate"] = 500
    assert compute_wqi(row(base)).iloc[0] > low
    base["ph"] = 12  # far from neutral
    assert compute_wqi(row(base)).iloc[0] > compute_wqi(row({**base, "ph": 7})).iloc[0]


def test_wqi_rejects_missing_values():
    r = row(IDEAL)
    r.loc[0, "ph"] = np.nan
    with pytest.raises(ValueError):
        compute_wqi(r)


def test_classes_are_balanced_and_ordered(df):
    wqi, labels, cutoffs = make_wqi_labels(df)
    assert cutoffs == sorted(cutoffs)
    assert set(labels) == {0, 1, 2, 3}
    assert np.bincount(labels).max() - np.bincount(labels).min() <= 4
    assert labels[wqi.values.argmin()] == 0 and labels[wqi.values.argmax()] == 3


def test_classify_scalar_and_array():
    cuts = [10.0, 20.0, 30.0]
    assert classify_wqi(5, cuts) == 0 and isinstance(classify_wqi(5, cuts), int)
    assert classify_wqi(10, cuts) == 1
    assert classify_wqi(99, cuts) == 3
    assert list(classify_wqi(np.array([1, 15, 25, 40]), cuts)) == [0, 1, 2, 3]
    assert class_name(0) == WQI_CLASSES[0]


def test_quartile_cutoffs():
    assert quartile_cutoffs(np.arange(101)) == [25.0, 50.0, 75.0]
