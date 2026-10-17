import numpy as np
import pandas as pd
import pytest

from config import FEATURES, TARGET
from src.data import Preprocessor, prepare, split_data


def test_dataset_matches_paper_table_ii(df):
    assert df.shape == (3276, 10)
    assert df["ph"].isna().sum() == 491
    assert df["Sulfate"].isna().sum() == 781
    assert df["Trihalomethanes"].isna().sum() == 162
    assert df[TARGET].mean() == pytest.approx(0.39, abs=0.005)


def test_split_is_70_15_15_stratified_and_disjoint(df):
    s = split_data(df[FEATURES], df[TARGET])
    n = len(df)
    assert len(s.X_train) == pytest.approx(0.70 * n, abs=2)
    assert len(s.X_val) == pytest.approx(0.15 * n, abs=2)
    assert len(s.X_test) == pytest.approx(0.15 * n, abs=2)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == n
    assert not (set(s.X_train.index) & set(s.X_test.index))
    assert not (set(s.X_train.index) & set(s.X_val.index))
    assert not (set(s.X_val.index) & set(s.X_test.index))
    for y in (s.y_train, s.y_val, s.y_test):
        assert y.mean() == pytest.approx(df[TARGET].mean(), abs=0.01)


def test_split_is_reproducible(df):
    a = split_data(df[FEATURES], df[TARGET], seed=7)
    b = split_data(df[FEATURES], df[TARGET], seed=7)
    c = split_data(df[FEATURES], df[TARGET], seed=8)
    assert a.X_test.index.equals(b.X_test.index)
    assert not a.X_test.index.equals(c.X_test.index)


def test_split_rejects_bad_ratios(df):
    with pytest.raises(ValueError):
        split_data(df[FEATURES], df[TARGET], ratios=(0.5, 0.2, 0.2))


def test_preprocessor_fits_on_train_only(df):
    splits, pre, scaled = prepare(df)
    assert pre.imputer.statistics_[0] == pytest.approx(splits.X_train["ph"].median())
    imputed_train = pre.impute(splits.X_train)
    assert pre.scaler.data_min_ == pytest.approx(imputed_train.min().values)
    assert pre.scaler.data_max_ == pytest.approx(imputed_train.max().values)


def test_scaled_output_is_clean_and_in_unit_range(df):
    _, _, scaled = prepare(df)
    for part in scaled.values():
        assert part.dtype == np.float32 and part.shape[1] == len(FEATURES)
        assert not np.isnan(part).any()
        assert part.min() >= 0.0 and part.max() <= 1.0
    assert scaled["train"].min(axis=0) == pytest.approx(0.0)
    assert scaled["train"].max(axis=0) == pytest.approx(1.0)


def test_missing_values_filled_with_median():
    X = pd.DataFrame({f: [1.0, 2.0, 3.0, np.nan] for f in FEATURES})
    pre = Preprocessor().fit(X)
    assert pre.impute(X).iloc[3, 0] == 2.0
