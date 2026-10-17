"""Data loading, leakage-free preprocessing and stratified 70/15/15 split."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PATH, FEATURES, RANDOM_STATE, SPLIT, TARGET


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


class Preprocessor:
    """Median imputation + min-max scaling to [0, 1], fitted on training data only."""

    def __init__(self, features=FEATURES):
        self.features = list(features)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler(clip=True)

    def fit(self, X):
        imputed = self.imputer.fit_transform(X[self.features])
        self.scaler.fit(imputed)
        return self

    def impute(self, X):
        """Raw features with missing values filled by the training medians."""
        return pd.DataFrame(self.imputer.transform(X[self.features]),
                            columns=self.features, index=X.index)

    def transform(self, X):
        return self.scaler.transform(self.impute(X).values).astype("float32")


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(X, y, seed=RANDOM_STATE, ratios=SPLIT):
    """Stratified train/validation/test split."""
    train, val, test = ratios
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("Split ratios must sum to 1")
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=val + test, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=test / (val + test), stratify=y_rest, random_state=seed)
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def prepare(df=None, seed=RANDOM_STATE, target=None, y=None):
    """Split, then fit the preprocessor on the training part only.

    Returns (splits, preprocessor, scaled) where scaled maps train/val/test to arrays.
    `y` can be given to stratify on a derived label (e.g. WQI class).
    """
    df = load_dataset() if df is None else df
    X = df[FEATURES]
    y = df[target or TARGET] if y is None else y
    splits = split_data(X, y, seed=seed)
    pre = Preprocessor().fit(splits.X_train)
    scaled = {k: pre.transform(getattr(splits, f"X_{k}")) for k in ("train", "val", "test")}
    return splits, pre, scaled
