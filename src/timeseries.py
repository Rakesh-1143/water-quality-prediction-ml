"""LSTM time-series forecasting of the next-step WQI (thesis abstract, time-series extension).

The Kaggle potability dataset has NO time column, so the default real series is the Brisbane
River buoy data (`src/realdata.py`). Any CSV with a `date` column plus the profile's input
features works too. `simulate_series` builds a clearly labelled SIMULATED series for tests only.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from config import FEATURES, LSTM_PARAMS, RANDOM_STATE, SIM_STEPS, SPLIT
from src.wqi import POTABILITY, compute_wqi


def simulate_series(df, n_steps=SIM_STEPS, seed=RANDOM_STATE):
    """Simulated daily series: yearly seasonality + trend + AR(1) noise, clipped to observed range."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    data = {}
    for f in FEATURES:
        col = df[f].dropna()
        mean, std = col.median(), col.std()
        noise = np.zeros(n_steps)
        for i in range(1, n_steps):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.45 * std * np.sqrt(1 - 0.81))
        season = 0.5 * std * np.sin(2 * np.pi * (t / 365.0 + rng.uniform()))
        trend = 0.3 * std * (t / n_steps - 0.5)
        data[f] = np.clip(mean + season + trend + noise, col.min(), col.max())
    out = pd.DataFrame(data)
    out.insert(0, "date", pd.date_range("2020-01-01", periods=n_steps, freq="D"))
    return out


def load_time_series(path=None, df=None, n_steps=SIM_STEPS, seed=RANDOM_STATE, features=FEATURES):
    """Read a user CSV (date + features) or fall back to the simulated series.

    Returns (series, is_simulated).
    """
    if path is None:
        return simulate_series(df, n_steps, seed), True
    return clean_series(pd.read_csv(path), features), False


def clean_series(ts, features=FEATURES):
    """Sort by date, keep date + features; fill gaps by interpolation, then by the median."""
    features = list(features)
    missing = [c for c in ["date"] + features if c not in ts.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ts = ts[["date"] + features].copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)
    ts[features] = ts[features].interpolate(limit_direction="both")
    ts[features] = ts[features].fillna(ts[features].median())
    return ts


def add_wqi(ts, profile=POTABILITY):
    ts = ts.copy()
    ts["wqi"] = compute_wqi(ts, profile)
    return ts


def chronological_split(n, ratios=SPLIT):
    """Index boundaries for train / val / test without shuffling (no future leakage)."""
    i_train = int(n * ratios[0])
    i_val = int(n * (ratios[0] + ratios[1]))
    return i_train, i_val


def segment_ids(dates, gap_factor=1.5):
    """Id of the continuous run each row belongs to.

    A new run starts after a gap longer than `gap_factor` x the typical time step, so data
    recorded before and after a sensor outage is never joined into one window.
    """
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if len(d) < 2:
        return np.zeros(len(d), dtype=int)
    diff = d.diff()
    return (diff > diff.median() * gap_factor).cumsum().to_numpy()


def make_windows(values, target, window, segments=None, return_index=False):
    """values (T, F), target (T,) -> X (n, window, F), y = target at the step after each window.

    With `segments` (see `segment_ids`), only windows whose rows, including the target step,
    all lie in the same continuous run are kept. `return_index` also returns the row index
    of each target.
    """
    if len(values) <= window:
        raise ValueError(f"Need more than {window} rows, got {len(values)}")
    idx = np.arange(len(values) - window)
    if segments is not None:
        segments = np.asarray(segments)
        idx = idx[segments[idx] == segments[idx + window]]
        if len(idx) == 0:
            raise ValueError(f"No continuous run of {window + 1} consecutive steps (data has gaps)")
    X = np.stack([values[i:i + window] for i in idx]).astype("float32")
    y = target[idx + window].astype("float32")
    return (X, y, idx + window) if return_index else (X, y)


def break_at_gaps(dates, *arrays, gap_factor=1.5):
    """Insert NaN points where the record has a gap so a line plot does not bridge it."""
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if len(d) < 2:
        return d, arrays
    diff = d.diff()
    step = diff.median()
    starts = np.flatnonzero((diff > step * gap_factor).to_numpy())
    new_d = d.copy()
    new_arrays = [a.copy() for a in arrays]
    for offset, g in enumerate(starts):
        pos = g + offset
        new_d = pd.concat([new_d.iloc[:pos], pd.Series([d[g - 1] + step]), new_d.iloc[pos:]],
                          ignore_index=True)
        new_arrays = [np.insert(a, pos, np.nan) for a in new_arrays]
    return new_d, new_arrays


def prepare_sequences(ts, window=LSTM_PARAMS["window"], ratios=SPLIT, profile=POTABILITY,
                      features=FEATURES):
    """Scale with train-only statistics and build windowed train/val/test sets.

    Inputs per step = `features` + WQI. Target = next-step WQI.
    """
    ts = add_wqi(ts, profile)
    cols = list(features) + ["wqi"]
    n = len(ts)
    i_train, i_val = chronological_split(n, ratios)
    scaler = MinMaxScaler().fit(ts[cols].iloc[:i_train])
    scaled = scaler.transform(ts[cols])
    segments = segment_ids(ts["date"])
    wqi_idx = cols.index("wqi")
    out = {"scaler": scaler, "cols": cols, "wqi_idx": wqi_idx, "n_segments": int(segments.max()) + 1}
    bounds = {"train": (0, i_train), "val": (i_train - window, i_val), "test": (i_val - window, n)}
    for name, (a, b) in bounds.items():
        X, y, idx = make_windows(scaled[a:b], scaled[a:b, wqi_idx], window, segments[a:b], return_index=True)
        out[name] = (X, y)
        if name == "test":
            out["dates_test"] = ts["date"].iloc[a + idx].reset_index(drop=True)
    return out


def inverse_wqi(scaler, wqi_idx, values):
    lo, hi = scaler.data_min_[wqi_idx], scaler.data_max_[wqi_idx]
    return np.asarray(values) * (hi - lo) + lo


def build_lstm(window, n_features, params=LSTM_PARAMS, seed=RANDOM_STATE):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(params["units"]),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]), loss="mse")
    return model


def train_lstm(model, train, val, params=LSTM_PARAMS, epochs=None, verbose=0):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                         restore_best_weights=True)
    return model.fit(*train, validation_data=val, epochs=epochs or params["epochs"],
                     batch_size=params["batch_size"], callbacks=[stop], verbose=verbose)


def to_delta(X, y, wqi_idx):
    """Target as the change from the last observed WQI (the LSTM learns the residual of persistence)."""
    return y - X[:, -1, wqi_idx]


def predict_scaled_wqi(model, X, wqi_idx):
    """Scaled next-step WQI = last observed WQI + predicted change."""
    return X[:, -1, wqi_idx] + model.predict(X, verbose=0).ravel()


def forecast_next(model, scaler, cols, recent, window=LSTM_PARAMS["window"], profile=POTABILITY, dates=None):
    """Predict the WQI of the step after the last `window` rows of `recent` (DataFrame of input features).

    If `dates` (one per row) is given, the last `window` rows must be continuous (no gap).
    """
    if len(recent) < window:
        raise ValueError(f"Need at least {window} rows, got {len(recent)}")
    if dates is not None:
        tail = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        seg = segment_ids(tail)
        if seg[-window:].min() != seg[-window:].max():
            raise ValueError(f"The last {window} rows contain a gap in time; provide {window} consecutive steps")
    features = [c for c in cols if c != "wqi"]
    recent = add_wqi(recent.iloc[-window:][features], profile)
    x = scaler.transform(recent[cols])[None].astype("float32")
    pred = predict_scaled_wqi(model, x, cols.index("wqi"))
    return float(inverse_wqi(scaler, cols.index("wqi"), pred)[0])
