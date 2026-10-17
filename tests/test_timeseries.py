import io

import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from src.pipeline_ts import load_lstm, save_lstm
from src.timeseries import (add_wqi, chronological_split, clean_series, forecast_next,
                            load_time_series, make_windows, prepare_sequences, simulate_series,
                            to_delta)


def test_simulated_series_is_deterministic_and_within_observed_range(df):
    a, b = simulate_series(df, 200, seed=1), simulate_series(df, 200, seed=1)
    assert a.equals(b) and len(a) == 200
    assert a["date"].is_monotonic_increasing
    for f in FEATURES:
        assert a[f].between(df[f].min(), df[f].max()).all()
    assert not a[FEATURES].isna().any().any()


def test_load_time_series_defaults_to_simulated(df):
    ts, simulated = load_time_series(df=df, n_steps=50)
    assert simulated and len(ts) == 50


def test_clean_series_sorts_interpolates_and_validates(df):
    ts = simulate_series(df, 30).sample(frac=1, random_state=0)  # shuffled
    ts.iloc[3, 1] = np.nan
    out = clean_series(ts)
    assert out["date"].is_monotonic_increasing and not out[FEATURES].isna().any().any()
    with pytest.raises(ValueError):
        clean_series(ts.drop(columns=["ph"]))


def test_make_windows_alignment():
    values = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float)
    X, y = make_windows(values, target, window=3)
    assert X.shape == (7, 3, 2) and y.shape == (7,)
    assert X[0][:, 0].tolist() == [0, 2, 4] and y[0] == 3      # target is the step AFTER the window
    assert X[-1][-1, 0] == 16 and y[-1] == 9
    with pytest.raises(ValueError):
        make_windows(values[:3], target[:3], window=3)


def test_chronological_split_has_no_overlap():
    i_train, i_val = chronological_split(1000)
    assert (i_train, i_val) == (700, 850)


def test_prepare_sequences_no_future_leakage(df):
    ts = simulate_series(df, 400)
    seq = prepare_sequences(ts, window=10)
    scaler = seq["scaler"]
    train_part = add_wqi(ts).iloc[:280][seq["cols"]]
    assert scaler.data_min_ == pytest.approx(train_part.min().values)   # fitted on train period only
    n_in = len(ts) - 10
    sizes = [len(seq[k][0]) for k in ("train", "val", "test")]
    assert sum(sizes) == n_in
    assert seq["train"][0].shape[1:] == (10, len(FEATURES) + 1)
    assert len(seq["dates_test"]) == 400 - 340


def test_to_delta_is_change_from_last_observation():
    X = np.zeros((2, 3, 2), dtype="float32")
    X[:, -1, 1] = [0.2, 0.5]
    assert to_delta(X, np.array([0.3, 0.4], dtype="float32"), 1) == pytest.approx([0.1, -0.1])


def test_lstm_pipeline_results(lstm_run):
    r = lstm_run["results"]
    assert r["simulated"] is True
    for k in ("lstm", "persistence"):
        assert set(r[k]) == {"rmse", "mae", "r2"} and r[k]["rmse"] >= 0
    assert len(r["test_true"]) == len(r["test_pred"]) == len(r["test_dates"]) > 0
    assert 0 <= r["level_accuracy"] <= 1


def test_lstm_save_load_and_forecast(lstm_run, df, tmp_path):
    save_lstm(lstm_run, str(tmp_path))
    b = load_lstm(str(tmp_path))
    ts = simulate_series(df, 60)
    v1 = forecast_next(b["model"], b["scaler"], b["cols"], ts[FEATURES], b["window"])
    v2 = forecast_next(lstm_run["model"], lstm_run["scaler"], lstm_run["cols"], ts[FEATURES], b["window"])
    assert np.isfinite(v1) and v1 == pytest.approx(v2, abs=1e-3)
    assert 50 < v1 < 400  # plausible WQI range for this dataset
    with pytest.raises(ValueError):
        forecast_next(b["model"], b["scaler"], b["cols"], ts[FEATURES].head(3), b["window"])


def test_csv_upload_path_roundtrip(df):
    csv = simulate_series(df, 40).to_csv(index=False)
    ts = clean_series(pd.read_csv(io.StringIO(csv)))
    assert len(ts) == 40 and pd.api.types.is_datetime64_any_dtype(ts["date"])
