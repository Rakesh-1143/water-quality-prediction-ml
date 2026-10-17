import io

import numpy as np
import pandas as pd
import pytest

from config import FEATURES
from src.pipeline_ts import load_lstm, save_lstm
from src.timeseries import (add_wqi, break_at_gaps, chronological_split, clean_series, forecast_next,
                            load_time_series, make_windows, prepare_sequences, segment_ids,
                            simulate_series, to_delta)


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


def with_gap(df, after_row, days=20):
    """Copy of a series where every row after `after_row` is shifted `days` later (a sensor outage)."""
    ts = df.copy()
    ts.loc[ts.index > after_row, "date"] += pd.Timedelta(days=days)
    return ts


def test_segment_ids_split_at_gaps():
    dates = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
                            "2024-01-05 00:00", "2024-01-05 01:00"])
    assert segment_ids(dates).tolist() == [0, 0, 0, 1, 1]
    assert segment_ids(dates[:1]).tolist() == [0]


def test_make_windows_never_crosses_a_gap():
    values = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float)
    segments = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X, y, idx = make_windows(values, target, 2, segments, return_index=True)
    assert idx.tolist() == [2, 3, 4, 7, 8, 9]            # no window includes rows from both runs
    assert y.tolist() == [2, 3, 4, 7, 8, 9]
    for window, t in zip(X, idx):
        assert window[:, 0].tolist() == [2 * (t - 2), 2 * (t - 1)]
    with pytest.raises(ValueError):
        make_windows(values, target, 5, np.arange(10))      # every row is its own run


def test_prepare_sequences_skips_windows_that_span_a_gap(df):
    ts = with_gap(simulate_series(df, 400), after_row=360)   # outage inside the test period
    seq = prepare_sequences(ts, window=24)
    dates = ts["date"]
    assert len(seq["test"][0]) == 36                          # 21 before the gap + 15 after, 24 skipped
    assert len(seq["dates_test"]) == 36
    assert not seq["dates_test"].between(dates[361], dates[384]).any()
    step = pd.Timedelta(days=1)
    for t in seq["dates_test"]:                               # target's window is 24 consecutive days
        assert (dates == t - 24 * step).any()
        window_dates = dates[(dates >= t - 24 * step) & (dates < t)]
        assert len(window_dates) == 24 and (window_dates.diff().dropna() == step).all()


def test_break_at_gaps_inserts_nan_without_losing_points():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    d, (a, b) = break_at_gaps(dates, [1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert len(d) == 6 and np.isnan(a[3]) and np.isnan(b[3])
    assert d[3] == pd.Timestamp("2024-01-04")
    assert a[~np.isnan(a)].tolist() == [1, 2, 3, 4, 5]
    d2, (c,) = break_at_gaps(dates[:3], [1, 2, 3])            # no gap -> unchanged
    assert len(d2) == 3 and c.tolist() == [1, 2, 3]


def test_forecast_rejects_a_window_with_a_gap(lstm_run, df):
    ts = with_gap(simulate_series(df, 60), after_row=50)     # gap inside the last 14 rows
    args = (lstm_run["model"], lstm_run["scaler"], lstm_run["cols"], ts[FEATURES], 14)
    with pytest.raises(ValueError, match="gap"):
        forecast_next(*args, dates=ts["date"])
    ok = simulate_series(df, 60)
    assert np.isfinite(forecast_next(*args, dates=ok["date"]))
