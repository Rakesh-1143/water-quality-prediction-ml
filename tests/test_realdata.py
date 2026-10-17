import numpy as np
import pandas as pd
import pytest

from config import BRISBANE_INPUTS, BRISBANE_PATH
from src.realdata import load_brisbane
from src.timeseries import segment_ids
from src.wqi import BRISBANE, compute_wqi


def raw_row(ts, ph=8.0, flag=None):
    row = {"Timestamp": ts, "Record number": 1, "Average Water Speed": 1.0, "Average Water Direction": 1.0,
           "Chlorophyll": 2.0, "Chlorophyll [quality]": 1020, "Temperature": 20.0, "Temperature [quality]": 1020,
           "Dissolved Oxygen": 6.0, "Dissolved Oxygen [quality]": 1020, "pH": ph, "pH [quality]": flag or 1020,
           "Specific Conductance": 40.0, "Specific Conductance [quality]": 1020,
           "Turbidity": 3.0, "Turbidity [quality]": 1020}
    return row


def test_real_dataset_file_is_the_brisbane_buoy_data():
    raw = pd.read_csv(BRISBANE_PATH)
    assert raw.shape == (30894, 20)
    assert pd.to_datetime(raw["Timestamp"]).min() == pd.Timestamp("2023-08-04 23:00")


@pytest.mark.parametrize("freq,min_rows", [("h", 5000), ("D", 250)])
def test_load_brisbane_is_clean(freq, min_rows):
    ts = load_brisbane(freq=freq)
    assert len(ts) >= min_rows
    assert list(ts.columns) == ["date"] + BRISBANE_INPUTS
    assert ts["date"].is_monotonic_increasing and ts["date"].is_unique
    assert not ts[BRISBANE_INPUTS].isna().any().any()
    assert ts["ph"].between(7, 9).all() and ts["dissolved_oxygen"].between(0, 15).all()
    assert (ts["turbidity"] >= 0).all() and (ts["chlorophyll"] >= 0).all()
    step = ts["date"].diff().dropna().min()
    assert step == pd.Timedelta(hours=1 if freq == "h" else 24)


def test_loader_sorts_averages_duplicates_and_masks_flagged_values(tmp_path):
    rows = [raw_row("2024-01-01 02:00:00"), raw_row("2024-01-01 00:00:00", ph=8.0),
            raw_row("2024-01-01 00:00:00", ph=7.0),            # duplicate timestamp -> averaged
            raw_row("2024-01-01 01:00:00", ph=99.0, flag=3000)]  # bad flag -> masked, interpolated
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    ts = load_brisbane(path, freq="h")
    assert len(ts) == 3 and ts["date"].is_monotonic_increasing
    assert ts["ph"].iloc[0] == pytest.approx(7.5)              # mean of the two duplicates
    assert ts["ph"].iloc[1] == pytest.approx((7.5 + 8.0) / 2)  # 99.0 removed, linear interpolation
    assert ts["ph"].max() < 9


def test_brisbane_wqi_profile():
    assert sum(BRISBANE.weights.values()) == pytest.approx(1.0)
    at_limits = pd.DataFrame([BRISBANE.standards])
    assert compute_wqi(at_limits, BRISBANE).iloc[0] == pytest.approx(100.0)
    ideal = pd.DataFrame([BRISBANE.ideal])
    assert compute_wqi(ideal, BRISBANE).iloc[0] == pytest.approx(0.0)
    good, bad = pd.DataFrame([BRISBANE.ideal]), pd.DataFrame([BRISBANE.ideal])
    bad["dissolved_oxygen"] = 3.0                              # low oxygen is worse
    good["dissolved_oxygen"] = 8.0
    assert compute_wqi(bad, BRISBANE).iloc[0] > compute_wqi(good, BRISBANE).iloc[0]


def test_lstm_pipeline_on_real_data(brisbane_run):
    r = brisbane_run["results"]
    assert r["simulated"] is False and r["dataset"] == "brisbane" and r["profile"] == "brisbane"
    assert r["features"] == BRISBANE_INPUTS and r["freq"] == "h" and r["window"] == 24
    assert len(r["test_true"]) == len(r["test_pred"]) == len(r["test_persistence"]) == r["n_test"] > 500
    assert np.isfinite(list(r["lstm"].values())).all()
    assert 0 <= r["level_accuracy"] <= 1


def test_real_series_has_gaps_that_windows_must_not_bridge():
    ts = load_brisbane(freq="h")
    seg = segment_ids(ts["date"])
    assert seg.max() + 1 > 10                                  # many outages in the sensor record
    assert ts["date"].diff().max() > pd.Timedelta(days=20)    # incl. a month-long one (Apr-May 2024)


def test_real_pipeline_windows_are_continuous_and_dates_align(brisbane_run):
    r = brisbane_run["results"]
    assert r["n_segments"] > 10 and len(r["test_dates"]) == r["n_test"]
    dates = pd.to_datetime(r["test_dates"])
    assert dates.is_monotonic_increasing
    ts = load_brisbane(freq="h").set_index("date")
    for t in dates[::50]:                                      # sampled: the 24 h before each target are all present
        window = ts.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(hours=1)]
        assert len(window) == 24
