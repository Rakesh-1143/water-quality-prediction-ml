"""Real monitoring data: Brisbane River water-quality buoy (30,894 readings, Aug 2023 - Jun 2024).

Source: Queensland Government Open Data Portal (Colmslie site), via Kaggle, Apache-2.0.
Raw quirks handled here: unsorted rows, 280 duplicated timestamps, 3-17% missing values per
parameter, and a quality-flag column per parameter.
"""
import pandas as pd

from config import BRISBANE_INPUTS, BRISBANE_PATH

RENAME = {
    "Timestamp": "date", "Chlorophyll": "chlorophyll", "Temperature": "temperature",
    "Dissolved Oxygen": "dissolved_oxygen", "pH": "ph", "Specific Conductance": "specific_conductance",
    "Turbidity": "turbidity",
}


def load_brisbane(path=BRISBANE_PATH, freq="D", features=BRISBANE_INPUTS):
    """Clean the raw CSV and resample to `freq` ('D' daily, 'h' hourly) means.

    - values with a quality flag outside 1000-1999 (suspect/bad) are set to missing
    - rows are sorted by time; duplicated timestamps are averaged by the resampling
    - gaps are linearly interpolated (at most 3 consecutive steps for daily, 12 for hourly);
      periods with longer gaps at the start/end are dropped
    """
    raw = pd.read_csv(path)
    df = raw.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"])
    for name, col in RENAME.items():
        flag = f"{name} [quality]"
        if flag in raw.columns:
            bad = raw[flag].notna() & ~raw[flag].between(1000, 1999)
            df.loc[bad, col] = float("nan")
    daily = df.set_index("date")[list(features)].sort_index().resample(freq).mean()
    limit = 3 if freq == "D" else 12
    daily = daily.interpolate(limit=limit, limit_area="inside").dropna().reset_index()
    return daily
