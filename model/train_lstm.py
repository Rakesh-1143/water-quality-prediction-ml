"""Train the LSTM next-step WQI forecaster on REAL time-series data.

Default: Brisbane River buoy data (data/brisbane_river_water_quality.csv), hourly.
  python model/train_lstm.py                       # Brisbane, hourly
  python model/train_lstm.py --freq D              # Brisbane, daily (only ~300 steps)
  python model/train_lstm.py --csv my.csv --profile potability   # your data: date + the 9 features
  python model/train_lstm.py --dataset simulated   # simulated series (testing only)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACT_DIR, BRISBANE_INPUTS, FEATURES, RANDOM_STATE, TS_DEFAULT_FREQ
from src.artifacts import save_results
from src.pipeline_ts import run_lstm_pipeline, save_lstm
from src.timeseries import load_time_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["brisbane", "simulated"], default="brisbane")
    parser.add_argument("--freq", choices=["h", "D"], default=TS_DEFAULT_FREQ)
    parser.add_argument("--csv", default=None, help="your own CSV (date + input features)")
    parser.add_argument("--profile", choices=["potability", "brisbane"], default="potability",
                        help="WQI profile / feature set of --csv")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--out-dir", default=ARTIFACT_DIR, help="where to save trained files")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.csv:
        features = FEATURES if args.profile == "potability" else BRISBANE_INPUTS
        ts, _ = load_time_series(args.csv, features=features)
        out = run_lstm_pipeline(ts=ts, features=features, profile_name=args.profile, freq=args.freq,
                                seed=RANDOM_STATE, epochs=args.epochs, window=args.window, dataset="custom")
    else:
        out = run_lstm_pipeline(dataset=args.dataset, freq=args.freq, seed=RANDOM_STATE,
                                epochs=args.epochs, window=args.window)
    r = out["results"]
    print(f"Series: {r['dataset']} ({'SIMULATED' if r['simulated'] else 'real'}), {r['n_steps']} steps, "
          f"freq {r['freq']}, window {r['window']}, test windows {r['n_test']}")
    print("LSTM        :", {k: round(v, 3) for k, v in r["lstm"].items()})
    print("Persistence :", {k: round(v, 3) for k, v in r["persistence"].items()})
    print(f"Quality-level accuracy: LSTM {r['level_accuracy']:.3f} vs persistence {r['level_accuracy_persistence']:.3f}")
    save_lstm(out, args.out_dir)
    save_results("lstm", r, args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
