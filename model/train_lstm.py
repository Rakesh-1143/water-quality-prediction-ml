"""Train the LSTM next-step WQI forecaster.

By default uses a SIMULATED series (the Kaggle data has no time column). To use real data:
python model/train_lstm.py --csv my_series.csv   (columns: date + the 9 water-quality features)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACT_DIR, RANDOM_STATE
from src.artifacts import save_results
from src.pipeline_ts import run_lstm_pipeline, save_lstm
from src.timeseries import load_time_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out-dir", default=ARTIFACT_DIR, help="where to save trained files")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    ts = None
    if args.csv:
        ts, _ = load_time_series(args.csv)
    out = run_lstm_pipeline(ts=ts, seed=RANDOM_STATE, epochs=args.epochs)
    r = out["results"]
    print("Series:", "SIMULATED" if r["simulated"] else "real", f"({r['n_steps']} steps)")
    print("LSTM        :", {k: round(v, 3) for k, v in r["lstm"].items()})
    print("Persistence :", {k: round(v, 3) for k, v in r["persistence"].items()})
    print(f"Quality-level accuracy: LSTM {r['level_accuracy']:.3f} vs persistence {r['level_accuracy_persistence']:.3f}")
    save_lstm(out, args.out_dir)
    save_results("lstm", r, args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
