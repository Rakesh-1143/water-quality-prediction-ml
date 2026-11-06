"""Create the synthetic demo dataset (NOT real data).

python model/make_synthetic_dataset.py [--n 5000] [--seed 7] [--label-noise 0.10]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR
from src.synthetic import DEFAULTS, generate, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULTS["n"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--label-noise", type=float, default=DEFAULTS["label_noise"],
                   help="share of labels flipped at random; sets the accuracy ceiling (1 - noise)")
    p.add_argument("--potable-rate", type=float, default=DEFAULTS["potable_rate"])
    p.add_argument("--out", default=os.path.join(BASE_DIR, "data", "synthetic_water_quality.csv"))
    a = p.parse_args()
    df, meta = generate(a.n, a.seed, a.potable_rate, a.label_noise)
    save(df, meta, a.out)
    print(f"Wrote {len(df)} rows to {a.out}")
    print(f"potable share {df['Potability'].mean():.3f}; label noise {meta['measured_noise']:.3f}; "
          f"maximum possible accuracy {meta['max_possible_accuracy']:.3f}")


if __name__ == "__main__":
    main()
