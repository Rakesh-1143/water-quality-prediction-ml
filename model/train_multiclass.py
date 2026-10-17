"""Train the multiclass WQI model (4 levels): SHAP-initialised ANN + XGBoost.

Usage: python model/train_multiclass.py [--epochs N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACT_DIR, RANDOM_STATE
from src.artifacts import save_results
from src.pipeline import run_multiclass_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=ARTIFACT_DIR, help="where to save trained files")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    out = run_multiclass_pipeline(seed=RANDOM_STATE, epochs=args.epochs)
    r = out["results"]
    print("Class counts:", r["class_counts"], "| WQI cut-offs:", [round(c, 1) for c in r["cutoffs"]])
    for name in ("ann_shap_only", "xgb_only", "hybrid"):
        m = r[name]
        print(f"{name:<14} acc {m['accuracy']:.3f}  macro-F1 {m['f1_macro']:.3f}  AUC(ovr) {m['roc_auc_ovr']:.3f}")
    out["bundle"].save(args.out_dir)
    save_results("wqi", r, args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
