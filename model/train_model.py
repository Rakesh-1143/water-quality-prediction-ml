"""Train the proposed hybrid model (SHAP-initialised ANN + XGBoost) for potable / not potable.

Also trains the baselines (RF, SVR, ANN-Xavier, XGBoost) and runs the ablation and
sensitivity studies of the IEEE paper. Usage: python model/train_model.py [--skip-sensitivity]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ARTIFACT_DIR, RANDOM_STATE
from src.artifacts import save_results
from src.pipeline import perturbation_test, run_binary_pipeline, run_sensitivity


def show(title, table):
    print(f"\n{title}")
    print(f"{'Model':<32}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}{'RMSE':>7}{'MAE':>7}")
    for name, m in table.items():
        print(f"{name:<32}{m['accuracy']:7.3f}{m['precision']:7.3f}{m['recall']:7.3f}"
              f"{m['f1']:7.3f}{m['roc_auc']:7.3f}{m['rmse']:7.3f}{m['mae']:7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--out-dir", default=ARTIFACT_DIR, help="where to save trained files")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (quick runs)")
    args = parser.parse_args()

    out = run_binary_pipeline(seed=RANDOM_STATE, epochs=args.epochs)
    bundle, results = out["bundle"], out["results"]
    show("Model comparison (test set)", results["comparison"])
    show("Ablation (test set)", results["ablation"])
    show("Hybrid with stacking (test set)", {"Hybrid (stacking)": results["hybrid_stacking"]})
    print(f"\nAlpha (validation-selected): {results['alpha']}")
    print("SHAP ranking:", ", ".join(f"{k} {v:.3f}" for k, v in results["shap_importance"].items()))

    top3 = list(results["shap_importance"])[:3]
    results["perturbation"] = perturbation_test(bundle, out["splits"].X_test,
                                                out["splits"].y_test.values, top3)
    print("Perturbation (+-10% on top-3 SHAP features):", results["perturbation"])
    if not args.skip_sensitivity:
        results["sensitivity"] = run_sensitivity(epochs=args.epochs)
        s = results["sensitivity"]
        print(f"5 random splits: accuracy {s['accuracy']['mean']:.3f} +- {s['accuracy']['std']:.3f}, "
              f"F1 {s['f1']['mean']:.3f} +- {s['f1']['std']:.3f}")

    bundle.save(args.out_dir)
    save_results("binary", results, args.out_dir)
    print(f"\nSaved model and results to {args.out_dir}")


if __name__ == "__main__":
    main()
