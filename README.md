# SHAP-initialized ANN + XGBoost Water Potability Prediction

Independent implementation of the methods in:
**A Hybrid Machine Learning Framework for Water Quality Index Prediction Using
Feature-Based Neural Network Initialization** (2026),
DOI: [10.1109/JSTARS.2026.3654017](https://doi.org/10.1109/JSTARS.2026.3654017).

The original nine water measurements feed a SHAP-initialized 16/8-neuron ANN
and an XGBoost classifier. Their probabilities are combined by weighted
averaging or validation-trained logistic regression. Future work is excluded.

## Setup (Python 3.10–3.12)

```bash
git clone --branch paper-alignment https://github.com/Rakesh-1143/water-quality-prediction-ml.git
cd water-quality-prediction-ml
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Experiments and application

```bash
# Table VIII parameters; baselines, fusion, ablations, perturbation and plots
python model/train_model.py
# Full suite using the paper's published Table VIII settings
python model/train_model.py --full
python model/verify_artifacts.py
# Optional exploratory tuning; may select settings different from Table VIII
python model/train_model.py --tune --full
streamlit run app/app.py
```

A quick integration smoke run is `python model/train_model.py --epochs 2
--output /tmp/wqi-smoke-model --reports /tmp/wqi-smoke-reports`. It verifies
execution only and is not evidence of paper accuracy.

Generated models live in `model/paper/`; measured results in `reports/`.
The prediction page requires all five matching new artifact files and never
loads the original 12-feature pickles. Dataset EDA is available before training.

`requirements-lock.txt` records the complete validated Python 3.12 Linux CPU environment.
Use `python -m pip install -r requirements-lock.txt` to reproduce that environment.
The main requirements remain the supported direct dependencies for Python 3.10–3.12.
`model/verify_artifacts.py` independently checks split disjointness, dataset hash,
saved predictions, recalculated metrics, and saved-model inference agreement.

## Verified snapshot

The committed model and reports are a fixed-settings reproduction attempt
(seed 42, 70/15/15 split). Holdout accuracy is **66.06%**, F1 **0.3927**,
and ROC-AUC **0.6519**. The paper's numerical results were **not reproduced**.
Eight tests pass, including model reload and Streamlit pages. Full five-split
sensitivity and final-hybrid SHAP were generated. Read the
[validation report](docs/VALIDATION.md) and [raw metrics](reports/metrics.csv)
before using these results in a thesis.

## What is measured

* Original Kaggle Water Potability dataset: 3276 rows, nine predictors and
  binary Potability label; median imputation and training-fit MinMax scaling.
* Stratified 70/15/15 train/validation/test partitions, no SMOTE.
* Tree SHAP feature weights, SHAP ANN, Xavier ANN, RF, RBF SVR and XGBoost.
* Both averaging (alpha .3–.7) and logistic stacking; validation-only selection.
* Accuracy, precision, recall, F1, AUC, RMSE, MAE, R², confusion matrices.
* Ablations, +/-10% feature perturbations, five-seed split sensitivity,
  pre/post importance alignment and plots.

Read [paper alignment](docs/PAPER_ALIGNMENT.md) for exact mapping, ambiguities,
software version differences and limitations. The paper's reported accuracy
86.9%, F1 .849 and AUC .894 are **reference results**, never fabricated measured
scores. Exact reproduction and equal accuracy are not guaranteed.

This is an educational binary classifier, not a laboratory water safety
certification or a continuous physical WQI estimator.

## Credits and license

Original repository by Hassan Ali:
https://github.com/hassan-ali786/water-quality-prediction-ml

Dataset by Aditya Kadiwal:
https://www.kaggle.com/datasets/adityakadiwal/water-potability

Original MIT copyright notice is retained in [LICENSE](LICENSE).
