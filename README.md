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
git clone https://github.com/Rakesh-1143/water-quality-prediction-ml.git
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
# Full suite adds tuning, five split repetitions, final-hybrid SHAP
python model/train_model.py --tune --full
streamlit run app/app.py
```

A quick integration smoke run is `python model/train_model.py --epochs 2
--output /tmp/wqi-smoke-model --reports /tmp/wqi-smoke-reports`. It verifies
execution only and is not evidence of paper accuracy.

Generated models live in `model/paper/`; measured results in `reports/`.
The prediction page requires all five matching new artifact files and never
loads the original 12-feature pickles. Dataset EDA is available before training.

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
