# Abstract implementation: measured results

Validated on 2026-10-08, Linux CPU / Python 3.12.

Feature initialization: `feature_glorot`. Selected fusion: `average_0.5`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.6687 | 0.6436 | 0.3385 | 0.4437 | 0.6620 |
| svr | 0.6748 | 0.7051 | 0.2865 | 0.4074 | 0.6506 |
| ann_xavier | 0.6443 | 0.6076 | 0.2500 | 0.3542 | 0.6532 |
| xgboost | 0.6443 | 0.5739 | 0.3438 | 0.4300 | 0.6516 |
| ann_shap | 0.6687 | 0.6790 | 0.2865 | 0.4029 | 0.6508 |
| hybrid_average_0.5 | 0.6707 | 0.6829 | 0.2917 | 0.4088 | 0.6690 |
| hybrid_stacking | 0.6626 | 0.6413 | 0.3073 | 0.4155 | 0.6671 |
| hybrid_xavier_stacking | 0.6606 | 0.6374 | 0.3021 | 0.4099 | 0.6636 |
| hybrid_selected | 0.6707 | 0.6829 | 0.2917 | 0.4088 | 0.6690 |

Hybrid confusion matrix (actual rows 0/1, predicted columns 0/1): `[[274, 26], [136, 56]]`.
Accuracy calculation: (274 + 56) / 492 = 67.07%.
RMSE: 0.460327; MAE: 0.428909; R-squared: 0.109485.

## Interpretation

The hybrid is executable and uses both model probabilities. On this holdout it does not outperform every baseline: SVR has higher accuracy. Potable-class recall remains low. These are limitations for discussion with the guide, not hidden errors.
Five seeds 42-46: mean accuracy 67.03%; population SD 1.17 percentage points.

The earlier paper interpretation measured 66.06% accuracy. This abstract variant measures 67.07%; neither reproduces the published 86.9% reference score. The variant was defined from the abstract; no test-set-driven parameter search was used. A single holdout difference is not a statistically established improvement.

## Completed verification

- Nine tests pass: core preprocessing/splits/fusion, both initialization methods, saved model reload, batched inference, and Streamlit pages for both snapshots.
- Independent verification checks all 492 test rows, nine result columns, saved hybrid predictions, partition sizes, fusion settings and artifact hashes.
- Full five-seed sensitivity and final-hybrid SHAP were generated.
- CLI sample predictions were executed successfully; the example is a demonstration, not an accuracy benchmark.

Evidence: [evaluation](../reports/abstract/evaluation.json), [verification](../reports/abstract/verification.json), [five seeds](../reports/abstract/split_sensitivity.json), [test log](../reports/abstract/test-results.txt), [sample predictions](../reports/abstract/demo_predictions.csv).

For the requirement mapping and explicit design choice, see [abstract implementation](ABSTRACT_IMPLEMENTATION.md). For laptop steps, see [guide demo](GUIDE_DEMO.md).
