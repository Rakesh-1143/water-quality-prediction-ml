# Verified reproduction attempt

Validation date: 2026-10-08. Future work excluded.

**The implementation runs successfully, but the paper's numerical results were not reproduced. Exact 100% equivalence cannot be established from the published details.**

## Measured holdout results

| Metric | Paper Table IX | This run |
|---|---:|---:|
| accuracy | 0.869 | 0.660569 |
| precision | 0.857 | 0.650602 |
| recall | 0.842 | 0.281250 |
| f1 | 0.849 | 0.392727 |
| roc_auc | 0.894 | 0.651858 |
| rmse | 0.244 | 0.468251 |
| mae | 0.178 | 0.441463 |

Accuracy gap: -20.84 percentage points. This is a result difference, not a code-match percentage.

## Evidence and settings

- Dataset: 3276 rows; nine predictors. SHA256 `111a9ba65f2d791003ac07b54b8e18e2f7e0bc1cfa37cfdb251d3758a4e8c24c`.
- Partition sizes: {'train': 2293, 'validation': 491, 'test': 492}; no overlap and all original rows accounted for.
- Seed: 42; selected fusion `average_0.4` using validation only.
- Published Table VIII ANN/XGBoost settings; no exploratory tuning used in this snapshot.
- Five split seeds: [42, 43, 44, 45, 46]; population accuracy SD 0.009671, F1 SD 0.030917.
- All 9 model metrics recomputed from saved test predictions within float32 precision; saved-model reload agrees within 1e-6.
- Final-hybrid Kernel SHAP generated for 32 held-out rows, 10 training background clusters and 512 samples.
- Unit, model and Streamlit integration suite: see `reports/test-results.txt`.

## Fixes validated

- Run seed is used for ANN data shuffling, including split repetitions.
- Prediction uses training-fitted median imputation for missing measurements and rejects infinities.
- CSV predictions use 17-digit export precision for subsequent runs.
- Added independent artifact verification and a complete environment lock.
- App instructions distinguish published fixed settings from optional tuning.

## Important limitations

- SHAP-initialized ANN predicts only the nonpotable class at threshold 0.5 in this seed. Its low performance is retained in the ablation evidence.
- The paper gives nine initialization weights without the exact 9x16 matrix. Repeating each feature weight across hidden units is documented; it is not verified author code.
- Exact author splits, seeds, grid ranges, fusion choice and full software environment are unavailable in the paper. Current validated dependencies differ from the reported historical environment.
- No test-set-driven tuning, relabeling or insertion of reference scores into measured reports was used.
- This evidence supports an executable independent reproduction attempt. It does not support claiming 86.9% measured accuracy or exact reproduction.

## Inspect or reproduce

- `python -m pip install -r requirements-lock.txt` (validated Python 3.12 environment)
- `python -m unittest discover -s tests -v`
- `python model/train_model.py --full`
- `python model/verify_artifacts.py`
- `streamlit run app/app.py`

Raw evidence: [metrics](../reports/metrics.csv), [evaluation](../reports/evaluation.json), [predictions](../reports/test_predictions.csv), [verification](../reports/verification.json), [five splits](../reports/split_sensitivity.json).

Paper reference: DOI 10.1109/JSTARS.2026.3654017, Tables VIII–IX and Sections III–IV. See [mapping and ambiguities](PAPER_ALIGNMENT.md).
