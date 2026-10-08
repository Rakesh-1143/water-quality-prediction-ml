# Future-work code and real-data requirements

These experimental training and inference workflows are implemented. They are **not validated
water-quality research results**: no genuine multiclass WQI, time-series or target-region dataset
has been supplied. Automated tests create temporary synthetic fixtures solely to check execution,
split controls, gradients and model reload. Fixture models and scores are not published as experiments.

The binary MTech abstract model and original paper attempt remain unchanged. The future-work
modules are separate and do not convert Potability into invented WQI classes or timestamps.
The Streamlit Future Work page shows these contracts and can load runs saved inside
`experiments/<run-name>` for CSV inference. Training runs through the CLI.

## Dataset contracts

All inputs use these nine measurement columns:
`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`.
Blank feature cells are median-imputed using the training/source preprocessing. Infinite values
are rejected. Keep feature units and meaning compatible across datasets.

| Workflow | Additional data | Implemented method |
|---|---|---|
| Multiclass | `WQI_class`: at least three genuine quality labels; >=8 rows/class | Training-only Tree SHAP; feature-weighted 16/8 ANN with softmax; multiclass XGBoost; validation-selected probability average; RF comparison |
| Time series | `timestamp`, numeric `WQI`; one monitoring site, chronological unique equally spaced observations | LSTM forecasting of continuous WQI; training-only input/target scaling; disjoint chronological windows; persistence comparison |
| Transfer | Target-region measurements and binary `Potability`; a compatible saved source ANN | Freeze feature layers, train output, then unfreeze and fine-tune at lower learning rate; validation chooses stage; source zero-shot comparison |
| Domain adaptation | Labeled source, unlabeled target adaptation measurements; optional separate labeled target test | DANN shared encoder, binary label head, domain head and gradient reversal; class loss uses source labels only |

WQI class definitions and index calculations must come from a justified method agreed with the
guide. This software accepts verified targets; it does not supply an arbitrary WQI formula.
The classification split is stratified 70/15/15. It assumes independent rows; for repeated
observations from the same station, prepare station-disjoint input partitions/protocol before
claiming geographic generalization. The current classification runners do not implement grouped
station splitting. DANN rejects identical measurement rows across source/adaptation/evaluation
inputs, but this alone does not establish regional independence or correct provenance.

## Commands

Run from the project root after installing `requirements.txt`. Paths below represent datasets you
must supply; they are not bundled files. Use a new empty output directory for every training run.

```bash
python model/future_work.py multiclass --data datasets/wqi_classes.csv --output experiments/multiclass
python model/future_work.py timeseries --data datasets/site_history.csv --lookback 12 --horizon 1 --output experiments/lstm
python model/future_work.py transfer --data datasets/target_region.csv --source-model model/abstract --output experiments/transfer
python model/future_work.py dann --source datasets/source_region.csv --target-adapt datasets/target_unlabeled.csv --target-test datasets/target_holdout.csv --output experiments/dann
python model/future_work.py predict --model experiments/multiclass --data datasets/new_measurements.csv --output predictions.csv
```

`--epochs` defaults to 100; early stopping uses validation data. `--seed` defaults to 42.
Multiclass supports `--label`; time series supports `--label` and `--timestamp`. The LSTM
`--horizon` is measured in sampling intervals. For inference, supply at least `lookback`
consecutive rows at the training sampling interval: the output is one forecast `horizon`
intervals after the final input timestamp. This is forecasting, not multiclass prediction.

Transfer keeps the original source preprocessing, records the source model hash, and rejects
target rows identical to the bundled source dataset when the source metadata identifies it.
Transferred inference is the ANN; it does not silently reuse an unadapted XGBoost hybrid.
DANN ignores any label columns present in the adaptation file. If `--target-test` is omitted,
the report contains **source-test scores only**, not a claim of target-domain performance.

Each training output contains `model.keras`, `preprocess.joblib`, `metadata.json`,
`evaluation.json`, and held-out `test_predictions.csv`. Multiclass also saves `xgboost.json`;
DANN with target test data also saves `target_test_predictions.csv`.
Use `FuturePredictor` or the CLI to reload; these experimental schema-2 artifacts are distinct
from the existing binary hybrid schema-1 artifacts.

## Verification and references

```bash
python -m unittest discover -s tests -v
python model/future_work.py --help
```

Tests exercise all four training paths and reload their predictions. They verify chronological
forecast boundaries/intervals, target-overlap rejection, multiclass requirements, unchanged source
preprocessing and the negative gradient of the reversal layer. The test log is
`docs/FUTURE_WORK_TESTS.txt`. It is software evidence, not a real-data accuracy report.

- Keras initializers: https://keras.io/api/layers/initializers/
- XGBoost multiclass objective: https://xgboost.readthedocs.io/en/stable/parameter.html
- Keras LSTM: https://keras.io/api/layers/recurrent_layers/lstm/
- Keras transfer learning: https://keras.io/guides/transfer_learning/
- TensorFlow custom gradients: https://www.tensorflow.org/api_docs/python/tf/custom_gradient
- Ganin et al., Domain-Adversarial Training of Neural Networks (2016): https://jmlr.org/papers/v17/15-239.html

Before guide submission, obtain the missing datasets and run these commands. Interpret accuracy,
macro-F1, forecast MAE/RMSE and target-domain performance only from those measured real-data runs.
