# Paper alignment and reproducibility

Reference: Ali Al Bataineh, Bandi Vamsi, Scott Alan Smith (2026),
A Hybrid Machine Learning Framework for Water Quality Index Prediction Using
Feature-Based Neural Network Initialization. DOI: 10.1109/JSTARS.2026.3654017.
This is an independent implementation, not author-supplied source code.
Future work (multiclass, forecasting, IoT, transfer learning) is excluded.

## Implemented mapping

| Paper | Implementation |
|---|---|
| III-B: 3276-row Kaggle Water Potability dataset, nine predictors | `wqi.core.load_data`, existing CSV; target excluded from features |
| III-B: median, IQR inspection without removal, MinMax, stratified 70/15/15 | `preprocessor`, `dataset_profile`, `split_data` |
| III-C and Algorithm 1: XGBoost regressor, Tree SHAP, mean absolute significance normalized by sum | `fit_experiment`, `shap_importance`, `input_kernel` |
| Table VIII / Algorithm 3: 9→16 ReLU→dropout .2→8 ReLU→1 sigmoid | `build_ann` |
| Adam .001, batch 32, binary cross-entropy, max 100 epochs, early stopping patience 10 | `fit_ann` |
| XGBoost 100 trees, depth 5, eta .1, subsample .8, L2 1, L1 .5 | `config.XGB_PARAMS` |
| III-E: alpha .3–.7 averaging and logistic regression stacking on validation | `fit_experiment`; both variants evaluated, validation selects deployment variant |
| III-F: RF, RBF SVR, Xavier ANN, standalone XGBoost | `fit_experiment` |
| III-F / IV-B: accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, R² | `metrics`, measured CSV/JSON, ROC and probability/loss plots |
| III-G: tuning on 20% of training data | `--tune`, explicit recorded XGBoost and ANN candidate grids |
| IV-C: initialization and boosting ablations | `ablations.json`, Xavier ANN, SHAP ANN, XGBoost, both hybrids, Xavier hybrid |
| IV-D: +/-10% pH/sulfate/conductivity perturbation; five split repeats | `feature_sensitivity.json`; `--full` adds `split_sensitivity.json` |
| IV-E: pre/post feature importance alignment | training Tree SHAP; `--full` adds Kernel SHAP on final hybrid and Spearman alignment |

## Ambiguities and explicit implementation choices

* The paper calls the dataset “10 features”, but this includes the binary target.
  Its ANN and Algorithm 3 use nine predictors. We use all nine; no manual
  interaction features, feature dropping, or SMOTE are introduced.
* The abstract describes sequential residual refinement, while III-E and
  Algorithm 2 explicitly describe independently trained ANN/XGBoost late
  fusion. We follow III-E and Algorithms 2–3, not an invented residual learner.
* Algorithm 1 provides nine normalized weights but does not specify how they
  fill a 9x16 kernel. We deterministically repeat each feature weight across
  the 16 hidden units, with zero biases and Glorot initialization in later
  layers. This is an interpretation. Identical first-layer columns may produce
  symmetric units; dropout supplies stochastic training variation. Authors'
  exact initialization cannot be confirmed without their code.
* III-B gives 70/15/15, while III-G mentions 20% of training for tuning.
  We keep the outer 70/15/15 and tune on an inner 80/20 stratified split of
  the training partition. Grids are our explicit choices because full search
  ranges are not published. Inner preprocessing/SHAP are fit on inner train.
* Paper prose/Algorithm 3 put some preprocessing before splitting and III-C
  mentions “all preprocessed data”; elsewhere it specifies training-only SHAP
  and a reserved test set. We fit preprocessing and SHAP on training only to
  avoid leakage. Test data never select weights, hyperparameters or fusion.
* Stacking is fitted on validation predictions as specified. That validation
  score is in-sample for the combiner, so it is a selection diagnostic, not an
  unbiased generalization estimate. Holdout test metrics are separate.
* Validation also drives early stopping. Final ANN uses restored best weights;
  the paper's separate instruction to retrain on full training is ambiguous
  regarding validation/early stopping. Training partition remains the outer 70%.
* The paper describes 100 epochs but plots 50. We use max 100 with early
  stopping and plot the actual completed epochs.
* SVR scores are clipped to [0,1] for binary probability-style metrics. These
  are regression scores, not calibrated probabilities. Threshold is .5.
* Final-model Kernel SHAP explains 32 held-out rows using 10 training-derived
  background clusters and 512 samples per explanation. This budget is an
  explicit computational approximation; not a claimed author setting.
* Dependencies use compatible modern versions for Python 3.10–3.12. The
  paper reports Python 3.10, TensorFlow 2.13, XGBoost 1.7.6, SHAP .41.0;
  unspecified package versions, seeds, exact splits and implementation details
  prevent claiming bit-for-bit numerical reproduction.

## Honest result reporting

No hardcoded paper scores are presented as measured results. The reference
86.9% accuracy, .849 F1 and .894 AUC are comparison values only. All output
reports contain actual predictions, row IDs, seeds, split indices, parameters
and dataset hash. The application refuses to use legacy `model.pkl` or
`scaler.pkl`. New artifacts must be trained together.

This implements the described components with documented interpretations;
it does not establish “100% exact reproduction”. Run the complete suite and
review measured evidence before claiming reproducibility or performance.

Official implementation references:
* https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
* https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
* https://scikit-learn.org/stable/common_pitfalls.html
