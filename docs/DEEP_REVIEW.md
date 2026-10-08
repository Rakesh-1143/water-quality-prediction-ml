# Deep code review — 8 October 2026

The current source and bundled binary models passed the checks below after four confirmed issues were fixed. This is evidence of the exercised software paths on Linux/Python 3.12, not a guarantee that every possible input/platform works or that the IEEE paper has been reproduced.

## Confirmed issues fixed

| Finding | Before | After / evidence |
|---|---|---|
| Extreme finite measurement values | Values such as 1e300 overflowed float32; inference returned NaN and incorrectly assigned Nonpotable | Validate transformed float32 inputs and all probability outputs; reject the prediction. Regression tests cover overflow, NaN and probabilities outside [0,1] |
| Duplicate CSV headers | Pandas silently renamed the second `ph` header; the prediction used the first | Shared CSV reader validates raw headers before Pandas, including empty headers and UTF-8 BOM. Used by training-data loading, prediction CLIs and upload pages |
| Single-class DANN target holdout | Training validation required both classes and eight samples/class even for a separate evaluation set | Evaluation accepts complete 0/1 labels in nonempty data, including three rows of one class; training requirements remain strict. Fixed class vocabulary is used for macro-F1, log loss and confusion matrix |
| EDA rerun resources and class label order | Matplotlib figures stayed open; labels assumed frequency order | Close every rendered figure; explicitly order class counts as 0 then 1 |

Nullable numeric missing measurements also now normalize to float64 and correctly use training medians. Future inference rejects nonfinite forecasts and invalid class probability matrices.

## Verification performed

| Area | Result |
|---|---|
| Full automated suite | **19 tests passed**; see [DEEP_REVIEW_TESTS.txt](DEEP_REVIEW_TESTS.txt) |
| Binary training | Integration test trains the complete binary experiment, saves artifacts and reloads predictions; pinned dependencies have no conflicts |
| Bundled paper snapshot | 492 held-out rows verified; split disjointness, coverage, labels, all recorded model metrics, dataset hash and hybrid reload checked; maximum probability discrepancy 2.96e-8 |
| Bundled abstract snapshot | Same checks on 492 held-out rows; maximum probability discrepancy 1.11e-16 |
| Streamlit pages | Home, EDA, Prediction, Experiments and Future Work run with AppTest without uncaught exceptions; both binary model selections and forms exercised |
| CSV application flow | Actual page handlers exercised with mocked uploaded file objects: valid binary CSV, invalid duplicate-header CSV, and saved multiclass future-model CSV |
| Streamlit server | Server started in a subprocess; `/_stcore/health` returned `ok`; process terminated after check |
| Prediction CLI | Bundled `examples/demo_measurements.csv` exported predictions using the saved abstract model |
| EDA notebook | All 13 code cells executed with the bundled 3,276-row dataset using the Agg backend |
| Multiclass | Temporary synthetic fixture trains ANN/XGBoost/RF, selects fusion using validation and reproduces saved probabilities after reload |
| LSTM | Temporary fixture verifies disjoint chronological windows, horizon/interval validation, future timestamp and saved held-out forecast replay |
| Transfer | Temporary fixture trains frozen head then fine-tunes, reloads probabilities and checks unchanged source preprocessing bytes |
| DANN | Temporary fixture checks reversed gradient, serialization, unlabeled adaptation, independent small single-class target evaluation and model reload |
| Source consistency | Python files parsed; `git diff --check` passed |

Training fixtures are synthetic software checks only. They are deleted after tests and do not constitute measured future-work research results. CSV file chooser interaction and browser visual layout were not manually tested; page handler and rendering coverage uses Streamlit AppTest. Windows/macOS execution was not tested.

## Research readiness and remaining limitations

The abstract snapshot accuracy remains **67.07%** (330/492); the paper attempt remains **66.06%**. The paper's reported **86.9%** is not reproduced. The binary ANN/potability task is not a measured continuous WQI regression experiment. The paper's feature-vector expansion is a documented interpretation; author's implementation is unavailable here.

Genuine multiclass WQI labels, site time-series WQI observations, and independent regional adaptation/evaluation datasets remain necessary before claiming completed future-work experiments. Classification currently assumes independent rows and uses stratified splits; it does not implement station-grouped splitting. Numeric range warnings do not establish physical validity or drinking-water safety. Probability calibration is unverified. ANN performance and low potable-class recall limit the binary research result even though code executes.

Original legacy artifacts/demo video remain for provenance. The current app and CLIs use schema-1 snapshots in `model/abstract` or `model/paper`, and separate schema-2 future-work artifacts. Use the updated README commands and results when presenting to the guide.
