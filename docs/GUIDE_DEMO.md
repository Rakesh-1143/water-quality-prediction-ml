# Guide demonstration

## Run on a laptop

Use Python 3.12 for the validated environment. Open a terminal in this project folder.

```bash
python -m venv venv
```

Windows PowerShell: `venv\Scripts\Activate.ps1`.
Windows Command Prompt: `venv\Scripts\activate.bat`.
Linux/macOS: `source venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

The committed models let you demonstrate prediction without retraining. Linux CPU Python 3.12
was tested; Windows execution has not been independently tested here. The complete Linux
environment is in `requirements-lock.txt`; use direct `requirements.txt` on other platforms.

## Five-minute walkthrough

1. Home: explain the goal: nine water measurements, a binary potable/nonpotable label.
2. EDA: show actual missing values, class counts, distributions and correlation. Median
   imputation and MinMax scaling are fitted on training data only.
3. Experiments: select **Abstract implementation**. Compare RF, SVR, Xavier ANN, SHAP ANN,
   XGBoost and the hybrid. Show measured metrics, loss curves and SHAP importance.
4. Prediction: enter nine measurements and click Predict. Explain class threshold 0.5,
   component probabilities and the limitations of a statistical safety prediction.
5. Batch prediction: upload `examples/demo_measurements.csv`, inspect output and download
   predictions. Blank cells use training medians; range flags identify extrapolated inputs.
6. Show `ABSTRACT_IMPLEMENTATION.md`, `ABSTRACT_RESULTS.md` and verification evidence.

## Predict without the UI

```bash
python model/predict.py examples/demo_measurements.csv --output demo-predictions.csv
```

Input needs the exact nine column names in the example. A Potability column is optional and
ignored at prediction time. Output rows retain input order and include probabilities, predicted
class, imputation count and the number of measurements outside the training range.

## Questions to answer honestly

- SHAP importance is computed from training data; it is not a causal claim about water chemistry.
- The abstract leaves initialization details open. The weighted random implementation is
  documented; the separately labeled paper interpretation remains available for comparison.
- An implemented method does not prove the paper's accuracy was reproduced or that the
  hybrid outperforms every baseline. Use the actual result tables.
- Future-work methods need additional datasets and separate evaluation.
- This is a fixed-dataset educational model; laboratory testing determines actual water safety.
