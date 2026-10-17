# Water Quality Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning project for water quality: a **hybrid framework (SHAP-initialised neural network + XGBoost)** for potability prediction, **multiclass WQI classification**, and **LSTM time-series forecasting**, with an interactive Streamlit app.

---

## Overview

This project combines machine learning, data preprocessing, feature engineering, and an interactive web interface to analyze water quality and predict potability in real time.

Implements the method of *Al Bataineh, Vamsi & Smith, "A Hybrid Machine Learning Framework for Water Quality Index Prediction Using Feature-Based Neural Network Initialization", IEEE JSTARS 19, 2026* and its multiclass / time-series extensions.

---

## Live Demo:
[Click here to view the Live Demo](https://water-quality-prediction-ml-eykh9eeqjxvj3gpzxurkmz.streamlit.app)

---

# Video Demo:

<video src="https://github.com/user-attachments/assets/f27c891d-3cc5-4558-a622-913d3e0d7797" width="100%" controls></video>

---

## Methods

| Task | Model |
|---|---|
| Potable / not potable | ANN (9-16-8-1, ReLU, dropout 0.2, Adam 1e-3) whose first-layer weights are scaled by normalised SHAP importance + XGBoost, fused by weighted average (alpha chosen on validation set from 0.3-0.7); stacking also available |
| Multiclass WQI (4 levels) | Same SHAP-initialised ANN (softmax) + XGBoost (multi:softprob) |
| Time series | LSTM (64 units) predicting the next-step WQI from the last 14 steps |

**Pipeline (no data leakage):** stratified 70/15/15 split first, then median imputation and min-max scaling fitted on the training part only. Early stopping (patience 10) on validation loss.

**SHAP initialisation:** an XGBoost model gives mean |SHAP| per feature, normalised to W_i = FI_i / sum(FI) (paper Eq. 3). Row i of the Glorot-initialised first-layer kernel is multiplied by 9 * W_i, so important features start with larger weights. The only difference to the Xavier baseline is this scaling (same seed).

## Results (test set, 15% hold-out, seed 42)

Potability (binary):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.669 | 0.641 | 0.344 | 0.447 | 0.666 | 0.464 | 0.432 |
| SVR | 0.675 | 0.705 | 0.286 | 0.407 | 0.651 | 0.488 | 0.384 |
| ANN (Xavier Init) | 0.657 | 0.717 | 0.198 | 0.310 | 0.623 | 0.472 | 0.453 |
| XGBoost (Standalone) | 0.644 | 0.571 | 0.354 | 0.437 | 0.668 | 0.467 | 0.414 |
| Proposed Hybrid Model | 0.677 | 0.694 | 0.307 | 0.426 | 0.684 | 0.459 | 0.429 |

Ablation:

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| ANN (Xavier Init) | 0.657 | 0.717 | 0.198 | 0.310 | 0.623 | 0.472 | 0.453 |
| ANN (SHAP Init) | 0.679 | 0.774 | 0.250 | 0.378 | 0.661 | 0.465 | 0.444 |
| ANN (Xavier Init + Boosting) | 0.663 | 0.681 | 0.255 | 0.371 | 0.671 | 0.463 | 0.437 |
| Proposed Hybrid Model (Final) | 0.677 | 0.694 | 0.307 | 0.426 | 0.684 | 0.459 | 0.429 |

- Weighted average alpha = 0.5. Stability over 5 random splits: accuracy 0.672 ± 0.021, F1 0.406 ± 0.077.
- **These numbers are lower than the paper's (86.9% accuracy, AUC 0.894).** On this dataset the hybrid is the best model by accuracy and AUC here, but the margins are small and within the split-to-split variation above. The paper's higher numbers could not be reproduced with a leakage-free protocol. The paper's Table V weights also sum to 1.079, which contradicts its Eq. 3; this code uses Eq. 3.
- The previous version of this repo reported 71% using SMOTE and scaling *before* the split (leakage); it has been removed.

Multiclass WQI (4 balanced levels): hybrid accuracy 0.907, macro-F1 0.907 (ANN only 0.878, XGBoost only 0.868).

**Important notes**
- The Kaggle dataset has **no WQI column**. WQI is computed with the weighted-arithmetic method (`src/wqi.py`, WHO/EPA-style limits). Nearly all samples exceed the limits, so fixed cut-offs would give a single class; the four levels are WQI **quartiles** (relative levels). Since labels are a function of the same 9 features, high accuracy is expected and not comparable to the potability task.
- The dataset has **no time column**. The bundled LSTM is trained on a **simulated** daily series (seasonality + AR(1) noise around the real feature ranges): R² 0.832, about equal to the "tomorrow = today" persistence baseline (0.832). It demonstrates the pipeline only. Train on real data with `python model/train_lstm.py --csv your_series.csv` (columns: `date` + the 9 features).

---

## Installation & Setup

```bash
git clone https://github.com/rakesh-1143/water-quality-prediction-ml.git
cd water-quality-prediction-ml
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

The trained models are included in `model/artifacts/`. Retrain with:

```bash
python model/train_model.py        # binary hybrid + baselines + ablation + sensitivity (~2-3 min)
python model/train_multiclass.py   # 4-level WQI
python model/train_lstm.py         # LSTM forecaster
```

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite (65+ tests) covers the split and preprocessing (no leakage), WQI formula, SHAP weights and initialisation, ANN/XGBoost/fusion, metrics, all three training scripts end to end, model save/load, that the saved models reproduce the reported metrics, the LSTM pipeline, and every Streamlit page (headless, with button clicks).

## App pages

| Page | What it does |
|---|---|
| Home | Headline metrics |
| EDA | Overview, class balance, missing values, distributions, IQR outliers, correlations |
| Prediction | 9 sliders -> potable / not potable, confidence, NN and XGBoost outputs, SHAP explanation |
| Multiclass WQI | WQI value and level by formula, model-predicted level, class probabilities, SHAP |
| Time Series | LSTM test-period plot and next-step forecast from an uploaded CSV |
| Model Performance | Comparison, ablation, ROC, loss curve, SHAP ranking, stability |

## Project Structure

```
config.py                 settings and hyperparameters (paper Table VIII)
src/                      data.py, wqi.py, shap_init.py, nn_models.py, hybrid.py, metrics.py,
                          pipeline.py, timeseries.py, pipeline_ts.py, artifacts.py, ui.py
model/                    train_model.py, train_multiclass.py, train_lstm.py, artifacts/ (trained files)
app/                      app.py and pages/
tests/                    pytest suite
data/water_potability.csv Kaggle dataset
notebooks/eda.ipynb       EDA notebook
```

## Dataset

[Kaggle Water Potability](https://www.kaggle.com/adityakadiwal/water-potability): 3276 samples, 9 features (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity), binary label `Potability` (39% potable). Missing values: pH 491, Sulfate 781, Trihalomethanes 162.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, TensorFlow/Keras, Streamlit, Matplotlib, Seaborn, pytest.

## Acknowledgments

- Dataset: [Kaggle Water Potability Dataset](https://www.kaggle.com/adityakadiwal/water-potability)
- Al Bataineh, Vamsi, Smith (IEEE JSTARS 2026) for the hybrid framework
- Scikit-learn, XGBoost, SHAP, TensorFlow, and Streamlit communities

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

