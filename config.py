import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"

RANDOM_STATE = 42
SPLIT = (0.70, 0.15, 0.15)  # train / validation / test (IEEE paper, Sec. III-B)

# Neural network (IEEE paper, Table VIII)
NN_PARAMS = dict(hidden=(16, 8), dropout=0.2, learning_rate=1e-3,
                 epochs=100, batch_size=32, patience=10)

# XGBoost (IEEE paper, Table VIII)
XGB_PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
                  subsample=0.8, reg_lambda=1.0, reg_alpha=0.5)

RF_PARAMS = dict(n_estimators=200)

# Weighted-average search range for the hybrid model (IEEE paper, Sec. III-E-2)
ALPHA_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)

# Multiclass WQI levels, best to worst
WQI_CLASSES = ["Excellent", "Good", "Poor", "Very Poor"]

# LSTM time-series model
LSTM_PARAMS = dict(window=14, units=64, dropout=0.2, learning_rate=1e-3,
                   epochs=60, batch_size=32, patience=8)
SIM_STEPS = 1500  # length of the simulated series (tests / demos only)

# Real time-series data: Brisbane River buoy (Queensland Government Open Data, Apache-2.0)
BRISBANE_PATH = os.path.join(BASE_DIR, "data", "brisbane_river_water_quality.csv")
BRISBANE_WQI_PARAMS = ["ph", "dissolved_oxygen", "turbidity", "chlorophyll"]
BRISBANE_INPUTS = BRISBANE_WQI_PARAMS + ["temperature", "specific_conductance"]
TS_WINDOWS = {"D": 7, "h": 24}
TS_DEFAULT_FREQ = "h"  # hourly: 6,229 steps (daily has only 302, too few for an LSTM)  # look-back window per resample frequency


# Website / contact details shown on the About and Contact pages
APP_NAME = "Water Quality ML"
GITHUB_URL = "https://github.com/Rakesh-1143/water-quality-prediction-ml"
CONTACT_EMAIL = ""  # set your address to enable the "Send by email" button on the Contact page
