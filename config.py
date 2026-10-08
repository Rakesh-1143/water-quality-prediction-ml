"""Paper-aligned binary potability experiment settings (Table VIII)."""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'water_potability.csv'
ARTIFACT_DIR = BASE_DIR / 'model' / 'paper'
REPORT_DIR = BASE_DIR / 'reports'
ABSTRACT_ARTIFACT_DIR = BASE_DIR / 'model' / 'abstract'
ABSTRACT_REPORT_DIR = REPORT_DIR / 'abstract'
FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10
XGB_PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
                  subsample=0.8, reg_lambda=1.0, reg_alpha=0.5,
                  eval_metric='logloss', n_jobs=2)
