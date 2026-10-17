import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.data import load_dataset

QUICK_EPOCHS = 3


@pytest.fixture(scope="session")
def df():
    return load_dataset()


@pytest.fixture(scope="session")
def binary_run(df):
    from src.pipeline import run_binary_pipeline
    return run_binary_pipeline(df, epochs=QUICK_EPOCHS)


@pytest.fixture(scope="session")
def multiclass_run(df):
    from src.pipeline import run_multiclass_pipeline
    return run_multiclass_pipeline(df, epochs=QUICK_EPOCHS)


@pytest.fixture(scope="session")
def lstm_run(df):
    from src.pipeline_ts import run_lstm_pipeline
    return run_lstm_pipeline(df=df, epochs=2)
