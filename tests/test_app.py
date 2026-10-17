"""Streamlit pages: run each page headlessly, click the buttons, check the output."""
import os

import pytest
from streamlit.testing.v1 import AppTest

from src.artifacts import BinaryBundle, MulticlassBundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMEOUT = 120

pytestmark = pytest.mark.skipif(not (BinaryBundle.exists() and MulticlassBundle.exists()),
                                reason="trained files missing: run the model/train_*.py scripts")


def page(name):
    return AppTest.from_file(os.path.join(ROOT, name), default_timeout=TIMEOUT).run()


@pytest.mark.parametrize("name", ["app/app.py", "app/pages/1_EDA.py", "app/pages/2_Prediction.py",
                                  "app/pages/3_Multiclass_WQI.py", "app/pages/4_Time_Series.py",
                                  "app/pages/5_Model_Performance.py"])
def test_page_renders_without_error(name):
    at = page(name)
    assert not at.exception, [e.value for e in at.exception]


def test_home_shows_metrics():
    at = page("app/app.py")
    assert len(at.metric) == 3
    assert "%" in at.metric[0].value


def test_eda_feature_selector_changes_chart():
    at = page("app/pages/1_EDA.py")
    assert len(at.metric) == 3 and at.metric[0].value == "3276" and at.metric[2].value == "1434"
    at.selectbox[0].select("Sulfate").run()
    assert not at.exception


def test_prediction_page_predicts_and_explains():
    at = page("app/pages/2_Prediction.py")
    assert len(at.sidebar.slider) == 9
    assert not at.success and not at.error            # nothing before the click
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) + len(at.error) == 1       # exactly one verdict
    assert any("Model Confidence" in i.value for i in at.info)


def test_prediction_changes_with_inputs():
    at = page("app/pages/2_Prediction.py")
    at.button[0].click().run()
    first = [i.value for i in at.info if "Model Confidence" in i.value][0]
    for s in at.sidebar.slider:
        if s.label.startswith("Solids"):
            s.set_value(60000.0)
        if s.label == "pH":
            s.set_value(2.0)
    at.button[0].click().run()
    second = [i.value for i in at.info if "Model Confidence" in i.value][0]
    assert not at.exception and first != second


def test_prediction_page_wqi_task_switch():
    at = page("app/pages/2_Prediction.py")
    at.sidebar.radio[0].set_value("WQI-based good / poor").run()
    assert not at.exception and any("formula" in c.value for c in at.caption)
    at.button[0].click().run()
    assert not at.exception
    verdicts = [x.value for x in list(at.success) + list(at.error)]
    assert len(verdicts) == 1 and ("GOOD" in verdicts[0] or "POOR" in verdicts[0])
    assert any("WQI by formula" in m.value for m in at.markdown)


def test_multiclass_page_formula_and_model():
    at = page("app/pages/3_Multiclass_WQI.py")
    assert len(at.sidebar.slider) == 9 and len(at.metric) == 1
    wqi_default = float(at.metric[0].value)
    at.button[0].click().run()
    assert not at.exception
    assert any(s.value.startswith("Predicted level:") for s in at.success)
    for s in at.sidebar.slider:
        if s.label.startswith("Sulfate"):
            s.set_value(480.0)
    at.run()
    assert float(at.metric[0].value) > wqi_default    # more sulfate -> worse (higher) WQI


def test_time_series_page_shows_real_data_notice_and_metrics():
    at = page("app/pages/4_Time_Series.py")
    assert not at.warning                                    # no "simulated" warning for real data
    assert any("real" in i.value.lower() and "Brisbane" in i.value for i in at.info)
    assert len(at.metric) == 3


def test_performance_page_tables():
    at = page("app/pages/5_Model_Performance.py")
    assert len(at.tabs) == 5 and len(at.dataframe) >= 5
    assert any("WQI-based good / poor" in h.value for h in at.subheader)
