"""Website pages (top navigation) and dashboard tools, run headless with button clicks."""
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

from src.artifacts import BinaryBundle, MulticlassBundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMEOUT = 120

pytestmark = pytest.mark.skipif(not (BinaryBundle.exists() and MulticlassBundle.exists()),
                                reason="trained files missing: run the model/train_*.py scripts")


def site(view=None, **state):
    at = AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=TIMEOUT)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if view:
        at.switch_page(f"views/{view}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def tool(name, **state):
    """Dashboard with the given tool selected."""
    return site("dashboard", dash_tool=name, **state)


def test_top_navigation_has_home_about_contact_dashboard():
    src = open(os.path.join(ROOT, "app", "app.py")).read()
    assert re.findall(r'st\.Page\("views/(\w+)\.py", title="(\w+)"', src) == [
        ("home", "Home"), ("about", "About"), ("contact", "Contact"), ("dashboard", "Dashboard")]
    assert 'position="top"' in src


@pytest.mark.parametrize("view", ["home", "about", "contact", "dashboard"])
def test_every_page_renders(view):
    site(view)


def test_home_shows_hero_metrics_and_dashboard_button():
    at = site()
    assert len(at.metric) == 3 and "%" in at.metric[0].value
    assert any(b.label == "Open dashboard →" for b in at.button)


def test_home_open_dashboard_button_goes_to_dashboard():
    at = site()
    next(b for b in at.button if b.key == "home_cta_dash").click().run()
    assert not at.exception and len(at.metric) == 4          # the dashboard KPI row


def test_home_tool_card_opens_that_tool_in_the_dashboard():
    at = site()
    next(b for b in at.button if b.key == "home_open_Predict").click().run()
    assert not at.exception
    assert len(at.sidebar.slider) == 9                       # the Predict tool is showing


def page_html(at):
    return " ".join(h.value for h in at.get("html"))


def test_home_about_button_goes_to_about():
    at = site()
    assert "Know your water quality, and why" in page_html(at)
    next(b for b in at.button if b.key == "home_cta_about").click().run()
    assert not at.exception
    assert "Hybrid machine learning for water quality" in page_html(at)
    assert "Know your water quality, and why" not in page_html(at)


def test_dashboard_overview_and_tool_switch_state():
    at = site("dashboard")
    assert len(at.metric) == 4 and at.session_state["dash_tool"] == "Overview"
    assert len(at.sidebar.slider) == 0                      # no input sidebar on the overview


def test_about_page_states_the_limitations_and_sources():
    html = page_html(site("about"))
    for expected in ("Limitations, stated plainly", "86.9%", "67.7%", "10.1109/JSTARS.2026.3654017",
                     "Brisbane River buoy", "Kaggle Water Potability"):
        assert expected in html, expected


def test_contact_form_validation_and_links():
    at = site("contact")
    at.button[0].click().run()                               # empty form
    assert len(at.error) == 3
    at.text_input(key="contact_name").set_value("Asha")
    at.text_input(key="contact_email").set_value("not-an-email")
    at.text_area(key="contact_message").set_value("short")
    at.button[0].click().run()
    assert {e.value for e in at.error} == {"Please enter a valid email address.",
                                           "Please write a message of at least 10 characters."}
    at.text_input(key="contact_email").set_value("asha@example.com")
    at.text_area(key="contact_message").set_value("Hello, I have a question about the model.")
    at.button[0].click().run()
    assert not at.exception and not at.error and any("ready" in s.value for s in at.success)


def test_explore_tool():
    at = tool("Explore data")
    assert len(at.metric) == 4 + 3 and at.metric[4].value == "3276" and at.metric[6].value == "1434"
    at.selectbox(key="eda_feature").select("Sulfate").run()
    assert not at.exception


def test_prediction_tool_predicts_and_explains():
    at = tool("Predict")
    assert len(at.sidebar.slider) == 9
    assert not at.success and not at.error
    at.button[0].click().run()
    assert not at.exception
    assert len(at.success) + len(at.error) == 1
    assert any("Model Confidence" in i.value for i in at.info)


def test_prediction_changes_with_inputs():
    at = tool("Predict")
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


def test_prediction_tool_wqi_task_switch():
    at = tool("Predict")
    at.sidebar.radio[0].set_value("WQI-based good / poor").run()
    assert not at.exception and any("formula" in c.value for c in at.caption)
    at.button[0].click().run()
    assert not at.exception
    verdicts = [x.value for x in list(at.success) + list(at.error)]
    assert len(verdicts) == 1 and ("GOOD" in verdicts[0] or "POOR" in verdicts[0])
    assert any("WQI by formula" in m.value for m in at.markdown)


def test_wqi_levels_tool_formula_and_model():
    at = tool("WQI levels")
    assert len(at.sidebar.slider) == 9
    wqi_default = float(at.metric[4].value)
    at.button[0].click().run()
    assert not at.exception
    assert any(s.value.startswith("Predicted level:") for s in at.success)
    for s in at.sidebar.slider:
        if s.label.startswith("Sulfate"):
            s.set_value(480.0)
    at.run()
    assert float(at.metric[4].value) > wqi_default          # more sulfate -> worse (higher) WQI


def test_forecast_tool_shows_real_data_notice_and_metrics():
    at = tool("Forecast")
    assert not at.warning
    assert any("real" in i.value.lower() and "Brisbane" in i.value for i in at.info)
    assert len(at.metric) == 4 + 3


def test_performance_tool_tables():
    at = tool("Performance")
    assert len(at.tabs) == 5 and len(at.dataframe) >= 5
    assert any("WQI-based good / poor" in h.value for h in at.subheader)
