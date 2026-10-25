"""Design system: palettes, escaping, JS safety, helpers, and every page under every theme."""
import os
import re
import shutil
import subprocess

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import theme
from src.artifacts import BinaryBundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAGES = ["home", "about", "contact", "dashboard"]


@pytest.fixture
def captured(monkeypatch):
    out = []
    monkeypatch.setattr(st, "html", lambda body, **kw: out.append((body, kw)))
    return out


@pytest.mark.parametrize("name", list(theme.PALETTES))
def test_palette_is_complete_and_css_has_no_unfilled_tokens(name):
    p = theme.PALETTES[name]
    css = theme._css(p)
    assert not re.findall(r"__[a-z0-9]+__", css)
    assert p["scene"] in ("waves", "orbs", "helix", "rings")
    for key in ("primary", "primary2", "accent", "ink", "bg1", "bg2"):
        assert p[key].startswith("#")
    assert "prefers-reduced-motion" in css                      # motion can be switched off


def test_each_theme_uses_a_different_scene_and_colour():
    assert len(theme.PALETTES) == 4 and "Blue" in theme.PALETTES
    assert len({p["scene"] for p in theme.PALETTES.values()}) == 4
    assert len({p["primary"] for p in theme.PALETTES.values()}) == 4


def test_safe_js_has_no_tag_like_less_than_and_keeps_meaning():
    out = theme.safe_js("for(i=0;i<n;i++){} if(a<=b){} x=a<b*2;")
    assert not re.search(r"<(?![ =])", out)
    assert "i < n" in out and "a <= b" in out and "a < b*2" in out


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("scene", ["waves", "orbs", "helix", "rings"])
def test_javascript_is_syntactically_valid(scene, tmp_path):
    scene_js = (theme._SCENE_JS.replace("__ID__", "c1").replace("__SCENE__", scene)
                .replace("__C1__", "#fff").replace("__C2__", "#fff").replace("__C3__", "#fff"))
    for name, code in (("global", theme._GLOBAL_JS), ("scene", scene_js)):
        f = tmp_path / f"{name}.js"
        f.write_text(theme.safe_js(code))
        done = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


def test_hero_escapes_user_text_and_enables_scene_script(captured):
    theme.hero("<b>k</b>", "<script>alert(1)</script>", "<img src=x onerror=1>", stats={"<i>x</i>": "<u>1</u>"})
    body, kw = captured[0]
    assert kw.get("unsafe_allow_javascript") is True
    assert "&lt;script&gt;alert(1)" in body and "<script>alert" not in body
    assert "<img" not in body and "&lt;img" in body
    assert body.count("<script>") == 1                           # only our own scene script


def test_gauge_ladder_leaderboard_and_cards_helpers(captured):
    theme.gauge(1.7, "x")
    theme.gauge(-1, "y")
    theme.ladder(["A", "B", "C", "D"], 2)
    theme.leaderboard({"m1": 0.6, "m2": 0.9, "m3": 1.4})
    theme.feature_cards([("<i>", "<t>", "<p>")])
    theme.timeline([("a", "b")])
    gauge_hi, gauge_lo, ladder, board, cards, timeline = (c[0] for c in captured)
    assert "--v:100.0" in gauge_hi and "100%" in gauge_hi and "--v:0.0" in gauge_lo and "0%" in gauge_lo
    assert ladder.count('class="on"') == 1 and ladder.index('class="on"') > ladder.index(">B<")
    assert board.count("wq-bar top") == 1 and "--w:100.0%" in board    # best highlighted, width clamped
    assert "&lt;t&gt;" in cards and "<t>" not in cards
    assert "wq-step" in timeline


def test_current_falls_back_for_unknown_theme(monkeypatch):
    monkeypatch.setattr(st, "session_state", {"design_theme": "Nope"})
    assert theme.current() == theme.DEFAULT_THEME
    monkeypatch.setattr(st, "session_state", {"_wq_theme": "Aurora"})
    assert theme.current() == "Aurora"


def test_matplotlib_follows_palette():
    import matplotlib as mpl
    theme.style_matplotlib(theme.PALETTES["Lagoon"])
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0] == theme.PALETTES["Lagoon"]["primary"]
    assert not mpl.rcParams["axes.spines.top"]


@pytest.mark.skipif(not BinaryBundle.exists(), reason="trained files missing")
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("name", list(theme.PALETTES))
def test_every_page_renders_in_every_theme(page, name):
    at = AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=120)
    at.session_state["design_theme"] = name
    at.run()
    at.switch_page(f"views/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    css = " ".join(m.value for m in at.markdown if "--wq-primary" in m.value)
    assert theme.PALETTES[name]["primary"] in css                 # the chosen palette is injected


def test_remove_stale_pages_dir_only_removes_folders_without_python_files(tmp_path):
    from src.ui import remove_stale_pages_dir
    assert remove_stale_pages_dir(str(tmp_path)) is False                    # nothing there
    stale = tmp_path / "pages" / "__pycache__"
    stale.mkdir(parents=True)
    (stale / "old.cpython-313.pyc").write_bytes(b"x")
    assert remove_stale_pages_dir(str(tmp_path)) is True and not (tmp_path / "pages").exists()
    real = tmp_path / "pages"
    real.mkdir()
    (real / "1_Page.py").write_text("print(1)")
    assert remove_stale_pages_dir(str(tmp_path)) is False and (real / "1_Page.py").exists()   # real pages are kept
