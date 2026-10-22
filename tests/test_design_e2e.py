"""Real-browser check of the animations (skipped when Playwright or Chromium is not available)."""
import glob
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pw_api = pytest.importorskip("playwright.sync_api")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMIUM = os.environ.get("WQ_CHROMIUM") or next(iter(sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))), None)
pytestmark = pytest.mark.skipif(not CHROMIUM or not os.path.exists(CHROMIUM), reason="no Chromium found (set WQ_CHROMIUM)")


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app", "app.py"),
                             "--server.headless", "true", "--server.port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3"))
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(1)
    yield url
    proc.terminate()
    proc.wait(timeout=20)


@pytest.fixture(scope="module")
def page(server):
    with pw_api.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_selector(".wq-hero", timeout=90000)
        pg.wait_for_timeout(3000)
        yield pg
        browser.close()


def scroll(pg, y):
    pg.evaluate(f"document.querySelector('[data-testid=stMain]').scrollTo(0,{y})")
    pg.wait_for_timeout(1500)


def test_scripts_run_and_3d_canvas_draws(page):
    assert page.evaluate("document.body.classList.contains('wq-js')")
    assert page.evaluate("!!document.getElementById('wq-progress')")
    drawn = page.evaluate("""()=>{const c=document.querySelector('.wq-hero canvas');
      const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=200)if(d[i]>0)n++;return n}""")
    assert drawn > 100


def test_canvas_animates_between_frames(page):
    grab = """()=>{const c=document.querySelector('.wq-hero canvas');return c.toDataURL().length}"""
    frames = set()
    for _ in range(4):
        frames.add(page.evaluate(grab))
        page.wait_for_timeout(300)
    assert len(frames) > 1


def test_scroll_reveals_elements_and_moves_progress_bar(page):
    scroll(page, 0)
    page.evaluate("document.getElementById('wq-progress').style.width='0%'")
    scroll(page, 1500)
    assert page.evaluate("document.querySelectorAll('.wq-in').length") > 5
    assert float(page.evaluate("parseFloat(document.getElementById('wq-progress').style.width)")) > 50
    assert page.evaluate("document.querySelectorAll('.wq-pre').length") == 0      # nothing left hidden


def test_hover_applies_3d_tilt_to_cards(page):
    scroll(page, 600)
    card = page.locator(".wq-card").first
    box = card.bounding_box()
    page.mouse.move(box["x"] + 20, box["y"] + 20)
    page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.3)
    page.wait_for_timeout(500)
    assert "rotateX" in page.evaluate("document.querySelector('.wq-card').style.transform")


def test_theme_switch_changes_palette_and_persists_across_pages(page):
    scroll(page, 0)
    page.get_by_text("Aurora", exact=True).first.click()
    page.wait_for_timeout(3000)
    assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--wq-primary')").strip() == "#6366f1"
    page.get_by_role("link", name="EDA").first.click()
    page.wait_for_selector("text=Dataset Overview", timeout=60000)
    page.wait_for_timeout(2500)
    assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--wq-primary')").strip() == "#6366f1"
    assert page.errors == []
