"""Entry point: top navigation bar (Home, About, Contact, Dashboard) and the shared design system."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))   # config.py, src/
sys.path.insert(0, HERE)                                        # views/

import streamlit as st

from config import APP_NAME
from src import theme
from src.ui import remove_stale_pages_dir

remove_stale_pages_dir(HERE)
st.set_page_config(page_title=APP_NAME, page_icon="💧", layout="wide")
theme.apply()
st.logo(os.path.join(HERE, "assets", "logo.svg"), size="large")

navigation = st.navigation(
    [st.Page("views/home.py", title="Home", icon="🏠", default=True),
     st.Page("views/about.py", title="About", icon="ℹ️"),
     st.Page("views/contact.py", title="Contact", icon="✉️"),
     st.Page("views/dashboard.py", title="Dashboard", icon="📊")],
    position="top",
)
navigation.run()
