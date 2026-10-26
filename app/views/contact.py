"""Contact: links and a message composer (no server-side storage)."""
import os
import re
import sys
from urllib.parse import quote

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from config import APP_NAME, CONTACT_EMAIL, GITHUB_URL
from src import theme

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOPICS = ["Question", "Feedback", "Bug report", "Collaboration"]

theme.hero("Contact", "Let’s talk about water quality",
           "Questions, feedback, bug reports or collaboration ideas are all welcome.", height=280, scene="orbs")

left, right = st.columns([1, 1.4], gap="large")
with left:
    theme.card("🐙", "GitHub", "Source code, issues and discussions live in the repository.")
    st.link_button("Open repository", GITHUB_URL, use_container_width=True)
    theme.card("🐞", "Report an issue", "Found a bug or want a feature? Open an issue and we will take a look.")
    st.link_button("New issue", f"{GITHUB_URL}/issues/new", use_container_width=True)
    if CONTACT_EMAIL:
        theme.card("✉️", "Email", CONTACT_EMAIL)
    theme.note("This site has no server-side mailbox: messages are never stored here. The buttons on the right hand "
               "your message to your email app or to GitHub.")

with right:
    st.subheader("Send a message")
    with st.form("contact_form", clear_on_submit=False):
        name = st.text_input("Your name", key="contact_name")
        email = st.text_input("Your email", key="contact_email")
        topic = st.selectbox("Topic", TOPICS, key="contact_topic")
        message = st.text_area("Message", height=160, key="contact_message")
        submitted = st.form_submit_button("Prepare message")

    if submitted:
        problems = []
        if not name.strip():
            problems.append("Please enter your name.")
        if not EMAIL_RE.match(email.strip()):
            problems.append("Please enter a valid email address.")
        if len(message.strip()) < 10:
            problems.append("Please write a message of at least 10 characters.")
        if problems:
            for p in problems:
                st.error(p)
        else:
            subject = f"[{APP_NAME}] {topic} from {name.strip()}"
            body = f"{message.strip()}\n\n— {name.strip()} ({email.strip()})"
            st.success("Your message is ready. Choose how to send it:")
            if CONTACT_EMAIL:
                st.link_button("Send by email", f"mailto:{CONTACT_EMAIL}?subject={quote(subject)}&body={quote(body)}",
                               use_container_width=True)
            st.link_button("Post as a GitHub issue",
                           f"{GITHUB_URL}/issues/new?title={quote(subject)}&body={quote(body)}",
                           use_container_width=True)
            if not CONTACT_EMAIL:
                st.caption("Set `CONTACT_EMAIL` in `config.py` to enable the email button.")

with st.expander("Frequently asked questions"):
    st.markdown("**Why is the accuracy only about 68%?** The nine inputs carry limited information about the Kaggle "
                "potability label; see the About page.\n\n"
                "**Can I use my own sensor data?** Yes: train the LSTM with `python model/train_lstm.py --csv your.csv`.\n\n"
                "**Is my input stored?** No. Predictions run in memory and nothing is saved.")

theme.footer(APP_NAME, GITHUB_URL)
