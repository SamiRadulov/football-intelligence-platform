"""Football Intelligence Platform — Streamlit entry point.

Run from the repo root:
    .venv\\Scripts\\streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Football Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/player_search.py", title="Player Search", icon="🔍"),
    st.Page("pages/player_comparison.py", title="Player Comparison", icon="📊"),
    st.Page("pages/team_style_map.py", title="Team Style Map", icon="🗺️"),
    st.Page("pages/team_profile.py", title="Team Profile", icon="🧭"),
    st.Page("pages/methodology.py", title="Methodology & Data Quality", icon="📖"),
]

st.navigation(PAGES).run()

st.sidebar.divider()
st.sidebar.caption(
    "Data: **StatsBomb Open Data** — Premier League 2015/16. "
    "Results are descriptive, not a judgement of quality."
)
