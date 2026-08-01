"""Home: what the platform does, what it is built on, and what it does not claim."""

import streamlit as st

from src.app import data

st.title("⚽ Football Intelligence Platform")
st.markdown(
    "Find **stylistically similar players** for recruitment, and group **teams by how "
    "they play** — every result traceable back to the underlying metrics."
)

summary = data.dataset_summary()
config = data.config()["dataset"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches", f"{summary['matches']:,}")
col2.metric("Events", f"{summary['events']:,}")
col3.metric("Qualified players", summary["players_qualified"])
col4.metric("Teams", summary["teams"])

st.caption(
    f"{config['competition_name']} {config['season_name']} · "
    f"raw data downloaded {summary['downloaded_at'] or 'unknown'}"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Player Recruitment Similarity")
    st.markdown(
        """
Pick a reference player and get a ranked list of players who **do similar things**,
with an explanation of *why* each one is similar and where they differ.

- Features are standardized **within role group**, so a centre-back is never
  compared against a winger's scale.
- Candidates are filtered by role, minutes and data coverage **before** scoring.
- Every result decomposes into the dimensions where the players match and clash.
"""
    )
    st.page_link("pages/player_search.py", label="Open Player Search", icon="🔍")

with right:
    st.subheader("Team Playing-Style Classification")
    st.markdown(
        """
28 style features per team per match — possession, directness, territory, width,
pressing, transitions, chance profile and risk — aggregated to a season profile
**and its match-to-match variability**.

- PCA reveals the main style axes; k-means groups teams into style clusters.
- Cluster labels were written **after** inspecting the centroids, never before.
- Nearest-team similarity crosses cluster boundaries, so no side is trapped by its label.
"""
    )
    st.page_link("pages/team_style_map.py", label="Open Team Style Map", icon="🗺️")

st.divider()

st.warning(
    "**These results are descriptive.** Similarity means two players *do similar "
    "things* — it is not evidence of equal quality, tactical fit for a particular "
    "system, or transfer value. Team clusters describe how sides played, not how well. "
    "This platform makes no match-result predictions.",
    icon="⚠️",
)

st.info(
    "Data provided by **StatsBomb**. See the Methodology page for metric definitions, "
    "validation results and known limitations.",
    icon="ℹ️",
)
