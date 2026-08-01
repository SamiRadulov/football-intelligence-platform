"""Player Search: pick a reference player, filter candidates, rank by similarity."""

import streamlit as st

from src.app import data
from src.similarity.engine import find_similar

st.title("🔍 Player Search")
st.caption(
    "Candidates are filtered by role, minutes and data coverage **before** similarity "
    "is computed, so an incompatible role can never surface on a statistical fluke."
)

players = data.player_options()
season = data.player_season()
config = data.config()

with st.sidebar:
    st.header("Reference player")
    choice = st.selectbox("Player", players["label"], index=None,
                          placeholder="Search for a player…")

    st.header("Candidate filters")
    min_minutes = st.slider("Minimum minutes played", 600, 3400, 900, step=100)
    top_n = st.slider("Results to show", 5, 25, 10, step=5)
    exclude_team = st.checkbox("Exclude team-mates", value=False)

if choice is None:
    st.info("Select a reference player in the sidebar to begin.", icon="👈")
    st.stop()

reference = players[players["label"] == choice].iloc[0]
role = reference["role"]
matrix = data.role_matrix(role)
ref_row = season[season["player_id"] == reference["player_id"]].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Player", data.display_name(ref_row))
col2.metric("Role group", role)
col3.metric("Position", ref_row["primary_position"])
col4.metric("Minutes", f"{ref_row['minutes']:.0f}")

results = find_similar(
    matrix, season, int(reference["player_id"]),
    top_n=top_n, min_minutes=min_minutes,
    min_coverage=config["similarity"]["min_feature_coverage"],
    exclude_same_team=exclude_team,
)

st.caption(
    f"Compared against {len(matrix.z) - 1} other **{role}** players "
    f"on {len(matrix.features)} role-specific features."
)

if results.empty:
    st.warning("No candidates match those filters. Try lowering the minutes threshold.")
    st.stop()

table = results.copy()
table["similarity"] = table["similarity"].round(3)
table["most alike on"] = table["most_similar_on"].apply(lambda fs: ", ".join(fs[:3]))
table["differs most on"] = table["biggest_differences"].apply(lambda fs: ", ".join(fs))
table["⚠"] = table["low_minutes"].map({True: "low minutes", False: ""})

st.dataframe(
    table[["player_name", "minutes", "similarity", "coverage",
           "most alike on", "differs most on", "⚠"]],
    hide_index=True,
    width='stretch',
    column_config={
        "player_name": st.column_config.TextColumn("Player"),
        "minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "similarity": st.column_config.ProgressColumn(
            "Similarity", min_value=0.0, max_value=1.0, format="%.3f"),
        "coverage": st.column_config.NumberColumn(
            "Data coverage", format="%.0f%%",
            help="Share of this role's features actually available for the player"),
    },
)

st.caption(
    "**Similarity** is weighted cosine on features standardized within the role group: "
    "it measures whether two players deviate from their role's average in the same "
    "directions. It is descriptive, not a ranking of quality."
)

with st.expander("Compare the reference against one of these players"):
    target = st.selectbox("Candidate", results["player_name"])
    st.page_link("pages/player_comparison.py", label="Open Player Comparison", icon="📊")
    st.caption(
        f"Select **{data.display_name(ref_row)}** and **{target}** there for the "
        "full side-by-side breakdown."
    )
