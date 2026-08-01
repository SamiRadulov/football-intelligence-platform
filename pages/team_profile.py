"""Team Profile: style fingerprint, nearest teams and match-to-match variability."""

import pandas as pd
import streamlit as st

from src.app import charts, data
from src.clustering.style_model import nearest_teams
from src.features.team_match import STYLE_FEATURES

st.title("🧭 Team Profile")

season = data.team_season()
style = data.team_style()
space = data.style_space()
matches = data.team_match()

with st.sidebar:
    st.header("Team")
    team = st.selectbox("Team", sorted(season["team"]), index=None,
                        placeholder="Select a team…")
    n_features = st.slider("Features to show", 8, len(STYLE_FEATURES), 14)

if team is None:
    st.info("Select a team in the sidebar.", icon="👈")
    st.stop()

row = season[season["team"] == team].iloc[0]
style_row = style[style["team"] == team].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Team", team)
col2.metric("Style cluster", style_row["cluster_label"])
col3.metric("Matches", int(row["matches"]))

# --- Style fingerprint -----------------------------------------------------
percentiles = pd.DataFrame({
    "feature": STYLE_FEATURES,
    "percentile": [float(row[f"pct_{f}"]) * 100 for f in STYLE_FEATURES],
    "value": [float(row[f"{f}_mean"]) for f in STYLE_FEATURES],
})
percentiles["distance_from_average"] = (percentiles["percentile"] - 50).abs()
most_distinctive = percentiles.nlargest(n_features, "distance_from_average")

st.subheader("Style fingerprint")
st.plotly_chart(
    charts.percentile_bars(most_distinctive, "feature", "percentile"),
    width='stretch',
)
st.caption(
    "Percentile against the other 19 teams, showing the features where this side is "
    "furthest from average. The dotted line is the league median."
)

# --- Nearest teams ---------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Most similar teams")
    near = nearest_teams(space, team, n=5)
    near = near.merge(style[["team", "cluster_label"]], on="team", how="left")
    st.dataframe(
        near.round(2), hide_index=True, width='stretch',
        column_config={
            "team": "Team",
            "distance": st.column_config.NumberColumn(
                "Distance", help="Euclidean distance in the standardized style space"),
            "cluster_label": "Their cluster",
        },
    )
    st.caption("Computed across all 28 features — neighbours may sit in another cluster.")

with right:
    st.subheader("Versus the cluster centroid")
    cluster_teams = style[style["cluster_label"] == style_row["cluster_label"]]["team"]
    centroid = space.z.loc[cluster_teams].mean()
    deltas = (space.z.loc[team] - centroid).sort_values()
    comparison = pd.concat([deltas.head(4), deltas.tail(4)]).to_frame("difference")
    comparison.index.name = "feature"
    st.dataframe(
        comparison.round(2).reset_index(), hide_index=True, width='stretch',
        column_config={"feature": "Feature",
                       "difference": st.column_config.NumberColumn(
                           "vs cluster (z)",
                           help="Positive means higher than the cluster average")},
    )
    st.caption(f"Distance to centroid: **{style_row['centroid_distance']:.2f}** — "
               "higher means the label describes this side less well.")

# --- Variability -----------------------------------------------------------
st.divider()
st.subheader("Match-to-match variability")
st.caption(
    "How much each feature moves from game to game. A rigid side and an adaptable one "
    "can share the same season average — the spread is what separates them."
)

variability = pd.DataFrame({
    "feature": STYLE_FEATURES,
    "sd": [float(row[f"{f}_sd"]) for f in STYLE_FEATURES],
})
shares = variability[variability["sd"] < 1.0]      # comparable 0-1 scale features
st.plotly_chart(
    charts.variability_bars(shares.nlargest(n_features, "sd"), "feature", "sd"),
    width='stretch',
)

with st.expander("Match-by-match detail"):
    team_matches = matches[matches["team"] == team]
    feature = st.selectbox("Feature", STYLE_FEATURES,
                           index=STYLE_FEATURES.index("possession_share"))
    detail = team_matches[["match_id", "opponent", feature]].sort_values("match_id")
    st.line_chart(detail.set_index("match_id")[feature])
    st.dataframe(detail.round(3), hide_index=True, width='stretch')
