"""Player Comparison: side-by-side percentiles, radar and the weighted-distance breakdown."""

import pandas as pd
import streamlit as st

from src.app import charts, data
from src.similarity.engine import explain_pair, weighted_cosine_matrix

st.title("📊 Player Comparison")
st.caption(
    "Both players must be in the same role group — comparing a centre-back to a "
    "winger on the same scale would be meaningless."
)

players = data.player_options()
season = data.player_season()

with st.sidebar:
    st.header("Players")
    roles = sorted(players["role"].unique())
    role = st.selectbox("Role group", roles, index=roles.index("AM_W") if "AM_W" in roles else 0)
    in_role = players[players["role"] == role].reset_index(drop=True)
    left_choice = st.selectbox("Player A", in_role["label"], index=None,
                               placeholder="Select a player…")
    right_choice = st.selectbox("Player B", in_role["label"], index=None,
                                placeholder="Select a player…")
    n_features = st.slider("Features on the radar", 6, 14, 10)

if left_choice is None or right_choice is None:
    st.info("Select two players of the same role group in the sidebar.", icon="👈")
    st.stop()
if left_choice == right_choice:
    st.warning("Pick two different players.")
    st.stop()

left_id = int(in_role[in_role["label"] == left_choice].iloc[0]["player_id"])
right_id = int(in_role[in_role["label"] == right_choice].iloc[0]["player_id"])
matrix = data.role_matrix(role)
indexed = season.set_index("player_id")
left_row, right_row = indexed.loc[left_id], indexed.loc[right_id]
left_name, right_name = data.display_name(left_row), data.display_name(right_row)

similarity = weighted_cosine_matrix(matrix.z, matrix.weights).loc[left_id, right_id]

col1, col2, col3 = st.columns(3)
col1.metric(left_name, f"{left_row['minutes']:.0f} min", left_row["primary_position"])
col2.metric(right_name, f"{right_row['minutes']:.0f} min", right_row["primary_position"])
col3.metric("Similarity", f"{similarity:.3f}")

breakdown = explain_pair(matrix, left_id, right_id)

# --- Radar of the most heavily weighted features --------------------------
weight_order = pd.Series(matrix.weights, index=matrix.features).sort_values(ascending=False)
radar_features = weight_order.head(n_features).index.tolist()


def _percentiles(player_id: int, features: list[str]) -> list[float]:
    row = indexed.loc[player_id]
    return [float((row.get(f"pct_{f}") or 0.0) * 100) for f in features]


st.subheader("Profile")
st.plotly_chart(
    charts.radar(
        labels=[f.replace("_p90", "").replace("_", " ") for f in radar_features],
        series={left_name: _percentiles(left_id, radar_features),
                right_name: _percentiles(right_id, radar_features)},
    ),
    width='stretch',
)
st.caption(f"Percentiles within the **{role}** group. Showing the {n_features} features "
           "weighted most heavily for this role.")

# --- Where they match and where they differ -------------------------------
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Most alike")
    closest = breakdown.head(5)
    st.dataframe(
        closest[["feature", "reference_value", "candidate_value"]].round(3),
        hide_index=True, width='stretch',
        column_config={"feature": "Feature", "reference_value": left_name,
                       "candidate_value": right_name},
    )
with right_col:
    st.subheader("Biggest differences")
    furthest = breakdown.tail(5).iloc[::-1]
    st.dataframe(
        furthest[["feature", "reference_value", "candidate_value"]].round(3),
        hide_index=True, width='stretch',
        column_config={"feature": "Feature", "reference_value": left_name,
                       "candidate_value": right_name},
    )

# --- Percentile comparison across every role feature ----------------------
st.subheader("All role features")
comparison = pd.DataFrame({
    "feature": matrix.features,
    left_name: _percentiles(left_id, matrix.features),
    right_name: _percentiles(right_id, matrix.features),
})
st.plotly_chart(
    charts.grouped_bars(comparison, "feature", [left_name, right_name]),
    width='stretch',
)

with st.expander("Raw values and weighted distance per feature"):
    detail = breakdown.rename(columns={
        "reference_value": left_name, "candidate_value": right_name,
        "gap": "weighted gap"})
    st.dataframe(
        detail[["feature", left_name, right_name, "weighted gap"]].round(3),
        hide_index=True, width='stretch',
    )
    st.caption(
        "**Weighted gap** is the absolute difference in standardized values, scaled by "
        "the feature's role weight — small means the players match on that dimension."
    )
