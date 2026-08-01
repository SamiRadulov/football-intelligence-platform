"""Methodology & Data Quality: definitions, validation results and limitations."""

import streamlit as st

from src.app import data

st.title("📖 Methodology & Data Quality")

summary = data.dataset_summary()
dataset = data.config()["dataset"]

st.subheader("Coverage")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches", f"{summary['matches']:,}")
col2.metric("Events", f"{summary['events']:,}")
col3.metric("Players ≥ 600 min", summary["players_qualified"])
col4.metric("Teams", summary["teams"])
st.caption(
    f"{dataset['competition_name']} {dataset['season_name']} "
    f"(`competition_id={dataset['competition_id']}`, `season_id={dataset['season_id']}`) · "
    f"raw data downloaded {summary['downloaded_at'] or 'unknown'}"
)

st.warning(
    "**Limitations in brief.** Similarity is descriptive, not a quality ranking. "
    "The style clusters describe one season of one league, and the possession cluster "
    "overlaps the strongest sides, so style and quality are partly entangled. "
    "StatsBomb Open Data publishes no player dates of birth, so age filtering is not "
    "possible from this source.",
    icon="⚠️",
)

tab_method, tab_metrics, tab_quality = st.tabs(
    ["Methodology", "Metric dictionary", "Data quality"]
)

with tab_method:
    st.markdown(data.read_markdown("METHODOLOGY.md"))

with tab_metrics:
    st.markdown(data.read_markdown("METRIC_DICTIONARY.md"))

with tab_quality:
    st.markdown(
        """
### Pipeline checks

The staging build fails if any **hard** check breaks, so the marts can never be
built from data that violates these guarantees:

- **Schema** — every column the downstream phases depend on exists.
- **Uniqueness** — `match_id`, `player_id`, `event_id` and `(match_id, player_id)`
  are unique.
- **Referential integrity** — every event and lineup row points at a known match
  and a known player.
- **Football sanity** — exactly 11 starters per team per match; no player is on the
  pitch longer than the match lasted.

One **warning-level** check surfaces genuine source-data quirks rather than hiding
them: summed team minutes should equal 11 × match length, and in about 15 of 760
team-matches it does not — a player leaving temporarily without a replacement, a
red card, or a lineup that internally contradicts itself. These are recorded, not
silently corrected.

### Team feature checks

- Possession share and field tilt must sum to exactly 1.0 across the two teams of
  every match.
- All share metrics must lie in [0, 1]; PPDA must be positive.
- Every team must have 38 matches.

### Feature checks

- No negative per-90 values; all percentiles within [0, 1].
- No player below the 600-minute threshold reaches the season mart.
- Rates are null below their minimum-attempt guard rather than being reported on a
  tiny sample.
"""
    )

st.divider()
st.subheader("Data source")
st.markdown(
    "Data provided by **[StatsBomb](https://github.com/statsbomb/open-data)** under "
    "their Open Data user agreement. Raw data is not redistributed by this project. "
    "See `DATA_SOURCES.md` in the repository for full attribution."
)
