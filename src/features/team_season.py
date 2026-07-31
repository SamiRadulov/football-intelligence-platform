"""mart_team_season: season style profile per team, with variability.

Each style feature is aggregated from the team's matches into three columns:

    <feature>_mean   the team's typical value
    <feature>_sd     how much it varies match to match
    pct_<feature>    percentile of the mean across the league

The standard deviation is a first-class output, not noise. A side that always
plays the same way and a side that adapts heavily to the opponent can share an
identical mean while being tactically very different, and forcing every match
into one identity would hide that.
"""

from __future__ import annotations

import pandas as pd

from .team_match import STYLE_FEATURES


def build_team_season(team_match: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-match style features to one row per team."""
    grouped = team_match.groupby(["team_id", "team"])

    aggregated = grouped[STYLE_FEATURES].agg(["mean", "std"])
    aggregated.columns = [f"{feature}_{stat.replace('std', 'sd')}"
                          for feature, stat in aggregated.columns]
    aggregated = aggregated.reset_index()

    aggregated.insert(2, "matches", grouped.size().to_numpy())

    for feature in STYLE_FEATURES:
        aggregated[f"pct_{feature}"] = aggregated[f"{feature}_mean"].rank(pct=True)

    return aggregated.sort_values("team").reset_index(drop=True)


def feature_columns(kind: str = "mean") -> list[str]:
    """Column names for the mean, sd or percentile view of the style features."""
    if kind == "pct":
        return [f"pct_{f}" for f in STYLE_FEATURES]
    if kind in ("mean", "sd"):
        return [f"{f}_{kind}" for f in STYLE_FEATURES]
    raise ValueError(f"unknown kind {kind!r}; expected 'mean', 'sd' or 'pct'")
