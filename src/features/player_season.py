"""mart_player_season: one row per player with season features for similarity.

Takes the per-match counts from player_match, sums them to season totals, and
turns them into the interpretable metrics defined in the metric dictionary:

    - per-90 values for volume metrics (count / minutes * 90)
    - rates/shares with minimum-attempt guards (null below the threshold)
    - possession-adjusted defensive volume (pressures/tackles/interceptions)
    - percentile-within-role for every metric (what users actually read)

Standardization and winsorization for the similarity model happen in Phase 4;
this table holds the human-readable values and their percentiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Defensive volume metrics that get possession-adjusted.
_PADJ_BASE = {"pressures": "padj_pressures_p90",
              "tackles": "padj_tackles_p90",
              "interceptions": "padj_interceptions_p90"}

# The metric columns whose percentile-within-role we expose (pct_<name>).
METRIC_COLUMNS = [
    "pass_attempts_p90", "pass_completion_pct", "forward_pass_share", "long_pass_share",
    "turnovers_p90",
    "prog_passes_p90", "prog_carries_p90", "final_third_entries_p90", "box_entries_p90",
    "key_passes_p90", "xg_assisted_p90", "crosses_box_p90", "through_balls_p90",
    "shots_p90", "npxg_p90", "box_shot_share", "npxg_per_shot",
    "padj_pressures_p90", "padj_tackles_p90", "padj_interceptions_p90",
    "blocks_p90", "recoveries_p90", "def_action_height",
    "dribble_success_pct", "aerial_involvement_p90", "aerial_win_pct",
    "touches_att_third_share", "receptions_final_third_p90", "pressured_actions_share",
    "zone14_receptions_p90", "zone14_entries_p90",
    "halfspace_touch_share", "wide_touch_share",
]


def team_possession_share(events: pd.DataFrame) -> pd.DataFrame:
    """Season possession share per team = its passes / all passes in its matches."""
    passes = events[events["type"] == "Pass"]
    per_team_match = passes.groupby(["match_id", "team_id"]).size().rename("team_passes")
    per_match = passes.groupby("match_id").size().rename("match_passes")
    joined = per_team_match.reset_index().merge(per_match, on="match_id")
    agg = joined.groupby("team_id").agg(team_passes=("team_passes", "sum"),
                                        match_passes=("match_passes", "sum"))
    agg["possession_share"] = agg["team_passes"] / agg["match_passes"]
    return agg.reset_index()[["team_id", "possession_share"]]


def _p90(count: pd.Series, minutes: pd.Series) -> pd.Series:
    return count / minutes * 90


def _guarded_rate(numerator: pd.Series, denominator: pd.Series, min_denom: float) -> pd.Series:
    """numerator/denominator, but null where denominator is below min_denom."""
    rate = numerator / denominator
    return rate.where(denominator >= min_denom)


def build_player_season(
    player_match: pd.DataFrame,
    fact_lineups: pd.DataFrame,
    roles: pd.DataFrame,
    team_possession: pd.DataFrame,
    dim_players: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    thresholds = config["thresholds"]
    min_minutes = thresholds["min_minutes_season"]

    # Season count totals per player (summed across matches and any team change).
    count_cols = [c for c in player_match.columns
                  if c not in ("match_id", "player_id", "team_id")]
    season = player_match.groupby("player_id", as_index=False)[count_cols].sum()

    # Dominant team per player (most minutes) -> its possession share.
    played = fact_lineups[fact_lineups["played"]]
    minutes_by_team = played.groupby(["player_id", "team_id"])["minutes"].sum().reset_index()
    dominant_team = (minutes_by_team.sort_values("minutes", ascending=False)
                     .drop_duplicates("player_id")[["player_id", "team_id"]])

    season = season.merge(roles[["player_id", "role", "minutes", "matches_played",
                                  "primary_position"]], on="player_id", how="inner")
    season = season.merge(dominant_team, on="player_id", how="left")
    season = season.merge(team_possession, on="team_id", how="left")
    season = season.merge(dim_players[["player_id", "player_name", "nickname"]],
                          on="player_id", how="left")

    # Apply the minutes threshold: below it, season per-90 metrics are unreliable.
    season = season[season["minutes"] >= min_minutes].reset_index(drop=True)

    m = season["minutes"]

    # --- Passing & security ---
    season["pass_attempts_p90"] = _p90(season["passes"], m)
    season["pass_completion_pct"] = _guarded_rate(
        season["passes_completed"], season["passes"], thresholds["min_pass_attempts_for_rate"])
    season["forward_pass_share"] = _guarded_rate(
        season["passes_forward"], season["passes_completed"], thresholds["min_pass_attempts_for_rate"])
    season["long_pass_share"] = _guarded_rate(
        season["passes_long"], season["passes"], thresholds["min_pass_attempts_for_rate"])
    season["turnovers_p90"] = _p90(season["turnovers"], m)

    # --- Progression ---
    season["prog_passes_p90"] = _p90(season["prog_passes"], m)
    season["prog_carries_p90"] = _p90(season["prog_carries"], m)
    season["final_third_entries_p90"] = _p90(season["final_third_entries"], m)
    season["box_entries_p90"] = _p90(season["box_entries"], m)

    # --- Creation ---
    season["key_passes_p90"] = _p90(season["key_passes"], m)
    season["xg_assisted_p90"] = _p90(season["xg_assisted"], m)
    season["crosses_box_p90"] = _p90(season["crosses_box"], m)
    season["through_balls_p90"] = _p90(season["through_balls"], m)

    # --- Shooting (non-penalty) ---
    season["shots_p90"] = _p90(season["shots"], m)
    season["npxg_p90"] = _p90(season["npxg"], m)
    season["box_shot_share"] = _guarded_rate(
        season["box_shots"], season["shots"], thresholds["min_shots_for_rate"])
    season["npxg_per_shot"] = _guarded_rate(
        season["npxg"], season["shots"], thresholds["min_shots_for_rate"])
    season["np_goals_minus_npxg"] = season["np_goals"] - season["npxg"]  # display only

    # --- Defending (possession-adjusted volume) ---
    padj_factor = 0.5 / (1 - season["possession_share"])
    for base, target in _PADJ_BASE.items():
        season[target] = _p90(season[base], m) * padj_factor
    season["blocks_p90"] = _p90(season["blocks"], m)
    season["recoveries_p90"] = _p90(season["recoveries"], m)
    season["def_action_height"] = (season["def_actions_x_sum"] / season["def_actions"]).where(
        season["def_actions"] > 0)

    # --- Duels & aerials ---
    season["dribble_success_pct"] = _guarded_rate(
        season["dribbles_completed"], season["dribbles"], thresholds["min_dribbles_for_rate"])
    aerials = season["aerials_won"] + season["aerials_lost"]
    season["aerial_involvement_p90"] = _p90(aerials, m)
    season["aerial_win_pct"] = _guarded_rate(
        season["aerials_won"], aerials, thresholds["min_aerials_for_rate"])

    # --- Off-ball context ---
    season["touches_att_third_share"] = _guarded_rate(
        season["touches_att_third"], season["touches"], thresholds["min_pass_attempts_for_rate"])
    season["receptions_final_third_p90"] = _p90(season["receptions_final_third"], m)
    season["pressured_actions_share"] = _guarded_rate(
        season["actions_pressured"], season["touches"], thresholds["min_pass_attempts_for_rate"])

    # --- Tactical zones ---
    # Shares are taken over attacking-half touches: "of the work you do in the
    # attacking half, how much is in the halfspaces vs out wide?"
    min_att_half = thresholds["min_att_half_touches_for_share"]
    season["zone14_receptions_p90"] = _p90(season["zone14_receptions"], m)
    season["zone14_entries_p90"] = _p90(season["zone14_entries"], m)
    season["halfspace_touch_share"] = _guarded_rate(
        season["halfspace_touches"], season["touches_att_half"], min_att_half)
    season["wide_touch_share"] = _guarded_rate(
        season["wide_touches"], season["touches_att_half"], min_att_half)

    # Percentile-within-role for every metric (rank-based, robust to outliers).
    for col in METRIC_COLUMNS:
        season[f"pct_{col}"] = season.groupby("role")[col].rank(pct=True)

    season["low_minutes"] = season["minutes"] < thresholds["low_minutes_warning"]

    identity = ["player_id", "player_name", "nickname", "role", "primary_position",
                "team_id", "minutes", "matches_played", "possession_share", "low_minutes"]
    ordered = identity + METRIC_COLUMNS + ["np_goals_minus_npxg"] + [f"pct_{c}" for c in METRIC_COLUMNS]
    return season[ordered].sort_values(["role", "player_name"]).reset_index(drop=True)
