"""mart_team_match: playing-style features for one team in one match.

Style is measured per match first, then aggregated to the season, because the
match-to-match *variation* is itself a style signal (see team_season.py).

Two things make team features harder than player features:

1. **Several metrics are relative to the opponent.** Possession share, field tilt
   and PPDA only mean something when both teams in the match are known, so raw
   counts are computed per team and then joined to the opponent's row.
2. **Coordinates are relative to the acting team.** StatsBomb always records
   events with the acting team attacking towards x = 120, so team A pressing at
   x = 60 meets team B playing at x = 60 in B's own frame. PPDA therefore pairs
   A's defensive actions at `x >= 48` with B's passes at `x <= 72` (= 120 - 48).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pitch import (
    ATT_HALF_X,
    FINAL_THIRD_X,
    HIGH_REGAIN_X,
    PITCH_LENGTH,
    PRESSING_ZONE_X,
    dist_to_goal,
    in_box,
    in_halfspace,
    in_wide_channel,
    in_zone14,
)
from .player_match import (
    FORWARD_MIN_METRES,
    LONG_PASS_MIN,
    PROGRESSIVE_MIN_METRES,
    SET_PIECE_PASS_TYPES,
    TOUCH_TYPES,
)

BACKWARD_MAX_METRES = -5.0   # a pass ending this much further from goal is backward

# `play_pattern` describes how the *possession started*, not where the shot came
# from, and it persists for the whole possession. Counting every shot in a
# corner/free-kick/throw-in possession as a "set-piece shot" gives 54% of all
# shots, which is nonsense — a quarter of all events in this season are tagged
# "From Throw In" simply because the phase began that way.
#
# A set-piece shot is therefore a shot from a corner or free-kick possession
# taken within SET_PIECE_WINDOW_SECONDS of the possession starting, i.e. the
# first phase of the routine. That yields 25% of shots, matching the published
# benchmark. Throw-ins are excluded entirely: they restart open play.
SET_PIECE_PATTERNS = {"From Corner", "From Free Kick"}
SET_PIECE_WINDOW_SECONDS = 10.0

# Counters need no such window: a possession tagged "From Counter" is by
# definition a fast direct attack, and the resulting share (~5%) is plausible.
COUNTER_PATTERN = "From Counter"

# Opponent passes are counted in their own 60% — the mirror of the pressing zone.
OPPONENT_PRESSED_ZONE_X = PITCH_LENGTH - PRESSING_ZONE_X

DEFENSIVE_ACTION_TYPES = ["Pressure", "Interception", "Block", "Clearance", "Ball Recovery"]

# Style features produced by compute_team_match, in output order.
STYLE_FEATURES = [
    # Possession & circulation
    "possession_share", "pass_completion", "passes_per_possession", "backward_pass_share",
    # Progression & directness
    "forward_dist_per_pass", "prog_actions_per_100_passes", "long_pass_share",
    "final_third_entries_per_100_passes",
    # Territory
    "field_tilt", "att_third_touch_share", "mean_action_x",
    # Width & crossing
    "wide_touch_share", "halfspace_touch_share", "zone14_entries_per_100_passes",
    "cross_share", "switch_share",
    # Pressing & defensive height
    "ppda", "def_action_height", "high_regain_share",
    # Transitions
    "counter_shot_share",
    # Chance profile
    "shots_per_100_passes", "box_shot_share", "mean_shot_distance", "npxg_per_shot",
    "set_piece_shot_share",
    # Risk & ball security
    "turnovers_per_100_passes", "pressured_pass_completion", "own_half_loss_share",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN (not inf) where the denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def _seconds_into_possession(events: pd.DataFrame) -> pd.Series:
    """How long after its possession began each event happened, in seconds.

    Computed over every event in the possession (including the defending team's,
    which share the possession id) so the start time is the true phase start.
    """
    event_time = events["minute"] * 60 + events["second"]
    start = event_time.groupby(
        [events["match_id"], events["possession"]]
    ).transform("min")
    return event_time - start


def _raw_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Per (match_id, team_id) raw counts, before any opponent-relative maths."""
    work = events.copy()
    work["secs_into_possession"] = _seconds_into_possession(work)
    e = work[work["team_id"].notna()].copy()

    is_pass = e["type"] == "Pass"
    completed = is_pass & e["pass_outcome"].isna()
    open_play = is_pass & ~e["pass_type"].isin(SET_PIECE_PASS_TYPES)
    open_play_completed = open_play & e["pass_outcome"].isna()
    is_carry = e["type"] == "Carry"
    is_touch = e["type"].isin(TOUCH_TYPES)

    start_dist = dist_to_goal(e["location_x"], e["location_y"])
    pass_gain = start_dist - dist_to_goal(e["pass_end_x"], e["pass_end_y"])
    carry_gain = start_dist - dist_to_goal(e["carry_end_x"], e["carry_end_y"])
    forward_delta = e["pass_end_x"] - e["location_x"]

    e["all_passes"] = is_pass.astype(int)
    e["passes"] = open_play.astype(int)
    e["passes_completed"] = open_play_completed.astype(int)
    e["passes_backward"] = (open_play_completed & (forward_delta <= BACKWARD_MAX_METRES)).astype(int)
    e["passes_long"] = (open_play & (e["pass_length"] >= LONG_PASS_MIN)).astype(int)
    e["forward_dist_sum"] = np.where(open_play_completed, forward_delta.fillna(0.0), 0.0)

    # Opponent-facing: passes played inside the team's own 60% of the pitch.
    e["passes_in_own_60"] = (is_pass & (e["location_x"] <= OPPONENT_PRESSED_ZONE_X)).astype(int)

    e["prog_passes"] = (open_play_completed & (pass_gain >= PROGRESSIVE_MIN_METRES)).astype(int)
    e["prog_carries"] = (is_carry & (carry_gain >= PROGRESSIVE_MIN_METRES)).astype(int)
    outside_ft = e["location_x"] < FINAL_THIRD_X
    e["final_third_entries"] = (
        (open_play_completed & outside_ft & (e["pass_end_x"] >= FINAL_THIRD_X))
        | (is_carry & outside_ft & (e["carry_end_x"] >= FINAL_THIRD_X))
    ).astype(int)
    in_z14 = in_zone14(e["location_x"], e["location_y"])
    e["zone14_entries"] = (
        (open_play_completed & ~in_z14 & in_zone14(e["pass_end_x"], e["pass_end_y"]))
        | (is_carry & ~in_z14 & in_zone14(e["carry_end_x"], e["carry_end_y"]))
    ).astype(int)

    # Field tilt uses completed passes inside the final third.
    e["final_third_passes"] = (completed & (e["location_x"] >= FINAL_THIRD_X)).astype(int)

    e["crosses"] = (is_pass & e["pass_cross"]).astype(int)
    e["switches"] = (is_pass & e["pass_switch"]).astype(int)

    e["touches"] = is_touch.astype(int)
    e["touches_x_sum"] = np.where(is_touch, e["location_x"].fillna(0.0), 0.0)
    e["touches_att_third"] = (is_touch & (e["location_x"] >= FINAL_THIRD_X)).astype(int)
    e["touches_att_half"] = (is_touch & (e["location_x"] >= ATT_HALF_X)).astype(int)
    e["wide_touches"] = (is_touch & in_wide_channel(e["location_x"], e["location_y"])).astype(int)
    e["halfspace_touches"] = (is_touch & in_halfspace(e["location_x"], e["location_y"])).astype(int)

    is_tackle = (e["type"] == "Duel") & (e["duel_type"] == "Tackle")
    def_action = e["type"].isin(DEFENSIVE_ACTION_TYPES) | is_tackle
    e["def_actions"] = def_action.astype(int)
    e["def_actions_x_sum"] = np.where(def_action, e["location_x"].fillna(0.0), 0.0)
    e["def_actions_high"] = (def_action & (e["location_x"] >= PRESSING_ZONE_X)).astype(int)
    is_recovery = e["type"] == "Ball Recovery"
    e["recoveries"] = is_recovery.astype(int)
    e["recoveries_high"] = (is_recovery & (e["location_x"] >= HIGH_REGAIN_X)).astype(int)

    np_shot = (e["type"] == "Shot") & (e["shot_type"] != "Penalty")
    e["shots"] = np_shot.astype(int)
    e["npxg"] = np.where(np_shot, e["shot_xg"].fillna(0.0), 0.0)
    e["box_shots"] = (np_shot & in_box(e["location_x"], e["location_y"])).astype(int)
    e["shot_dist_sum"] = np.where(np_shot, start_dist.fillna(0.0), 0.0)
    e["counter_shots"] = (np_shot & (e["play_pattern"] == COUNTER_PATTERN)).astype(int)
    e["set_piece_shots"] = (
        np_shot
        & e["play_pattern"].isin(SET_PIECE_PATTERNS)
        & (e["secs_into_possession"] <= SET_PIECE_WINDOW_SECONDS)
    ).astype(int)

    failed_pressured_pass = is_pass & e["pass_outcome"].notna() & e["under_pressure"]
    turnover = (e["type"] == "Miscontrol") | (e["type"] == "Dispossessed") | failed_pressured_pass
    e["turnovers"] = turnover.astype(int)
    e["turnovers_own_half"] = (turnover & (e["location_x"] < ATT_HALF_X)).astype(int)
    e["pressured_passes"] = (is_pass & e["under_pressure"]).astype(int)
    e["pressured_passes_completed"] = (completed & e["under_pressure"]).astype(int)

    count_columns = [
        "all_passes", "passes", "passes_completed", "passes_backward", "passes_long",
        "forward_dist_sum", "passes_in_own_60", "prog_passes", "prog_carries",
        "final_third_entries", "zone14_entries", "final_third_passes",
        "crosses", "switches",
        "touches", "touches_x_sum", "touches_att_third", "touches_att_half",
        "wide_touches", "halfspace_touches",
        "def_actions", "def_actions_x_sum", "def_actions_high",
        "recoveries", "recoveries_high",
        "shots", "npxg", "box_shots", "shot_dist_sum", "counter_shots", "set_piece_shots",
        "turnovers", "turnovers_own_half", "pressured_passes", "pressured_passes_completed",
    ]
    counts = e.groupby(["match_id", "team_id", "team"], as_index=False)[count_columns].sum()

    # Possession sequences belonging to this team.
    possessions = (
        e[e["possession_team"] == e["team"]]
        .groupby(["match_id", "team_id"])["possession"]
        .nunique()
        .rename("n_possessions")
        .reset_index()
    )
    return counts.merge(possessions, on=["match_id", "team_id"], how="left")


def _attach_opponent(counts: pd.DataFrame) -> pd.DataFrame:
    """Join each team-match row to the other team's row in the same match."""
    opponent_columns = ["all_passes", "final_third_passes", "passes_in_own_60", "shots", "npxg"]
    slim = counts[["match_id", "team_id", "team"] + opponent_columns]
    pairs = counts.merge(slim, on="match_id", suffixes=("", "_opp"))
    return pairs[pairs["team_id"] != pairs["team_id_opp"]].reset_index(drop=True)


def compute_team_match(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (match_id, team_id) with playing-style features."""
    counts = _raw_counts(events)
    df = _attach_opponent(counts)

    passes = df["passes"]
    completed = df["passes_completed"]
    shots = df["shots"]
    touches = df["touches"]
    per_100 = passes / 100

    out = df[["match_id", "team_id", "team", "team_opp"]].copy()
    out = out.rename(columns={"team_opp": "opponent"})

    # Possession & circulation
    out["possession_share"] = _safe_div(df["all_passes"], df["all_passes"] + df["all_passes_opp"])
    out["pass_completion"] = _safe_div(completed, passes)
    out["passes_per_possession"] = _safe_div(completed, df["n_possessions"])
    out["backward_pass_share"] = _safe_div(df["passes_backward"], completed)

    # Progression & directness
    out["forward_dist_per_pass"] = _safe_div(df["forward_dist_sum"], completed)
    out["prog_actions_per_100_passes"] = _safe_div(df["prog_passes"] + df["prog_carries"], per_100)
    out["long_pass_share"] = _safe_div(df["passes_long"], passes)
    out["final_third_entries_per_100_passes"] = _safe_div(df["final_third_entries"], per_100)

    # Territory
    out["field_tilt"] = _safe_div(
        df["final_third_passes"], df["final_third_passes"] + df["final_third_passes_opp"])
    out["att_third_touch_share"] = _safe_div(df["touches_att_third"], touches)
    out["mean_action_x"] = _safe_div(df["touches_x_sum"], touches)

    # Width & crossing
    out["wide_touch_share"] = _safe_div(df["wide_touches"], df["touches_att_half"])
    out["halfspace_touch_share"] = _safe_div(df["halfspace_touches"], df["touches_att_half"])
    out["zone14_entries_per_100_passes"] = _safe_div(df["zone14_entries"], per_100)
    out["cross_share"] = _safe_div(df["crosses"], df["final_third_entries"])
    out["switch_share"] = _safe_div(df["switches"], passes)

    # Pressing & defensive height — lower PPDA means more aggressive pressing.
    out["ppda"] = _safe_div(df["passes_in_own_60_opp"], df["def_actions_high"])
    out["def_action_height"] = _safe_div(df["def_actions_x_sum"], df["def_actions"])
    out["high_regain_share"] = _safe_div(df["recoveries_high"], df["recoveries"])

    # Transitions
    out["counter_shot_share"] = _safe_div(df["counter_shots"], shots)

    # Chance profile
    out["shots_per_100_passes"] = _safe_div(shots, per_100)
    out["box_shot_share"] = _safe_div(df["box_shots"], shots)
    out["mean_shot_distance"] = _safe_div(df["shot_dist_sum"], shots)
    out["npxg_per_shot"] = _safe_div(df["npxg"], shots)
    out["set_piece_shot_share"] = _safe_div(df["set_piece_shots"], shots)

    # Risk & ball security
    out["turnovers_per_100_passes"] = _safe_div(df["turnovers"], per_100)
    out["pressured_pass_completion"] = _safe_div(
        df["pressured_passes_completed"], df["pressured_passes"])
    out["own_half_loss_share"] = _safe_div(df["turnovers_own_half"], df["turnovers"])

    return out.sort_values(["match_id", "team_id"]).reset_index(drop=True)
