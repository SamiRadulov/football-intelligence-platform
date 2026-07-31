"""mart_player_match: raw metric counts per player per match, from fact_events.

This module encodes the operational definitions in docs/METRIC_DICTIONARY.md as
per-event flags, then sums them per (match, player). It deliberately stores
**counts** (numerators and denominators), not rates: per-90 values and shares
are computed later at season level, where summing counts first and dividing once
is the statistically correct order.

Pitch geometry (StatsBomb units, 120 x 80, attacking towards x = 120):
    - opponent goal centre = (120, 40)
    - penalty box = x >= 102 and 18 <= y <= 62
    - final third = x >= 80
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pitch import (  # noqa: F401 - re-exported for convenience
    ATT_HALF_X,
    BOX_X_MIN,
    BOX_Y_MAX,
    BOX_Y_MIN,
    FINAL_THIRD_X,
    GOAL_X,
    GOAL_Y,
    dist_to_goal,
    in_box,
    in_halfspace,
    in_wide_channel,
    in_zone14,
)

FORWARD_MIN_METRES = 5.0        # forward pass: end at least this much nearer the goal line
PROGRESSIVE_MIN_METRES = 10.0   # progressive: reduces distance-to-goal by at least this
LONG_PASS_MIN = 30.0            # long pass: pass length at least this (pitch units)

# Passes we treat as set pieces and exclude from open-play passing volume.
SET_PIECE_PASS_TYPES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}

# On-ball "touch" event types.
TOUCH_TYPES = {"Pass", "Ball Receipt*", "Carry", "Shot", "Dribble", "Clearance", "Miscontrol"}


def _xg_assisted(events: pd.DataFrame) -> pd.DataFrame:
    """Sum the xG of shots back onto the player whose pass created them.

    Shots carry `shot_key_pass_id` pointing at the assisting pass event; we join
    that to the pass to credit its passer (this is expected-assists / xA).
    """
    shots = events[(events["type"] == "Shot") & events["shot_key_pass_id"].notna()]
    shots = shots[["match_id", "shot_key_pass_id", "shot_xg"]]
    passes = events[events["type"] == "Pass"][["match_id", "event_id", "player_id", "team_id"]]
    merged = shots.merge(
        passes, left_on=["match_id", "shot_key_pass_id"], right_on=["match_id", "event_id"]
    )
    return (
        merged.groupby(["match_id", "player_id", "team_id"])["shot_xg"]
        .sum()
        .reset_index()
        .rename(columns={"shot_xg": "xg_assisted"})
    )


def compute_player_match(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (match_id, player_id, team_id) with raw metric counts."""
    e = events[events["player_id"].notna()].copy()

    is_pass = e["type"] == "Pass"
    completed = is_pass & e["pass_outcome"].isna()  # null outcome = completed pass
    open_play_pass = is_pass & ~e["pass_type"].isin(SET_PIECE_PASS_TYPES)
    open_play_completed = open_play_pass & e["pass_outcome"].isna()

    start_dist = dist_to_goal(e["location_x"], e["location_y"])
    pass_end_dist = dist_to_goal(e["pass_end_x"], e["pass_end_y"])
    carry_end_dist = dist_to_goal(e["carry_end_x"], e["carry_end_y"])
    is_carry = e["type"] == "Carry"

    # --- Passing & security ---
    e["passes"] = open_play_pass.astype(int)
    e["passes_completed"] = open_play_completed.astype(int)
    e["passes_forward"] = (open_play_completed & (e["pass_end_x"] - e["location_x"] >= FORWARD_MIN_METRES)).astype(int)
    e["passes_long"] = (open_play_pass & (e["pass_length"] >= LONG_PASS_MIN)).astype(int)

    # --- Progression ---
    e["prog_passes"] = (open_play_completed & (start_dist - pass_end_dist >= PROGRESSIVE_MIN_METRES)).astype(int)
    e["prog_carries"] = (is_carry & (start_dist - carry_end_dist >= PROGRESSIVE_MIN_METRES)).astype(int)
    enters_final_third = (e["location_x"] < FINAL_THIRD_X)
    pass_ft = open_play_completed & enters_final_third & (e["pass_end_x"] >= FINAL_THIRD_X)
    carry_ft = is_carry & enters_final_third & (e["carry_end_x"] >= FINAL_THIRD_X)
    e["final_third_entries"] = (pass_ft | carry_ft).astype(int)
    pass_box = open_play_completed & ~in_box(e["location_x"], e["location_y"]) & in_box(e["pass_end_x"], e["pass_end_y"])
    carry_box = is_carry & ~in_box(e["location_x"], e["location_y"]) & in_box(e["carry_end_x"], e["carry_end_y"])
    e["box_entries"] = (pass_box | carry_box).astype(int)

    # --- Creation ---
    e["key_passes"] = (is_pass & e["pass_shot_assist"]).astype(int)
    e["crosses_box"] = (completed & e["pass_cross"] & in_box(e["pass_end_x"], e["pass_end_y"])).astype(int)
    e["through_balls"] = (open_play_completed & e["pass_through_ball"]).astype(int)

    # --- Shooting (non-penalty) ---
    is_shot = e["type"] == "Shot"
    np_shot = is_shot & (e["shot_type"] != "Penalty")
    e["shots"] = np_shot.astype(int)
    e["npxg"] = np.where(np_shot, e["shot_xg"].fillna(0.0), 0.0)
    e["np_goals"] = (np_shot & (e["shot_outcome"] == "Goal")).astype(int)
    e["box_shots"] = (np_shot & in_box(e["location_x"], e["location_y"])).astype(int)

    # --- Defending ---
    is_tackle = (e["type"] == "Duel") & (e["duel_type"] == "Tackle")
    e["pressures"] = (e["type"] == "Pressure").astype(int)
    e["tackles"] = is_tackle.astype(int)
    e["interceptions"] = (e["type"] == "Interception").astype(int)
    e["blocks"] = (e["type"] == "Block").astype(int)
    e["recoveries"] = (e["type"] == "Ball Recovery").astype(int)
    def_action = e["type"].isin(["Pressure", "Interception", "Block", "Clearance", "Ball Recovery"]) | is_tackle
    e["def_actions"] = def_action.astype(int)
    e["def_actions_x_sum"] = np.where(def_action, e["location_x"].fillna(0.0), 0.0)

    # --- Duels & aerials ---
    is_dribble = e["type"] == "Dribble"
    e["dribbles"] = is_dribble.astype(int)
    e["dribbles_completed"] = (is_dribble & (e["dribble_outcome"] == "Complete")).astype(int)
    e["aerials_won"] = e["aerial_won"].astype(int)
    e["aerials_lost"] = ((e["type"] == "Duel") & (e["duel_type"] == "Aerial Lost")).astype(int)

    # --- Off-ball context ---
    is_touch = e["type"].isin(TOUCH_TYPES)
    e["touches"] = is_touch.astype(int)
    e["touches_att_third"] = (is_touch & (e["location_x"] >= FINAL_THIRD_X)).astype(int)
    receipt_complete = (e["type"] == "Ball Receipt*") & e["ball_receipt_outcome"].isna()
    e["receptions_final_third"] = (receipt_complete & (e["location_x"] >= FINAL_THIRD_X)).astype(int)
    e["actions_pressured"] = (is_touch & e["under_pressure"]).astype(int)

    # --- Tactical zones (attacking half) ---
    # Receiving the ball in Zone 14 is the sharpest signal of a player finding
    # space between the lines, which a pure carry/entry count would miss.
    in_z14 = in_zone14(e["location_x"], e["location_y"])
    e["touches_att_half"] = (is_touch & (e["location_x"] >= ATT_HALF_X)).astype(int)
    e["zone14_receptions"] = (receipt_complete & in_z14).astype(int)
    pass_z14 = open_play_completed & ~in_z14 & in_zone14(e["pass_end_x"], e["pass_end_y"])
    carry_z14 = is_carry & ~in_z14 & in_zone14(e["carry_end_x"], e["carry_end_y"])
    e["zone14_entries"] = (pass_z14 | carry_z14).astype(int)
    e["halfspace_touches"] = (is_touch & in_halfspace(e["location_x"], e["location_y"])).astype(int)
    e["wide_touches"] = (is_touch & in_wide_channel(e["location_x"], e["location_y"])).astype(int)

    # --- Turnovers ---
    failed_pressured_pass = is_pass & e["pass_outcome"].notna() & e["under_pressure"]
    e["turnovers"] = (
        (e["type"] == "Miscontrol") | (e["type"] == "Dispossessed") | failed_pressured_pass
    ).astype(int)

    count_columns = [
        "passes", "passes_completed", "passes_forward", "passes_long",
        "prog_passes", "prog_carries", "final_third_entries", "box_entries",
        "key_passes", "crosses_box", "through_balls",
        "shots", "npxg", "np_goals", "box_shots",
        "pressures", "tackles", "interceptions", "blocks", "recoveries",
        "def_actions", "def_actions_x_sum",
        "dribbles", "dribbles_completed", "aerials_won", "aerials_lost",
        "touches", "touches_att_third", "receptions_final_third", "actions_pressured",
        "touches_att_half", "zone14_receptions", "zone14_entries",
        "halfspace_touches", "wide_touches",
        "turnovers",
    ]
    per_match = e.groupby(["match_id", "player_id", "team_id"], as_index=False)[count_columns].sum()

    xa = _xg_assisted(events)
    per_match = per_match.merge(xa, on=["match_id", "player_id", "team_id"], how="left")
    per_match["xg_assisted"] = per_match["xg_assisted"].fillna(0.0)
    return per_match
