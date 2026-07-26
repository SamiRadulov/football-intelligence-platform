"""Unit tests for per-match metric computation (synthetic events, no network)."""

import pandas as pd

from src.features.player_match import compute_player_match

_DEFAULTS = {
    "event_id": None, "match_id": 1, "team_id": 1, "player_id": 1, "type": None,
    "pass_outcome": None, "pass_type": None, "location_x": None, "location_y": None,
    "pass_end_x": None, "pass_end_y": None, "carry_end_x": None, "carry_end_y": None,
    "pass_length": None, "pass_cross": False, "pass_through_ball": False,
    "pass_shot_assist": False, "shot_xg": None, "shot_outcome": None, "shot_type": None,
    "shot_key_pass_id": None, "under_pressure": False, "duel_type": None,
    "dribble_outcome": None, "ball_receipt_outcome": None, "aerial_won": False,
}


# Numeric columns are float64 in the real Parquet-backed table.
_NUMERIC = ["location_x", "location_y", "pass_end_x", "pass_end_y",
            "carry_end_x", "carry_end_y", "pass_length", "shot_xg"]


def _events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**_DEFAULTS, **r, "event_id": f"e{i}"} for i, r in enumerate(rows)])
    for col in _NUMERIC:
        df[col] = df[col].astype("float64")
    return df


def _player_row(rows: list[dict]):
    out = compute_player_match(_events(rows))
    return out.set_index("player_id").loc[1]


def test_forward_and_progressive_pass():
    # (50,40) -> (65,40): +15 forward, distance-to-goal cut by 15 -> progressive.
    row = _player_row([
        {"type": "Pass", "location_x": 50, "location_y": 40, "pass_end_x": 65, "pass_end_y": 40},
    ])
    assert row["passes"] == 1 and row["passes_completed"] == 1
    assert row["passes_forward"] == 1
    assert row["prog_passes"] == 1


def test_short_sideways_pass_is_not_progressive():
    row = _player_row([
        {"type": "Pass", "location_x": 50, "location_y": 40, "pass_end_x": 52, "pass_end_y": 45},
    ])
    assert row["passes_forward"] == 0
    assert row["prog_passes"] == 0


def test_box_entry_by_pass():
    # start outside the final third (x=70), end inside the penalty area.
    row = _player_row([
        {"type": "Pass", "location_x": 70, "location_y": 40, "pass_end_x": 110, "pass_end_y": 40},
    ])
    assert row["box_entries"] == 1
    assert row["final_third_entries"] == 1  # crossed x=80 as well


def test_pass_starting_in_final_third_is_not_an_entry():
    row = _player_row([
        {"type": "Pass", "location_x": 95, "location_y": 40, "pass_end_x": 110, "pass_end_y": 40},
    ])
    assert row["final_third_entries"] == 0  # already in the final third
    assert row["box_entries"] == 1          # but still enters the box


def test_key_pass_and_xg_assist_link():
    rows = [
        {"type": "Pass", "player_id": 1, "pass_shot_assist": True,
         "location_x": 80, "location_y": 40, "pass_end_x": 100, "pass_end_y": 40},
        {"type": "Shot", "player_id": 2, "shot_type": "Open Play", "shot_xg": 0.4,
         "shot_key_pass_id": "e0", "location_x": 105, "location_y": 40, "shot_outcome": "Goal"},
    ]
    out = compute_player_match(_events(rows)).set_index("player_id")
    assert out.loc[1, "key_passes"] == 1
    assert round(out.loc[1, "xg_assisted"], 3) == 0.4   # credited to the passer


def test_penalty_shot_excluded_from_np_metrics():
    row = _player_row([
        {"type": "Shot", "shot_type": "Penalty", "shot_xg": 0.76,
         "location_x": 108, "location_y": 40, "shot_outcome": "Goal"},
        {"type": "Shot", "shot_type": "Open Play", "shot_xg": 0.2,
         "location_x": 106, "location_y": 40, "shot_outcome": "Off T"},
    ])
    assert row["shots"] == 1                    # penalty not counted
    assert round(row["npxg"], 3) == 0.2
    assert row["np_goals"] == 0


def test_aerials_and_turnovers():
    row = _player_row([
        {"type": "Clearance", "aerial_won": True, "location_x": 20, "location_y": 40},
        {"type": "Duel", "duel_type": "Aerial Lost", "location_x": 25, "location_y": 40},
        {"type": "Miscontrol", "location_x": 60, "location_y": 40},
        {"type": "Dispossessed", "location_x": 70, "location_y": 40},
        {"type": "Pass", "pass_outcome": "Incomplete", "under_pressure": True,
         "location_x": 50, "location_y": 40, "pass_end_x": 60, "pass_end_y": 40},
    ])
    assert row["aerials_won"] == 1 and row["aerials_lost"] == 1
    assert row["turnovers"] == 3   # miscontrol + dispossessed + failed pressured pass
