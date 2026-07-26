"""Unit tests for event flattening (synthetic events, no network)."""

import pandas as pd

from src.transformations.events import CANONICAL_COLUMNS, flatten_events


def _events():
    return [
        {  # a match-control event with no player and no location
            "id": "e1", "index": 1, "period": 1, "timestamp": "00:00:00.000",
            "minute": 0, "second": 0, "type": {"name": "Half Start"},
            "possession": 1, "possession_team": {"name": "A"},
            "play_pattern": {"name": "Regular Play"}, "team": {"id": 1, "name": "A"},
        },
        {  # a completed pass (pass.outcome absent means complete)
            "id": "e2", "index": 2, "period": 1, "timestamp": "00:00:01.000",
            "minute": 0, "second": 1, "type": {"name": "Pass"},
            "possession": 1, "possession_team": {"name": "A"},
            "play_pattern": {"name": "Regular Play"}, "team": {"id": 1, "name": "A"},
            "player": {"id": 100, "name": "Passer"}, "position": {"name": "Center Back"},
            "location": [30.0, 40.0], "under_pressure": True,
            "related_events": ["e3", "e4"],
            "pass": {"length": 15.0, "angle": 0.1, "height": {"name": "Ground Pass"},
                     "end_location": [45.0, 42.0], "recipient": {"id": 101, "name": "Mate"},
                     "switch": True, "aerial_won": True},
        },
    ]


def test_flatten_has_canonical_schema():
    df = flatten_events(999, _events())
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 2
    assert (df["match_id"] == 999).all()


def test_location_is_split_into_x_y():
    df = flatten_events(999, _events())
    pass_row = df[df["type"] == "Pass"].iloc[0]
    assert pass_row["location_x"] == 30.0
    assert pass_row["location_y"] == 40.0
    assert pass_row["pass_end_x"] == 45.0
    assert pass_row["pass_end_y"] == 42.0


def test_boolean_flags_default_false_and_present_true():
    df = flatten_events(999, _events())
    half_start = df[df["type"] == "Half Start"].iloc[0]
    pass_row = df[df["type"] == "Pass"].iloc[0]
    assert half_start["under_pressure"] is False or half_start["under_pressure"] == False  # noqa: E712
    assert bool(pass_row["under_pressure"]) is True
    assert bool(pass_row["pass_switch"]) is True
    assert bool(pass_row["pass_cross"]) is False  # absent -> False


def test_missing_player_id_is_null():
    df = flatten_events(999, _events())
    half_start = df[df["type"] == "Half Start"].iloc[0]
    assert pd.isna(half_start["player_id"])


def test_nested_aerial_won_is_coalesced():
    # aerial_won lives under the type block (pass.aerial_won), not top-level.
    df = flatten_events(999, _events())
    pass_row = df[df["type"] == "Pass"].iloc[0]
    half_start = df[df["type"] == "Half Start"].iloc[0]
    assert bool(pass_row["aerial_won"]) is True
    assert bool(half_start["aerial_won"]) is False


def test_related_events_joined_to_string():
    df = flatten_events(999, _events())
    pass_row = df[df["type"] == "Pass"].iloc[0]
    assert pass_row["related_events"] == "e3;e4"
