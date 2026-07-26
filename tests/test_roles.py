"""Unit tests for position -> role classification and season role assignment."""

import pandas as pd

from src.features.roles import assign_player_roles, classify_position


def test_classify_positions_to_role_groups():
    assert classify_position("Right Center Back") == "CB"
    assert classify_position("Center Back") == "CB"
    assert classify_position("Left Back") == "FB"
    assert classify_position("Right Wing Back") == "FB"       # Wing Back -> FB, not winger
    assert classify_position("Center Defensive Midfield") == "CM"
    assert classify_position("Left Center Midfield") == "CM"
    assert classify_position("Right Midfield") == "AM_W"       # wide midfield -> winger
    assert classify_position("Center Attacking Midfield") == "AM_W"
    assert classify_position("Left Wing") == "AM_W"
    assert classify_position("Center Forward") == "ST"
    assert classify_position("Secondary Striker") == "ST"


def test_goalkeeper_and_unknown_are_none():
    assert classify_position("Goalkeeper") is None
    assert classify_position(None) is None
    assert classify_position("") is None


def test_assign_role_uses_most_played_position():
    # Player 1 plays mostly CB (two matches) but once as FB -> role CB.
    lineups = pd.DataFrame([
        {"match_id": 1, "player_id": 1, "position": "Center Back", "minutes": 90, "played": True},
        {"match_id": 2, "player_id": 1, "position": "Center Back", "minutes": 90, "played": True},
        {"match_id": 3, "player_id": 1, "position": "Left Back", "minutes": 90, "played": True},
        {"match_id": 1, "player_id": 2, "position": "Goalkeeper", "minutes": 90, "played": True},
    ])
    roles = assign_player_roles(lineups).set_index("player_id")
    assert roles.loc[1, "role"] == "CB"
    assert roles.loc[1, "matches_played"] == 3
    assert roles.loc[1, "minutes"] == 270
    assert 2 not in roles.index  # goalkeeper dropped
