"""Unit tests for minutes calculation (synthetic lineups, no network)."""

from src.transformations.minutes import (
    build_lineup_rows,
    match_end_minute,
    parse_clock,
)


def test_parse_clock_minutes_and_seconds():
    assert parse_clock("00:00") == 0.0
    assert parse_clock("45:30") == 45.5
    assert parse_clock("56:53") == 56 + 53 / 60


def test_parse_clock_with_hours():
    assert parse_clock("1:00:00") == 60.0


def test_match_end_minute_uses_latest_event():
    events = [
        {"minute": 45, "second": 0},
        {"minute": 94, "second": 12},
        {"minute": 90, "second": 30},
    ]
    assert match_end_minute(events) == 94 + 12 / 60


def _lineups():
    return [
        {
            "team_id": 1,
            "team_name": "Test FC",
            "lineup": [
                {  # starter who plays to the whistle
                    "player_id": 10,
                    "player_name": "Starter Stays",
                    "player_nickname": None,
                    "positions": [
                        {"position": "Right Back", "from": "00:00", "to": None,
                         "start_reason": "Starting XI", "end_reason": "Final Whistle"},
                    ],
                },
                {  # starter subbed off at 60'
                    "player_id": 11,
                    "player_name": "Starter Off",
                    "positions": [
                        {"position": "Center Forward", "from": "00:00", "to": "60:00",
                         "start_reason": "Starting XI", "end_reason": "Substitution - Off"},
                    ],
                },
                {  # sub who comes on at 60'
                    "player_id": 12,
                    "player_name": "Sub On",
                    "positions": [
                        {"position": "Center Forward", "from": "60:00", "to": None,
                         "start_reason": "Substitution - On", "end_reason": "Final Whistle"},
                    ],
                },
                {  # unused substitute
                    "player_id": 13,
                    "player_name": "Bench Warmer",
                    "positions": [],
                },
            ],
        }
    ]


def test_build_lineup_rows_minutes_and_flags():
    rows = {r["player_id"]: r for r in build_lineup_rows(1, _lineups(), match_end=95.0)}

    assert rows[10]["minutes"] == 95.0
    assert rows[10]["is_starter"] and rows[10]["played"]

    assert rows[11]["minutes"] == 60.0
    assert rows[11]["is_starter"]

    assert rows[12]["minutes"] == 35.0
    assert not rows[12]["is_starter"] and rows[12]["played"]

    assert rows[13]["minutes"] == 0.0
    assert not rows[13]["played"]
    assert rows[13]["position"] is None


def test_gap_between_spells_is_excluded():
    # A player who goes off temporarily (Player Off) and returns (Player On)
    # must not be credited for the time spent off the pitch.
    lineups = [
        {
            "team_id": 1,
            "team_name": "Test FC",
            "lineup": [
                {
                    "player_id": 30,
                    "player_name": "Off And Back",
                    "positions": [
                        {"position": "Left Back", "from": "00:00", "to": "40:00",
                         "start_reason": "Starting XI", "end_reason": "Player Off"},
                        {"position": "Left Back", "from": "43:00", "to": None,
                         "start_reason": "Player On", "end_reason": "Final Whistle"},
                    ],
                }
            ],
        }
    ]
    row = build_lineup_rows(1, lineups, match_end=95.0)[0]
    # 40 min + (95 - 43) = 92, NOT 95 (the 3-minute gap is excluded).
    assert row["minutes"] == 92.0


def test_primary_position_is_longest_spell():
    lineups = [
        {
            "team_id": 1,
            "team_name": "Test FC",
            "lineup": [
                {
                    "player_id": 20,
                    "player_name": "Mover",
                    "positions": [
                        {"position": "Left Wing", "from": "00:00", "to": "20:00",
                         "start_reason": "Starting XI", "end_reason": "Tactical Shift"},
                        {"position": "Center Forward", "from": "20:00", "to": None,
                         "start_reason": "Tactical Shift", "end_reason": "Final Whistle"},
                    ],
                }
            ],
        }
    ]
    row = build_lineup_rows(1, lineups, match_end=95.0)[0]
    assert row["position"] == "Center Forward"  # 75 min > 20 min
    assert row["minutes"] == 95.0
