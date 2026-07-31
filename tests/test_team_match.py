"""Unit tests for team style features (synthetic matches, no network)."""

import pandas as pd
import pytest

from src.features.team_match import compute_team_match
from src.features.team_season import build_team_season

_DEFAULTS = {
    "match_id": 1, "team_id": 1, "team": "A", "type": None, "minute": 0, "second": 0,
    "possession": 1, "possession_team": "A", "play_pattern": "Regular Play",
    "location_x": 60.0, "location_y": 40.0, "pass_end_x": None, "pass_end_y": None,
    "carry_end_x": None, "carry_end_y": None, "pass_outcome": None, "pass_type": None,
    "pass_length": None, "pass_cross": False, "pass_switch": False,
    "shot_xg": None, "shot_type": None, "under_pressure": False, "duel_type": None,
}
_NUMERIC = ["location_x", "location_y", "pass_end_x", "pass_end_y",
            "carry_end_x", "carry_end_y", "pass_length", "shot_xg"]


def _events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**_DEFAULTS, **r} for r in rows])
    for col in _NUMERIC:
        df[col] = df[col].astype("float64")
    return df


def _pass(team_id, team, **kw):
    return {"type": "Pass", "team_id": team_id, "team": team,
            "pass_end_x": 65.0, "pass_end_y": 40.0, **kw}


# --- opponent-relative features -------------------------------------------

def test_possession_share_sums_to_one_and_reflects_pass_counts():
    rows = [_pass(1, "A") for _ in range(3)] + [_pass(2, "B") for _ in range(1)]
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "possession_share"] == pytest.approx(0.75)
    assert out.loc["B", "possession_share"] == pytest.approx(0.25)
    assert out["possession_share"].sum() == pytest.approx(1.0)


def test_field_tilt_uses_final_third_passes_only():
    rows = [
        _pass(1, "A", location_x=90.0, pass_end_x=95.0),   # A in final third
        _pass(1, "A", location_x=90.0, pass_end_x=95.0),
        _pass(1, "A", location_x=30.0, pass_end_x=35.0),   # A deep: ignored by tilt
        _pass(2, "B", location_x=90.0, pass_end_x=95.0),   # B in final third
    ]
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "field_tilt"] == pytest.approx(2 / 3)
    assert out.loc["B", "field_tilt"] == pytest.approx(1 / 3)


def test_ppda_pairs_our_high_defensive_actions_with_their_deep_passes():
    # B plays 4 passes inside its own 60% (x <= 72); A makes 2 defensive
    # actions in its attacking 60% (x >= 48). PPDA(A) = 4 / 2 = 2.
    rows = (
        [_pass(2, "B", location_x=30.0) for _ in range(4)]
        + [{"type": "Pressure", "team_id": 1, "team": "A", "location_x": 70.0},
           {"type": "Interception", "team_id": 1, "team": "A", "location_x": 60.0}]
        + [_pass(1, "A", location_x=50.0)]
    )
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "ppda"] == pytest.approx(2.0)


def test_deep_defensive_actions_do_not_count_towards_ppda():
    rows = (
        [_pass(2, "B", location_x=30.0) for _ in range(4)]
        + [{"type": "Pressure", "team_id": 1, "team": "A", "location_x": 20.0}]
        + [_pass(1, "A", location_x=50.0)]
    )
    out = compute_team_match(_events(rows)).set_index("team")
    assert pd.isna(out.loc["A", "ppda"])       # no qualifying actions -> undefined


# --- set pieces ------------------------------------------------------------

def _shot(team_id, team, **kw):
    return {"type": "Shot", "team_id": team_id, "team": team, "shot_type": "Open Play",
            "location_x": 105.0, "location_y": 40.0, "shot_xg": 0.1, **kw}


def test_set_piece_shot_counts_only_the_first_phase():
    rows = [
        # corner possession, shot 4s in -> a set-piece shot
        _shot(1, "A", play_pattern="From Corner", possession=1, minute=0, second=4),
        # same corner possession, shot 40s in -> open play by then
        _shot(1, "A", play_pattern="From Corner", possession=1, minute=0, second=40),
        _pass(1, "A", possession=1, minute=0, second=0),
        _pass(2, "B", possession=2),
    ]
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "set_piece_shot_share"] == pytest.approx(0.5)


def test_throw_in_possessions_are_not_set_pieces():
    rows = [
        _shot(1, "A", play_pattern="From Throw In", possession=1, minute=0, second=2),
        _pass(1, "A", possession=1, minute=0, second=0),
        _pass(2, "B", possession=2),
    ]
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "set_piece_shot_share"] == pytest.approx(0.0)


def test_penalties_are_excluded_from_shot_metrics():
    rows = [
        _shot(1, "A", shot_type="Penalty", shot_xg=0.78),
        _shot(1, "A", shot_type="Open Play", shot_xg=0.05, location_x=95.0),
        _pass(2, "B"),
    ]
    out = compute_team_match(_events(rows)).set_index("team")
    assert out.loc["A", "npxg_per_shot"] == pytest.approx(0.05)
    assert out.loc["A", "box_shot_share"] == pytest.approx(0.0)  # the 95.0 shot is outside


# --- structure and aggregation --------------------------------------------

def test_one_row_per_team_per_match():
    rows = [_pass(1, "A"), _pass(2, "B"),
            _pass(1, "A", match_id=2), _pass(2, "B", match_id=2)]
    out = compute_team_match(_events(rows))
    assert len(out) == 4
    assert not out.duplicated(["match_id", "team_id"]).any()


def test_season_aggregation_reports_mean_sd_and_matches():
    rows = []
    for match_id, a_passes in ((1, 3), (2, 1)):
        rows += [_pass(1, "A", match_id=match_id) for _ in range(a_passes)]
        rows += [_pass(2, "B", match_id=match_id)]
    team_match = compute_team_match(_events(rows))
    season = build_team_season(team_match).set_index("team")

    assert season.loc["A", "matches"] == 2
    # possession share was 0.75 then 0.5 -> mean 0.625
    assert season.loc["A", "possession_share_mean"] == pytest.approx(0.625)
    assert season.loc["A", "possession_share_sd"] > 0
    assert 0.0 <= season.loc["A", "pct_possession_share"] <= 1.0
