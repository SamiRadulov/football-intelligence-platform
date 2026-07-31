"""Phase 5: build the team style feature marts from the curated layer.

Reads fact_events from data/curated.duckdb, builds mart_team_match (style
features per team per match) and mart_team_season (season mean, match-to-match
standard deviation and league percentile), validates them, and writes both back
to Parquet and DuckDB.

Usage (from the repo root):
    .venv\\Scripts\\python scripts\\build_team_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402
from src.features.team_match import STYLE_FEATURES, compute_team_match  # noqa: E402
from src.features.team_season import build_team_season  # noqa: E402

DB_PATH = DATA_DIR / "curated.duckdb"
STAGING_DIR = DATA_DIR / "staging"

EVENT_COLUMNS = [
    "match_id", "team_id", "team", "type", "minute", "second",
    "possession", "possession_team", "play_pattern",
    "location_x", "location_y", "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y",
    "pass_outcome", "pass_type", "pass_length", "pass_cross", "pass_switch",
    "shot_xg", "shot_type", "under_pressure", "duel_type",
]

# Shares must lie in [0, 1]; these are the style features expressed as shares.
SHARE_FEATURES = [
    "possession_share", "pass_completion", "backward_pass_share", "field_tilt",
    "att_third_touch_share", "wide_touch_share", "halfspace_touch_share",
    "counter_shot_share", "box_shot_share", "set_piece_shot_share",
    "high_regain_share", "pressured_pass_completion", "own_half_loss_share",
]


def validate(team_match, team_season, expected_matches: int) -> None:
    """Fail loudly if the team marts violate basic guarantees."""
    assert len(team_match) == expected_matches * 2, (
        f"expected two rows per match ({expected_matches * 2}), got {len(team_match)}"
    )
    assert not team_match.duplicated(["match_id", "team_id"]).any(), "duplicate team-match rows"

    for share in SHARE_FEATURES:
        column = team_match[share].dropna()
        assert column.between(0, 1).all(), f"{share} outside [0, 1]"

    # Possession share and field tilt are two-team splits and must sum to 1.
    for feature in ("possession_share", "field_tilt"):
        totals = team_match.groupby("match_id")[feature].sum().dropna()
        assert ((totals - 1.0).abs() < 1e-9).all(), f"{feature} does not sum to 1 per match"

    assert (team_season["matches"] == 38).all(), "every team should have 38 matches"
    assert (team_match["ppda"].dropna() > 0).all(), "PPDA must be positive"
    print("Team feature validation passed.")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    print("Loading events...")
    events = con.execute(f"SELECT {', '.join(EVENT_COLUMNS)} FROM fact_events").df()
    n_matches = con.execute("SELECT count(*) FROM dim_matches").fetchone()[0]
    print(f"  events={len(events):,}, matches={n_matches}")

    print("Computing team-match style features...")
    team_match = compute_team_match(events)
    print("Aggregating to team-season profiles...")
    team_season = build_team_season(team_match)

    validate(team_match, team_season, n_matches)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    team_match.to_parquet(STAGING_DIR / "mart_team_match.parquet", index=False)
    team_season.to_parquet(STAGING_DIR / "mart_team_season.parquet", index=False)
    con.execute("CREATE OR REPLACE TABLE mart_team_match AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "mart_team_match.parquet")])
    con.execute("CREATE OR REPLACE TABLE mart_team_season AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "mart_team_season.parquet")])
    con.close()

    print(f"\nmart_team_match rows: {len(team_match)} "
          f"({len(STYLE_FEATURES)} style features)")
    print(f"mart_team_season rows: {len(team_season)} teams")
    print("\nTeam style marts built.")


if __name__ == "__main__":
    main()
