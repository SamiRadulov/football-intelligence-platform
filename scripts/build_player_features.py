"""Phase 3: build the player feature marts from the curated layer.

Reads fact_events / fact_lineups / dim_players from data/curated.duckdb, builds
mart_player_match (raw counts) and mart_player_season (per-90, possession-
adjusted and percentile features), and writes both back to Parquet and DuckDB.

Usage (from the repo root):
    .venv/Scripts/python scripts/build_player_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, load_config  # noqa: E402
from src.features.player_match import compute_player_match  # noqa: E402
from src.features.player_season import (  # noqa: E402
    METRIC_COLUMNS,
    build_player_season,
    team_possession_share,
)
from src.features.roles import assign_player_roles  # noqa: E402


def validate_player_season(season, min_minutes: int) -> None:
    """Fail loudly if the season feature matrix violates basic guarantees."""
    assert season["role"].notna().all(), "some players have no role"
    assert (season["minutes"] >= min_minutes).all(), "sub-threshold player leaked in"
    assert not season["player_id"].duplicated().any(), "duplicate player rows"

    per90 = [c for c in METRIC_COLUMNS if c.endswith("_p90")]
    negative = season[per90].lt(0).any()
    assert not negative.any(), f"negative per-90 values: {list(negative[negative].index)}"

    for pct in [f"pct_{c}" for c in METRIC_COLUMNS]:
        col = season[pct].dropna()
        assert col.between(0, 1).all(), f"{pct} outside [0, 1]"

    for share in ["pass_completion_pct", "box_shot_share", "dribble_success_pct",
                  "aerial_win_pct", "touches_att_third_share", "pressured_actions_share",
                  "halfspace_touch_share", "wide_touch_share"]:
        col = season[share].dropna()
        assert col.between(0, 1).all(), f"{share} outside [0, 1]"
    print("Feature validation passed.")

DB_PATH = DATA_DIR / "curated.duckdb"
STAGING_DIR = DATA_DIR / "staging"

# Columns from fact_events needed to compute every player metric.
EVENT_COLUMNS = [
    "event_id", "match_id", "player_id", "team_id", "type",
    "location_x", "location_y", "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y",
    "pass_outcome", "pass_type", "pass_length", "pass_cross", "pass_through_ball",
    "pass_shot_assist", "shot_xg", "shot_outcome", "shot_type", "shot_key_pass_id",
    "under_pressure", "duel_type", "dribble_outcome", "ball_receipt_outcome", "aerial_won",
]


def main() -> None:
    config = load_config()
    con = duckdb.connect(str(DB_PATH))

    print("Loading curated tables...")
    events = con.execute(f"SELECT {', '.join(EVENT_COLUMNS)} FROM fact_events").df()
    fact_lineups = con.execute("SELECT * FROM fact_lineups").df()
    dim_players = con.execute("SELECT * FROM dim_players").df()
    dim_matches = con.execute("SELECT * FROM dim_matches").df()
    print(f"  events={len(events):,}, lineups={len(fact_lineups):,}")

    print("Assigning roles and computing per-match metrics...")
    roles = assign_player_roles(fact_lineups)
    player_match = compute_player_match(events)
    team_possession = team_possession_share(events)

    print("Aggregating to season features...")
    player_season = build_player_season(
        player_match=player_match,
        fact_lineups=fact_lineups,
        roles=roles,
        team_possession=team_possession,
        dim_players=dim_players,
        dim_matches=dim_matches,
        config=config,
    )

    validate_player_season(player_season, config["thresholds"]["min_minutes_season"])

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    player_match.to_parquet(STAGING_DIR / "mart_player_match.parquet", index=False)
    player_season.to_parquet(STAGING_DIR / "mart_player_season.parquet", index=False)
    con.execute("CREATE OR REPLACE TABLE mart_player_match AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "mart_player_match.parquet")])
    con.execute("CREATE OR REPLACE TABLE mart_player_season AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "mart_player_season.parquet")])

    n_by_role = player_season["role"].value_counts().to_dict()
    print(f"\nmart_player_match rows: {len(player_match):,}")
    print(f"mart_player_season rows: {len(player_season)} "
          f"(>= {config['thresholds']['min_minutes_season']} min)")
    print(f"  by role: {n_by_role}")
    con.close()
    print("\nPlayer feature marts built.")


if __name__ == "__main__":
    main()
