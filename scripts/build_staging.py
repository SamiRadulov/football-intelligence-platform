"""Phase 2: build the canonical staging layer from the raw JSON.

Reads data/raw/, produces four canonical tables as Parquet in data/staging/,
loads them into a DuckDB database (data/curated.duckdb), and runs the
data-quality checks. Exits non-zero if any hard check fails.

Usage (from the repo root):
    .venv/Scripts/python scripts/build_staging.py
    .venv/Scripts/python scripts/build_staging.py --limit 10   # quick run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RAW_DIR, load_config  # noqa: E402
from src.transformations.events import flatten_events  # noqa: E402
from src.transformations.matches import build_dim_matches  # noqa: E402
from src.transformations.minutes import (  # noqa: E402
    build_fact_lineups,
    build_lineup_rows,
    match_end_minute,
)
from src.transformations.players import build_dim_players  # noqa: E402
from src.validation.checks import run_all_checks, summarize  # noqa: E402

STAGING_DIR = DATA_DIR / "staging"
EVENTS_DIR = STAGING_DIR / "fact_events"
DB_PATH = DATA_DIR / "curated.duckdb"

# Slim projection of events kept in memory for the quality checks (the full
# 1.3M-row wide table is streamed to Parquet per match, never all held at once).
CHECK_COLUMNS = ["event_id", "match_id", "player_id", "type", "minute", "second"]


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process at most N matches")
    args = parser.parse_args()

    dataset = load_config()["dataset"]
    matches_file = RAW_DIR / "matches" / str(dataset["competition_id"]) / f"{dataset['season_id']}.json"
    matches_raw = _load_json(matches_file)

    dim_matches = build_dim_matches(matches_raw)
    match_ids = dim_matches["match_id"].tolist()
    if args.limit is not None:
        match_ids = match_ids[: args.limit]
    print(f"Building staging for {len(match_ids)} matches...")

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in EVENTS_DIR.glob("*.parquet"):
        stale.unlink()

    all_lineups: list[list[dict]] = []
    lineup_rows: list[dict] = []
    events_slim: list[pd.DataFrame] = []

    for i, match_id in enumerate(match_ids, start=1):
        lineups_raw = _load_json(RAW_DIR / "lineups" / f"{match_id}.json")
        events_raw = _load_json(RAW_DIR / "events" / f"{match_id}.json")

        all_lineups.append(lineups_raw)
        end = match_end_minute(events_raw)
        lineup_rows.extend(build_lineup_rows(match_id, lineups_raw, end))

        events_df = flatten_events(match_id, events_raw)
        events_df.to_parquet(EVENTS_DIR / f"{match_id}.parquet", index=False)
        events_slim.append(events_df[CHECK_COLUMNS])

        if i % 50 == 0 or i == len(match_ids):
            print(f"  {i}/{len(match_ids)} matches")

    dim_players = build_dim_players(all_lineups)
    fact_lineups = build_fact_lineups(lineup_rows)

    # Restrict dim_matches to the processed matches (matters when --limit is used).
    dim_matches = dim_matches[dim_matches["match_id"].isin(match_ids)].reset_index(drop=True)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    dim_matches.to_parquet(STAGING_DIR / "dim_matches.parquet", index=False)
    dim_players.to_parquet(STAGING_DIR / "dim_players.parquet", index=False)
    fact_lineups.to_parquet(STAGING_DIR / "fact_lineups.parquet", index=False)
    print(
        f"\nTables: dim_matches={len(dim_matches)}, dim_players={len(dim_players)}, "
        f"fact_lineups={len(fact_lineups)}, fact_events={len(match_ids)} files"
    )

    # Load everything into DuckDB (the curated layer the app and marts query).
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE dim_matches AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "dim_matches.parquet")])
    con.execute("CREATE OR REPLACE TABLE dim_players AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "dim_players.parquet")])
    con.execute("CREATE OR REPLACE TABLE fact_lineups AS SELECT * FROM read_parquet(?)",
                [str(STAGING_DIR / "fact_lineups.parquet")])
    con.execute(
        "CREATE OR REPLACE TABLE fact_events AS "
        "SELECT * FROM read_parquet(?, union_by_name=true)",
        [str(EVENTS_DIR / "*.parquet")],
    )
    n_events = con.execute("SELECT count(*) FROM fact_events").fetchone()[0]
    con.close()
    print(f"DuckDB loaded at {DB_PATH} (fact_events rows: {n_events:,})")

    # Data-quality gate.
    fact_events_slim = pd.concat(events_slim, ignore_index=True)
    results = run_all_checks(dim_matches, dim_players, fact_lineups, fact_events_slim)
    hard_ok, report = summarize(results)
    print("\nData-quality checks:")
    print(report)

    if not hard_ok:
        print("\nBUILD FAILED: one or more hard checks did not pass.")
        sys.exit(1)
    print("\nStaging build complete. All hard checks passed.")


if __name__ == "__main__":
    main()
