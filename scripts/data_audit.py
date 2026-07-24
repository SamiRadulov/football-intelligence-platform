"""Phase 0 data audit: list StatsBomb Open Data competitions and count matches.

Answers one question: which competition-season has enough complete matches
for meaningful player and team comparisons?

Usage (from the repo root):
    .venv/Scripts/python scripts/data_audit.py

Output:
    - docs/data_audit.csv  (full audit table, one row per competition-season)
    - a printed summary of the best candidates
"""

from pathlib import Path

import pandas as pd
from statsbombpy import sb

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data_audit.csv"

# A single league season needs roughly 300+ matches (e.g. 20 teams x 38 rounds / 2).
# Tournaments (World Cup, Euro) have ~50-64 matches: fewer minutes per player,
# which weakens season-level per-90 comparisons.
MIN_MATCHES_FOR_LEAGUE = 200


def audit() -> pd.DataFrame:
    """Return one row per competition-season with its match count."""
    competitions = sb.competitions()
    rows = []
    for comp in competitions.itertuples(index=False):
        try:
            matches = sb.matches(
                competition_id=comp.competition_id, season_id=comp.season_id
            )
            n_matches = len(matches)
        except Exception as exc:  # a missing matches file should not kill the audit
            print(f"  ! {comp.competition_name} {comp.season_name}: {exc}")
            n_matches = 0
        rows.append(
            {
                "competition_id": comp.competition_id,
                "season_id": comp.season_id,
                "competition": comp.competition_name,
                "season": comp.season_name,
                "country": comp.country_name,
                "gender": comp.competition_gender,
                "matches": n_matches,
            }
        )
        print(f"  {comp.competition_name:<30} {comp.season_name:<10} {n_matches:>4} matches")
    return pd.DataFrame(rows).sort_values("matches", ascending=False)


def main() -> None:
    print("Fetching competition list from StatsBomb Open Data...")
    table = audit()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFull audit written to {OUTPUT_PATH}")

    league_size = table[table["matches"] >= MIN_MATCHES_FOR_LEAGUE]
    print(f"\nCandidates with >= {MIN_MATCHES_FOR_LEAGUE} matches (full league seasons):")
    print(league_size.to_string(index=False))


if __name__ == "__main__":
    main()
