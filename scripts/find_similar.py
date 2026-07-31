"""Phase 4: query the player similarity engine from the command line.

Usage (from the repo root):
    .venv/Scripts/python scripts/find_similar.py "Vardy"
    .venv/Scripts/python scripts/find_similar.py "Kante" --top 5 --min-minutes 1500
    .venv/Scripts/python scripts/find_similar.py "Ozil" --explain

Name matching is a case-insensitive substring search over player names and
nicknames; if several players match, they are listed so you can be more specific.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, load_config  # noqa: E402
from src.similarity.engine import explain_pair, find_similar  # noqa: E402
from src.similarity.feature_matrix import build_role_matrix  # noqa: E402

DB_PATH = DATA_DIR / "curated.duckdb"


def resolve_player(player_season: pd.DataFrame, query: str) -> pd.Series | None:
    """Find the one player matching `query`, or print the ambiguity and return None."""
    q = query.casefold()
    hit = player_season[
        player_season["player_name"].str.casefold().str.contains(q, regex=False)
        | player_season["nickname"].fillna("").str.casefold().str.contains(q, regex=False)
    ]
    if hit.empty:
        print(f"No player matching {query!r} with enough minutes.")
        return None
    if len(hit) > 1:
        print(f"{len(hit)} players match {query!r} — be more specific:")
        for _, row in hit.iterrows():
            print(f"  {row['player_name']}  ({row['role']}, {row['minutes']:.0f} min)")
        return None
    return hit.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("player", help="player name (substring, case-insensitive)")
    parser.add_argument("--top", type=int, default=10, help="how many candidates")
    parser.add_argument("--min-minutes", type=float, default=0.0)
    parser.add_argument("--explain", action="store_true",
                        help="show the per-feature breakdown for the top match")
    args = parser.parse_args()

    config = load_config()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    player_season = con.execute("SELECT * FROM mart_player_season").df()
    con.close()

    ref = resolve_player(player_season, args.player)
    if ref is None:
        sys.exit(1)

    role = ref["role"]
    matrix = build_role_matrix(
        player_season, role, config["role_features"][role],
        config["similarity"]["winsorize_lower"], config["similarity"]["winsorize_upper"],
    )
    results = find_similar(
        matrix, player_season, int(ref["player_id"]),
        top_n=args.top, min_minutes=args.min_minutes,
        min_coverage=config["similarity"]["min_feature_coverage"],
    )

    print(f"\nReference: {ref['player_name']}  "
          f"({role}, {ref['primary_position']}, {ref['minutes']:.0f} min)")
    print(f"Compared against {len(matrix.z) - 1} other {role} players "
          f"on {len(matrix.features)} role-specific features.\n")

    display = results[["player_name", "minutes", "similarity", "coverage", "low_minutes"]]
    print(display.to_string(index=False))

    print("\nWhy the top matches:")
    for _, row in results.head(3).iterrows():
        print(f"\n  {row['player_name']}  (similarity {row['similarity']:.3f})")
        print(f"    most alike:  {', '.join(row['most_similar_on'])}")
        print(f"    differs on:  {', '.join(row['biggest_differences'])}")

    if args.explain and not results.empty:
        top_id = int(results.iloc[0]["player_id"])
        breakdown = explain_pair(matrix, int(ref["player_id"]), top_id)
        print(f"\nFull breakdown vs {results.iloc[0]['player_name']} "
              "(sorted by weighted gap, closest first):")
        show = breakdown.copy()
        for col in ("gap", "reference_value", "candidate_value"):
            show[col] = show[col].round(3)
        print(show[["feature", "gap", "reference_value", "candidate_value"]].to_string(index=False))

    print("\nSimilarity is descriptive: it means these players do similar things, "
          "not that they are equally good or interchangeable.")


if __name__ == "__main__":
    main()
