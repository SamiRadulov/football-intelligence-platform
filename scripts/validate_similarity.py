"""Phase 4 validation: is the similarity ranking stable and how much do weights matter?

Three checks, all reported as top-10 overlap (how many of a player's ten most
similar players survive a change):

1. **Match resampling** — rebuild the season features from a random 80% of matches
   and re-rank. A tool whose answers change completely when a few matches are
   dropped is fitting noise, not style.
2. **Minutes threshold** — raise the qualifying threshold and re-rank, to show the
   answers are not an artefact of one cut-off.
3. **Weight sensitivity** — compare role-specific weights against equal weights, to
   show how much the analytical choices actually drive the output.

Usage (from the repo root):
    .venv/Scripts/python scripts/validate_similarity.py
    .venv/Scripts/python scripts/validate_similarity.py --seeds 5 --top 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, load_config  # noqa: E402
from src.features.player_season import build_player_season  # noqa: E402
from src.features.roles import assign_player_roles  # noqa: E402
from src.similarity.engine import weighted_cosine_matrix  # noqa: E402
from src.similarity.feature_matrix import build_role_matrix  # noqa: E402

DB_PATH = DATA_DIR / "curated.duckdb"


def top_n_sets(
    player_season: pd.DataFrame, config: dict, top_n: int,
    equal_weights: bool = False,
) -> dict[int, list[int]]:
    """Map each player to their top-N most similar players, across all roles."""
    sim_cfg = config["similarity"]
    out: dict[int, list[int]] = {}
    for role, weights in config["role_features"].items():
        if equal_weights:
            weights = dict.fromkeys(weights, 1.0)
        if (player_season["role"] == role).sum() < 2:
            continue
        matrix = build_role_matrix(
            player_season, role, weights,
            sim_cfg["winsorize_lower"], sim_cfg["winsorize_upper"],
        )
        sim = weighted_cosine_matrix(matrix.z, matrix.weights)
        values = sim.to_numpy(copy=True)
        np.fill_diagonal(values, -np.inf)              # never rank a player against self
        sim = pd.DataFrame(values, index=sim.index, columns=sim.columns)
        eligible = matrix.coverage >= sim_cfg["min_feature_coverage"]
        sim = sim.loc[eligible[eligible].index, eligible[eligible].index]
        for player_id in sim.index:
            out[player_id] = sim.loc[player_id].nlargest(top_n).index.tolist()
    return out


def mean_overlap(
    base: dict[int, list[int]], other: dict[int, list[int]], top_n: int
) -> tuple[float, int]:
    """Average share of the baseline top-N that survives the change.

    Only baseline recommendations that *could* still be returned are counted.
    Resampling and higher minutes thresholds push some players below the
    qualifying threshold entirely; without this restriction those players would
    be scored as "lost recommendations" when they simply left the candidate
    pool, which measures pool shrinkage rather than ranking stability.
    """
    shared = set(base) & set(other)
    if not shared:
        return float("nan"), 0
    universe = set(other)
    overlaps = []
    for player in shared:
        still_eligible = [c for c in base[player] if c in universe]
        if not still_eligible:
            continue
        overlaps.append(len(set(still_eligible) & set(other[player])) / len(still_eligible))
    if not overlaps:
        return float("nan"), 0
    return float(np.mean(overlaps)), len(overlaps)


def rebuild_season(
    player_match: pd.DataFrame, fact_lineups: pd.DataFrame, dim_players: pd.DataFrame,
    team_possession: pd.DataFrame, config: dict, match_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Rebuild mart_player_season, optionally from a subset of matches."""
    if match_ids is not None:
        keep = set(match_ids)
        player_match = player_match[player_match["match_id"].isin(keep)]
        fact_lineups = fact_lineups[fact_lineups["match_id"].isin(keep)]
    roles = assign_player_roles(fact_lineups)
    return build_player_season(
        player_match=player_match, fact_lineups=fact_lineups, roles=roles,
        team_possession=team_possession, dim_players=dim_players, config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="resampling repeats")
    parser.add_argument("--top", type=int, default=10, help="top-N to compare")
    parser.add_argument("--keep-share", type=float, default=0.8,
                        help="share of matches kept when resampling")
    args = parser.parse_args()

    config = load_config()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    player_match = con.execute("SELECT * FROM mart_player_match").df()
    fact_lineups = con.execute("SELECT * FROM fact_lineups").df()
    dim_players = con.execute("SELECT * FROM dim_players").df()
    baseline = con.execute("SELECT * FROM mart_player_season").df()
    con.close()

    # Team possession share is a season-level team property; hold it fixed so the
    # resampling measures feature stability, not possession-estimate noise.
    team_possession = baseline[["team_id", "possession_share"]].drop_duplicates()

    base_top = top_n_sets(baseline, config, args.top)
    print(f"Baseline: {len(baseline)} players, comparing top-{args.top} lists.\n")

    # 1. Match resampling ---------------------------------------------------
    all_matches = sorted(player_match["match_id"].unique())
    n_keep = int(len(all_matches) * args.keep_share)
    print(f"1. Match resampling (keep {args.keep_share:.0%} = {n_keep}/{len(all_matches)} "
          f"matches, {args.seeds} seeds)")
    scores = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        subset = rng.choice(all_matches, size=n_keep, replace=False).tolist()
        season = rebuild_season(player_match, fact_lineups, dim_players,
                                team_possession, config, match_ids=subset)
        overlap, n = mean_overlap(base_top, top_n_sets(season, config, args.top), args.top)
        scores.append(overlap)
        print(f"   seed {seed}: {overlap:.1%} of still-eligible recommendations retained "
              f"({n} players comparable, {len(season)} qualified)")
    print(f"   mean: {np.mean(scores):.1%}  (sd {np.std(scores):.1%})\n")

    # 2. Minutes threshold sensitivity --------------------------------------
    print("2. Minutes threshold sensitivity")
    for threshold in (900, 1200):
        raised = baseline[baseline["minutes"] >= threshold].reset_index(drop=True)
        overlap, n = mean_overlap(base_top, top_n_sets(raised, config, args.top), args.top)
        print(f"   >= {threshold} min ({len(raised)} players): {overlap:.1%} retained "
              f"({n} comparable)")
    print()

    # 3. Weight sensitivity -------------------------------------------------
    equal_top = top_n_sets(baseline, config, args.top, equal_weights=True)
    overlap, n = mean_overlap(base_top, equal_top, args.top)
    print("3. Weight sensitivity (role-specific vs equal weights)")
    print(f"   {overlap:.1%} of the top-{args.top} retained ({n} players)")
    print("   Lower overlap means the role weights are doing real work; near 100%")
    print("   would mean the weights barely matter.\n")

    print("Interpretation: high resampling stability = the rankings reflect season-long")
    print("style, not a handful of matches. Similarity remains descriptive, not a")
    print("judgement of quality or transfer suitability.")


if __name__ == "__main__":
    main()
