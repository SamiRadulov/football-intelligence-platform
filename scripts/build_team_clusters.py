"""Phase 6: team playing-style classification (PCA + clustering).

Two-stage by design, because k and the cluster names must be chosen *after*
looking at the data, never before:

    # 1. Explore: PCA axes, k evaluation table, candidate centroids
    .venv\\Scripts\\python scripts\\build_team_clusters.py --explore

    # 2. Set `clustering.k` and `clustering.labels` in artifacts/feature_config.yml,
    #    then fit and persist the model
    .venv\\Scripts\\python scripts\\build_team_clusters.py

Writes model_team_style (one row per team: PCA coordinates, cluster, label,
distance to centroid and nearest teams) to Parquet and DuckDB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.style_model import (  # noqa: E402
    build_style_space,
    cluster_profiles,
    describe_cluster,
    evaluate_k,
    feature_subset_stability,
    fit_clusters,
    nearest_teams,
    outlier_distances,
)
from src.config import DATA_DIR, load_config  # noqa: E402

DB_PATH = DATA_DIR / "curated.duckdb"
STAGING_DIR = DATA_DIR / "staging"
N_NEAREST = 3


def load_team_season() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM mart_team_season").df()
    con.close()
    return df


def explore(space, cfg) -> None:
    print(f"PCA: {space.components.shape[1]} components explain "
          f"{space.explained_variance.sum():.1%} of variance")
    for i, ratio in enumerate(space.explained_variance, start=1):
        print(f"   PC{i}: {ratio:.1%}")

    print("\nWhat the first two axes are made of (largest loadings):")
    for pc in ("PC1", "PC2"):
        if pc not in space.loadings.columns:
            continue
        ordered = space.loadings[pc].sort_values(ascending=False)
        top = ", ".join(f"{f} {v:+.2f}" for f, v in ordered.head(4).items())
        bottom = ", ".join(f"{f} {v:+.2f}" for f, v in ordered.tail(4).items())
        print(f"   {pc}  high: {top}")
        print(f"        low:  {bottom}")

    print("\nChoosing k (silhouette and stability across random restarts):")
    table = evaluate_k(space.components, cfg["k_candidates"], cfg["n_seeds"],
                       cfg["random_state"])
    print(table.to_string(index=False))
    print("\nPrefer the smallest k that is stable and interpretable — with only")
    print("20 teams a marginally better silhouette is not worth an extra cluster.")

    for k in cfg["k_candidates"]:
        labels = fit_clusters(space.components, k, cfg["random_state"])
        profiles = cluster_profiles(space, labels)
        print(f"\n--- k = {k} ---")
        for cluster in sorted(set(labels)):
            members = [t for t, lab in zip(space.teams, labels) if lab == cluster]
            print(f"  cluster {cluster} ({len(members)}): {', '.join(members)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore", action="store_true",
                        help="print PCA axes and k evaluation instead of fitting")
    args = parser.parse_args()

    config = load_config()
    cfg = config["clustering"]
    team_season = load_team_season()
    space = build_style_space(team_season, cfg["pca_variance_target"], cfg["random_state"])

    if args.explore:
        explore(space, cfg)
        return

    k = cfg.get("k")
    if not k:
        print("clustering.k is not set in artifacts/feature_config.yml.")
        print("Run with --explore first, then choose k from the evidence.")
        sys.exit(1)

    labels = fit_clusters(space.components, k, cfg["random_state"])
    profiles = cluster_profiles(space, labels)
    label_names = {int(key): value for key, value in (cfg.get("labels") or {}).items()}

    print(f"Fitted k={k} on {space.components.shape[1]} PCA components "
          f"({space.explained_variance.sum():.1%} of variance).\n")

    print("Cluster centroids — the evidence behind each label:")
    for cluster in sorted(set(labels)):
        members = [t for t, lab in zip(space.teams, labels) if lab == cluster]
        name = label_names.get(cluster, "<unnamed>")
        print(f"\n  Cluster {cluster}: {name}  ({len(members)} teams)")
        print(f"    {', '.join(members)}")
        described = describe_cluster(profiles, cluster, n=5)
        for feature, row in described.iterrows():
            direction = "HIGH" if row["mean_z"] > 0 else "LOW "
            print(f"      {direction} {feature:<38} {row['mean_z']:+.2f}")

    stability = feature_subset_stability(space, k, cfg["pca_variance_target"],
                                         random_state=cfg["random_state"])
    print(f"\nFeature-subset stability (drop 20% of features, 20 repeats): ARI {stability:.2f}")

    outliers = outlier_distances(space, labels)
    print("\nTeams furthest from their own centroid (least well described by their label):")
    print(outliers.head(4).to_string(index=False))

    # Assemble the model table.
    rows = []
    for team, label in zip(space.teams, labels):
        near = nearest_teams(space, team, N_NEAREST)
        record = {
            "team": team,
            "cluster": int(label),
            "cluster_label": label_names.get(int(label), f"cluster_{label}"),
            "centroid_distance": float(
                outliers.loc[outliers["team"] == team, "distance"].iloc[0]),
            "nearest_teams": ", ".join(near["team"]),
        }
        for pc in space.components.columns:
            record[pc.lower()] = float(space.components.loc[team, pc])
        rows.append(record)
    model = pd.DataFrame(rows).sort_values(["cluster", "team"]).reset_index(drop=True)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGING_DIR / "model_team_style.parquet"
    model.to_parquet(path, index=False)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE model_team_style AS SELECT * FROM read_parquet(?)",
                [str(path)])
    con.close()

    print(f"\nmodel_team_style written ({len(model)} teams).")
    if not label_names:
        print("No cluster labels configured yet — inspect the centroids above, then")
        print("set clustering.labels in artifacts/feature_config.yml and re-run.")


if __name__ == "__main__":
    main()
