"""Team playing-style classification: PCA, clustering and nearest-team similarity.

Pipeline:

    season style means
      -> z-score across the league
      -> PCA down to the components explaining `pca_variance_target`
      -> KMeans over a small range of k, scored on silhouette AND stability
      -> clusters named from their centroids, only after inspection

Two constraints shape every choice here:

1. **There are only 20 teams.** That is a very small sample for clustering, so
   the feature space is reduced before fitting, k is kept small, and stability
   across random restarts is treated as more informative than silhouette alone.
2. **Hard cluster boundaries are lossy.** A team sitting between two styles gets
   one label but is not well described by it, so nearest-team similarity is also
   computed in the original standardized feature space and shown alongside.

Clustering uses season **means**. The match-to-match standard deviations are
reported next to a team's profile as a variability read, but are not clustered
on: 28 means plus 28 standard deviations over 20 teams would be far more
dimensions than the sample can support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..features.team_match import STYLE_FEATURES


@dataclass
class StyleSpace:
    """The standardized feature space and its PCA projection."""

    teams: pd.Index
    features: list[str]
    z: pd.DataFrame              # index = team, columns = features (z-scores)
    components: pd.DataFrame     # index = team, columns = PC1..PCn
    explained_variance: np.ndarray
    loadings: pd.DataFrame       # index = features, columns = PC1..PCn
    pca: PCA


def build_style_space(
    team_season: pd.DataFrame, variance_target: float, random_state: int = 42
) -> StyleSpace:
    """Standardize the season style means and project them onto PCA axes."""
    mean_columns = [f"{feature}_mean" for feature in STYLE_FEATURES]
    raw = team_season.set_index("team")[mean_columns]
    raw.columns = STYLE_FEATURES

    z = (raw - raw.mean()) / raw.std(ddof=0)
    z = z.fillna(0.0)

    # Fit the full decomposition first, then keep the components needed to reach
    # the variance target.
    full = PCA(random_state=random_state).fit(z.to_numpy())
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=random_state).fit(z.to_numpy())
    names = [f"PC{i + 1}" for i in range(n_components)]
    components = pd.DataFrame(pca.transform(z.to_numpy()), index=z.index, columns=names)
    loadings = pd.DataFrame(pca.components_.T, index=STYLE_FEATURES, columns=names)

    return StyleSpace(
        teams=z.index, features=STYLE_FEATURES, z=z, components=components,
        explained_variance=pca.explained_variance_ratio_, loadings=loadings, pca=pca,
    )


def evaluate_k(
    components: pd.DataFrame, k_candidates: list[int], n_seeds: int, random_state: int = 42
) -> pd.DataFrame:
    """Score each candidate k on silhouette and stability.

    Stability is the mean adjusted Rand index between the labellings produced by
    different random restarts: 1.0 means every restart finds the same partition.
    With 20 teams this is the more trustworthy of the two measures, because a
    silhouette computed on so few points moves a lot for small changes.
    """
    x = components.to_numpy()
    rows = []
    for k in k_candidates:
        labellings = [
            KMeans(n_clusters=k, n_init=10, random_state=random_state + seed).fit_predict(x)
            for seed in range(n_seeds)
        ]
        pairwise = [
            adjusted_rand_score(labellings[i], labellings[j])
            for i in range(len(labellings))
            for j in range(i + 1, len(labellings))
        ]
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labellings[0])),
                "stability_ari": float(np.mean(pairwise)),
                "smallest_cluster": int(np.bincount(labellings[0]).min()),
            }
        )
    return pd.DataFrame(rows)


def feature_subset_stability(
    space: StyleSpace, k: int, variance_target: float,
    drop_fraction: float = 0.2, n_repeats: int = 20, random_state: int = 42,
) -> float:
    """Mean ARI when a random fraction of features is dropped and the model refit.

    A style solution that depends on a handful of particular features is not a
    real structure in the data.
    """
    baseline = fit_clusters(space.components, k, random_state)
    rng = np.random.default_rng(random_state)
    n_keep = int(round(len(space.features) * (1 - drop_fraction)))

    scores = []
    for _ in range(n_repeats):
        kept = rng.choice(space.features, size=n_keep, replace=False)
        subset = space.z[list(kept)]
        full = PCA(random_state=random_state).fit(subset.to_numpy())
        n_components = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_),
                                           variance_target) + 1)
        projected = PCA(n_components=n_components, random_state=random_state).fit_transform(
            subset.to_numpy())
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(projected)
        scores.append(adjusted_rand_score(baseline, labels))
    return float(np.mean(scores))


def fit_clusters(components: pd.DataFrame, k: int, random_state: int = 42) -> np.ndarray:
    """Fit KMeans on the PCA components and return cluster labels."""
    return KMeans(n_clusters=k, n_init=50, random_state=random_state).fit_predict(
        components.to_numpy())


def cluster_profiles(space: StyleSpace, labels: np.ndarray) -> pd.DataFrame:
    """Mean z-score per feature for each cluster — the evidence for its label."""
    profile = space.z.copy()
    profile["cluster"] = labels
    return profile.groupby("cluster").mean().T


def describe_cluster(profiles: pd.DataFrame, cluster: int, n: int = 6) -> pd.DataFrame:
    """The features that most define one cluster, strongest deviation first."""
    column = profiles[cluster].sort_values(ascending=False)
    return pd.concat([column.head(n), column.tail(n)]).to_frame("mean_z")


def nearest_teams(space: StyleSpace, team: str, n: int = 5) -> pd.DataFrame:
    """Closest teams by Euclidean distance in the standardized feature space.

    Euclidean rather than cosine: for team style the *magnitude* of a trait
    matters, not only its direction. A side that presses slightly higher than
    average should not read as identical to one that presses far higher.

    Distances are computed on the original standardized features, not the PCA
    components, so nothing is lost to the discarded variance — and a team can be
    close to a side in another cluster, which hard labels would hide.
    """
    if team not in space.z.index:
        raise KeyError(f"unknown team {team!r}")
    distances = np.linalg.norm(space.z.to_numpy() - space.z.loc[team].to_numpy(), axis=1)
    result = pd.DataFrame({"team": space.z.index, "distance": distances})
    return result[result["team"] != team].nsmallest(n, "distance").reset_index(drop=True)


def outlier_distances(space: StyleSpace, labels: np.ndarray) -> pd.DataFrame:
    """Each team's distance to its own cluster centroid, furthest first.

    Surfaces the sides that a single label describes badly, rather than hiding
    them inside a cluster.
    """
    z = space.z.to_numpy()
    centroids = {c: z[labels == c].mean(axis=0) for c in np.unique(labels)}
    distances = [np.linalg.norm(row - centroids[label]) for row, label in zip(z, labels)]
    return (
        pd.DataFrame({"team": space.z.index, "cluster": labels, "distance": distances})
        .sort_values("distance", ascending=False)
        .reset_index(drop=True)
    )
