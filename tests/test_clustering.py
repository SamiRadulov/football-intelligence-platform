"""Unit tests for team style clustering (synthetic teams, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.clustering.style_model import (
    build_style_space,
    cluster_profiles,
    evaluate_k,
    fit_clusters,
    nearest_teams,
    outlier_distances,
)
from src.features.team_match import STYLE_FEATURES


def _team_season(n_per_group: int = 4, seed: int = 0) -> pd.DataFrame:
    """Two clearly separated style groups, so structure is known in advance."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, offset in enumerate((-2.0, 2.0)):
        for i in range(n_per_group):
            row = {"team_id": group * 100 + i, "team": f"G{group}T{i}", "matches": 38}
            for feature in STYLE_FEATURES:
                row[f"{feature}_mean"] = offset + rng.normal(0, 0.15)
            rows.append(row)
    return pd.DataFrame(rows)


def test_style_space_is_standardized_and_reduced():
    space = build_style_space(_team_season(), variance_target=0.8)
    assert list(space.z.columns) == STYLE_FEATURES
    assert abs(space.z.to_numpy().mean()) < 1e-9          # z-scores centred
    assert space.components.shape[0] == len(space.z)
    assert space.components.shape[1] <= len(STYLE_FEATURES)
    assert space.explained_variance.sum() >= 0.8


def test_pca_keeps_fewer_components_for_a_lower_target():
    data = _team_season(n_per_group=6)
    wide = build_style_space(data, variance_target=0.95)
    narrow = build_style_space(data, variance_target=0.5)
    assert narrow.components.shape[1] <= wide.components.shape[1]


def test_clustering_recovers_known_groups():
    space = build_style_space(_team_season(), variance_target=0.8)
    labels = fit_clusters(space.components, k=2)
    first_group = labels[:4]
    second_group = labels[4:]
    assert len(set(first_group)) == 1        # each planted group stays together
    assert len(set(second_group)) == 1
    assert first_group[0] != second_group[0]


def test_evaluate_k_reports_silhouette_and_stability():
    space = build_style_space(_team_season(n_per_group=5), variance_target=0.8)
    table = evaluate_k(space.components, [2, 3], n_seeds=5)
    assert list(table["k"]) == [2, 3]
    assert table["stability_ari"].between(-1, 1).all()
    assert table["silhouette"].between(-1, 1).all()
    # Two well-separated groups: k=2 should be the more stable solution.
    assert table.loc[table["k"] == 2, "stability_ari"].iloc[0] == pytest.approx(1.0)


def test_cluster_profiles_are_per_feature_means():
    space = build_style_space(_team_season(), variance_target=0.8)
    labels = fit_clusters(space.components, k=2)
    profiles = cluster_profiles(space, labels)
    assert set(profiles.index) == set(STYLE_FEATURES)
    assert sorted(profiles.columns) == sorted(set(labels))
    # The two groups sit on opposite sides of the mean on every feature.
    assert (profiles[0] * profiles[1] < 0).all()


def test_nearest_teams_excludes_self_and_sorts_by_distance():
    space = build_style_space(_team_season(), variance_target=0.8)
    near = nearest_teams(space, "G0T0", n=3)
    assert "G0T0" not in near["team"].tolist()
    assert near["distance"].is_monotonic_increasing
    # Closest neighbours should come from the same planted group.
    assert near.iloc[0]["team"].startswith("G0")


def test_nearest_teams_rejects_unknown_team():
    space = build_style_space(_team_season(), variance_target=0.8)
    with pytest.raises(KeyError, match="Nope"):
        nearest_teams(space, "Nope")


def test_outlier_distances_are_sorted_furthest_first():
    space = build_style_space(_team_season(), variance_target=0.8)
    labels = fit_clusters(space.components, k=2)
    outliers = outlier_distances(space, labels)
    assert len(outliers) == len(space.z)
    assert outliers["distance"].is_monotonic_decreasing
