"""Build the standardized feature matrix used for similarity, one role at a time.

Pipeline for a role:

    raw season metrics
      -> winsorize at the 1st/99th percentile (within role)
      -> z-score (within role)
      -> impute missing values to 0 (the role mean)
      -> track per-player coverage

Standardizing *within role* is the point: a centre-back's 40 passes per 90 and a
winger's 25 mean completely different things, so each role gets its own scale.
Winsorizing first stops one freak season from stretching the scale for everyone.

Missing values are imputed to the role mean (z = 0) rather than dropped, so a
player with one unavailable metric can still be compared — but `coverage` records
how much was really there, and the engine refuses candidates below a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RoleMatrix:
    """Standardized features for every player in one role group."""

    role: str
    features: list[str]           # feature names, in matrix column order
    weights: np.ndarray           # per-feature similarity weight
    z: pd.DataFrame               # index = player_id, columns = features (z-scores)
    coverage: pd.Series           # index = player_id, share of features present
    raw: pd.DataFrame             # index = player_id, the un-standardized values


def winsorize(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Clip a series to its own [lower_q, upper_q] quantiles, ignoring nulls."""
    valid = series.dropna()
    if valid.empty:
        return series
    low, high = valid.quantile(lower_q), valid.quantile(upper_q)
    return series.clip(lower=low, upper=high)


def zscore(series: pd.Series) -> pd.Series:
    """Standardize to mean 0 / sd 1. A constant feature returns all zeros."""
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / sd


def build_role_matrix(
    player_season: pd.DataFrame,
    role: str,
    role_weights: dict[str, float],
    winsorize_lower: float,
    winsorize_upper: float,
) -> RoleMatrix:
    """Build the standardized matrix for one role group."""
    players = player_season[player_season["role"] == role].set_index("player_id")

    features = [f for f in role_weights if f in players.columns]
    missing = set(role_weights) - set(features)
    if missing:
        raise KeyError(f"role {role}: features missing from the mart: {sorted(missing)}")

    raw = players[features]
    coverage = raw.notna().sum(axis=1) / len(features)

    z = pd.DataFrame(index=raw.index, columns=features, dtype="float64")
    for feature in features:
        clipped = winsorize(raw[feature], winsorize_lower, winsorize_upper)
        z[feature] = zscore(clipped)
    z = z.fillna(0.0)  # impute to the role mean; `coverage` records the gap

    weights = np.array([role_weights[f] for f in features], dtype="float64")
    return RoleMatrix(role=role, features=features, weights=weights,
                      z=z, coverage=coverage, raw=raw)


def build_all_role_matrices(
    player_season: pd.DataFrame, config: dict
) -> dict[str, RoleMatrix]:
    """Build one RoleMatrix per role defined in role_features."""
    sim = config["similarity"]
    return {
        role: build_role_matrix(
            player_season, role, weights,
            sim["winsorize_lower"], sim["winsorize_upper"],
        )
        for role, weights in config["role_features"].items()
    }
