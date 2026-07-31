"""Weighted cosine similarity over role-standardized features, with explanations.

Method
------
Similarity is weighted cosine on z-scores within a role group:

    sim(a, b) = sum(w * a * b) / (sqrt(sum(w * a^2)) * sqrt(sum(w * b^2)))

which is plain cosine on sqrt(w)-scaled vectors. Because the inputs are z-scores
(centred on the role mean), this measures whether two players deviate from their
role's average in the same directions and proportions — i.e. profile *shape*.

Two deliberate choices:

1. **Filter before scoring.** Candidates are restricted to the reference player's
   role, and to the minutes/coverage thresholds, *before* similarity is computed.
   A centre-back must never surface for a winger query because of a coincidental
   statistical shape.
2. **Every result is explainable.** The weighted per-feature distance is decomposed
   so each candidate comes with the dimensions where the players are closest and
   the dimensions where they differ most.

Similarity is descriptive. It says two players *do similar things* — not that they
are equally good, would fit the same system, or are worth the same fee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_matrix import RoleMatrix


def weighted_cosine_matrix(z: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Full player-by-player weighted cosine similarity for one role."""
    scaled = z.to_numpy() * np.sqrt(weights)          # cosine on sqrt(w)-scaled rows
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0                            # a fully average player
    unit = scaled / norms
    sim = unit @ unit.T
    return pd.DataFrame(np.clip(sim, -1.0, 1.0), index=z.index, columns=z.index)


def explain_pair(matrix: RoleMatrix, ref_id: int, cand_id: int) -> pd.DataFrame:
    """Per-feature breakdown of why two players are (dis)similar.

    `gap` is the weighted absolute z-distance on that feature: small means the
    players match there, large means it is a genuine difference. Raw values are
    included so the explanation is readable in football terms.
    """
    ref_z, cand_z = matrix.z.loc[ref_id], matrix.z.loc[cand_id]
    gap = (ref_z - cand_z).abs() * matrix.weights
    return pd.DataFrame(
        {
            "feature": matrix.features,
            "gap": gap.to_numpy(),
            "reference_value": matrix.raw.loc[ref_id].to_numpy(),
            "candidate_value": matrix.raw.loc[cand_id].to_numpy(),
            "reference_z": ref_z.to_numpy(),
            "candidate_z": cand_z.to_numpy(),
        }
    ).sort_values("gap").reset_index(drop=True)


def find_similar(
    matrix: RoleMatrix,
    player_season: pd.DataFrame,
    player_id: int,
    top_n: int = 10,
    min_minutes: float = 0.0,
    min_coverage: float = 0.7,
    exclude_same_team: bool = False,
) -> pd.DataFrame:
    """Rank the most similar players to `player_id` within its role group.

    Filtering (role, minutes, coverage) happens before scoring. Returns one row
    per candidate with the similarity score and context flags.
    """
    if player_id not in matrix.z.index:
        raise KeyError(f"player_id {player_id} is not in role group {matrix.role}")

    meta = player_season.set_index("player_id")
    ref = meta.loc[player_id]

    eligible = matrix.z.index[
        (matrix.coverage >= min_coverage)
        & (meta.loc[matrix.z.index, "minutes"] >= min_minutes)
        & (matrix.z.index != player_id)
    ]
    if exclude_same_team:
        same_team = meta.loc[eligible, "team_id"] == ref["team_id"]
        eligible = eligible[~same_team.to_numpy()]

    if len(eligible) == 0:
        return pd.DataFrame(
            columns=["player_id", "player_name", "role", "minutes", "similarity"]
        )

    sim = weighted_cosine_matrix(matrix.z, matrix.weights).loc[player_id, eligible]
    ranked = sim.sort_values(ascending=False).head(top_n)

    rows = []
    for cand_id, score in ranked.items():
        cand = meta.loc[cand_id]
        breakdown = explain_pair(matrix, player_id, cand_id)
        rows.append(
            {
                "player_id": cand_id,
                "player_name": cand["player_name"],
                "role": matrix.role,
                "minutes": round(float(cand["minutes"]), 1),
                "similarity": round(float(score), 4),
                "coverage": round(float(matrix.coverage.loc[cand_id]), 2),
                "low_minutes": bool(cand["low_minutes"]),
                "most_similar_on": breakdown["feature"].head(5).tolist(),
                "biggest_differences": breakdown["feature"].tail(3).tolist()[::-1],
            }
        )
    return pd.DataFrame(rows)
