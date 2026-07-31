"""Unit tests for the similarity engine (synthetic players, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.similarity.engine import explain_pair, find_similar, weighted_cosine_matrix
from src.similarity.feature_matrix import build_role_matrix, winsorize, zscore

WEIGHTS = {"shots_p90": 2.0, "key_passes_p90": 1.0, "padj_tackles_p90": 1.0}


def _season(rows: list[dict]) -> pd.DataFrame:
    base = {"role": "ST", "minutes": 2000.0, "low_minutes": False, "team_id": 1,
            "nickname": None, "primary_position": "Center Forward"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _matrix(df):
    return build_role_matrix(df, "ST", WEIGHTS, 0.01, 0.99)


# --- scaling helpers -------------------------------------------------------

def test_winsorize_pulls_in_the_outlier():
    s = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0])
    out = winsorize(s, 0.0, 0.8)
    assert out.max() == pytest.approx(s.quantile(0.8))  # outlier clipped to the cap
    assert out.max() < 100.0
    assert out.min() == 0.0
    assert out.iloc[:4].tolist() == [0.0, 1.0, 2.0, 3.0]  # in-range values untouched


def test_winsorize_ignores_nulls():
    s = pd.Series([1.0, None, 2.0, 50.0])
    out = winsorize(s, 0.0, 0.5)
    assert out.isna().sum() == 1        # the null stays null
    assert out.max() < 50.0


def test_zscore_of_constant_feature_is_zero():
    assert (zscore(pd.Series([5.0, 5.0, 5.0])) == 0.0).all()


def test_zscore_is_standardized():
    z = zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=0) - 1.0) < 1e-12


# --- similarity maths ------------------------------------------------------

def test_identical_vectors_have_similarity_one():
    z = pd.DataFrame({"a": [1.0, 1.0], "b": [-2.0, -2.0]}, index=[1, 2])
    sim = weighted_cosine_matrix(z, np.array([1.0, 1.0]))
    assert sim.loc[1, 2] == pytest.approx(1.0)
    assert sim.loc[1, 1] == pytest.approx(1.0)


def test_opposite_vectors_have_similarity_minus_one():
    z = pd.DataFrame({"a": [1.0, -1.0], "b": [2.0, -2.0]}, index=[1, 2])
    sim = weighted_cosine_matrix(z, np.array([1.0, 1.0]))
    assert sim.loc[1, 2] == pytest.approx(-1.0)


def test_similarity_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    z = pd.DataFrame(rng.normal(size=(6, 3)), columns=["a", "b", "c"], index=range(6))
    sim = weighted_cosine_matrix(z, np.array([1.0, 2.0, 0.5]))
    assert np.allclose(sim.to_numpy(), sim.to_numpy().T)


def test_all_average_player_does_not_divide_by_zero():
    # A player exactly on the role mean has a zero vector; must not produce NaN.
    z = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]}, index=[1, 2])
    sim = weighted_cosine_matrix(z, np.array([1.0, 1.0]))
    assert np.isfinite(sim.to_numpy()).all()


def test_weights_change_the_ranking():
    # Player 2 matches the reference on 'shots_p90', player 3 on 'key_passes_p90'.
    z = pd.DataFrame(
        {"shots_p90": [1.0, 1.0, 0.0], "key_passes_p90": [1.0, 0.0, 1.0]},
        index=[1, 2, 3],
    )
    shots_heavy = weighted_cosine_matrix(z, np.array([5.0, 1.0])).loc[1]
    passes_heavy = weighted_cosine_matrix(z, np.array([1.0, 5.0])).loc[1]
    assert shots_heavy[2] > shots_heavy[3]
    assert passes_heavy[3] > passes_heavy[2]


# --- matrix construction ---------------------------------------------------

def test_coverage_reflects_missing_features():
    df = _season([
        {"player_id": 1, "player_name": "Full", "shots_p90": 3.0,
         "key_passes_p90": 1.0, "padj_tackles_p90": 0.5},
        {"player_id": 2, "player_name": "Partial", "shots_p90": 2.0,
         "key_passes_p90": None, "padj_tackles_p90": 0.4},
    ])
    m = _matrix(df)
    assert m.coverage.loc[1] == pytest.approx(1.0)
    assert m.coverage.loc[2] == pytest.approx(2 / 3)
    assert m.z.loc[2, "key_passes_p90"] == 0.0   # imputed to the role mean


def test_missing_feature_in_mart_raises():
    df = _season([{"player_id": 1, "player_name": "X", "shots_p90": 1.0,
                   "key_passes_p90": 1.0}])
    with pytest.raises(KeyError, match="padj_tackles_p90"):
        _matrix(df)


# --- filtering and results -------------------------------------------------

def _three_strikers():
    return _season([
        {"player_id": 1, "player_name": "Reference", "shots_p90": 3.0,
         "key_passes_p90": 1.0, "padj_tackles_p90": 0.5},
        {"player_id": 2, "player_name": "Similar", "shots_p90": 2.9,
         "key_passes_p90": 1.1, "padj_tackles_p90": 0.5},
        {"player_id": 3, "player_name": "Different", "shots_p90": 0.2,
         "key_passes_p90": 3.0, "padj_tackles_p90": 2.0, "minutes": 700.0},
    ])


def test_reference_player_is_excluded_from_own_results():
    df = _three_strikers()
    out = find_similar(_matrix(df), df, player_id=1)
    assert 1 not in out["player_id"].tolist()


def test_min_minutes_filter_applies_before_scoring():
    df = _three_strikers()
    out = find_similar(_matrix(df), df, player_id=1, min_minutes=1000)
    assert out["player_id"].tolist() == [2]     # player 3 has 700 minutes


def test_results_are_deterministic():
    df = _three_strikers()
    m = _matrix(df)
    first = find_similar(m, df, player_id=1)
    second = find_similar(m, df, player_id=1)
    pd.testing.assert_frame_equal(first, second)


def test_unknown_player_raises():
    df = _three_strikers()
    with pytest.raises(KeyError, match="999"):
        find_similar(_matrix(df), df, player_id=999)


def test_explanations_are_ordered_closest_first():
    df = _three_strikers()
    m = _matrix(df)
    breakdown = explain_pair(m, 1, 3)
    assert breakdown["gap"].is_monotonic_increasing
    assert set(breakdown["feature"]) == set(WEIGHTS)


def test_result_row_carries_context_flags():
    df = _three_strikers()
    out = find_similar(_matrix(df), df, player_id=1, min_coverage=0.0)
    row = out.iloc[0]
    assert row["player_name"] == "Similar"
    assert 0.0 <= row["coverage"] <= 1.0
    assert len(row["most_similar_on"]) <= 5
    assert len(row["biggest_differences"]) <= 3
