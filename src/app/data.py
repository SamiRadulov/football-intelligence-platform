"""Cached data access for the Streamlit app.

Streamlit re-runs the whole script on every interaction, so everything that
touches disk or does real work is cached:

    @st.cache_resource  the DuckDB connection and the fitted style space
    @st.cache_data      dataframes loaded from the marts

Without this, moving a slider would re-open the database and re-fit PCA.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import streamlit as st

from ..clustering.style_model import StyleSpace, build_style_space
from ..config import DATA_DIR, PROJECT_ROOT, load_config
from ..similarity.feature_matrix import RoleMatrix, build_role_matrix

DB_PATH = DATA_DIR / "curated.duckdb"
MANIFEST_PATH = DATA_DIR / "raw" / "manifest.json"


@st.cache_resource
def connection() -> duckdb.DuckDBPyConnection:
    """One read-only DuckDB connection shared across reruns."""
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_resource
def config() -> dict:
    return load_config()


def _table(name: str) -> pd.DataFrame:
    return connection().execute(f"SELECT * FROM {name}").df()


@st.cache_data
def player_season() -> pd.DataFrame:
    return _table("mart_player_season")


@st.cache_data
def team_season() -> pd.DataFrame:
    return _table("mart_team_season")


@st.cache_data
def team_match() -> pd.DataFrame:
    return _table("mart_team_match")


@st.cache_data
def team_style() -> pd.DataFrame:
    return _table("model_team_style")


@st.cache_data
def dataset_summary() -> dict:
    """Coverage figures and the raw-layer download timestamp for the home page."""
    con = connection()
    counts = {
        "matches": con.execute("SELECT count(*) FROM dim_matches").fetchone()[0],
        "events": con.execute("SELECT count(*) FROM fact_events").fetchone()[0],
        "players_qualified": con.execute("SELECT count(*) FROM mart_player_season").fetchone()[0],
        "teams": con.execute("SELECT count(*) FROM mart_team_season").fetchone()[0],
    }
    downloaded_at = None
    if MANIFEST_PATH.exists():
        downloaded_at = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("downloaded_at")
    counts["downloaded_at"] = downloaded_at
    return counts


@st.cache_resource
def role_matrix(role: str) -> RoleMatrix:
    """The standardized feature matrix for one role group."""
    cfg = config()
    return build_role_matrix(
        player_season(), role, cfg["role_features"][role],
        cfg["similarity"]["winsorize_lower"], cfg["similarity"]["winsorize_upper"],
    )


@st.cache_resource
def style_space() -> StyleSpace:
    """The fitted PCA style space for teams."""
    cfg = config()["clustering"]
    return build_style_space(team_season(), cfg["pca_variance_target"], cfg["random_state"])


def display_name(row: pd.Series) -> str:
    """Prefer a player's common short name over their full registered name."""
    nickname = row.get("nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname
    return row["player_name"]


@st.cache_data
def player_options() -> pd.DataFrame:
    """Selectable players with a readable label, sorted by name."""
    players = player_season()[
        ["player_id", "player_name", "nickname", "role", "primary_position",
         "minutes", "team_id"]
    ].copy()
    players["label"] = players.apply(
        lambda r: f"{display_name(r)}  ·  {r['role']}  ·  {r['minutes']:.0f} min", axis=1
    )
    return players.sort_values("label").reset_index(drop=True)


def read_markdown(filename: str) -> str:
    """Load a documentation file from the repository root or docs/."""
    for candidate in (PROJECT_ROOT / filename, PROJECT_ROOT / "docs" / filename):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return f"_{filename} not found._"
