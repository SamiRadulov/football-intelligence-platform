# Football Intelligence Platform

Player recruitment similarity and team playing-style classification, built on [StatsBomb Open Data](https://github.com/statsbomb/open-data).

> **Status: early development.** This README will grow with the project.

## What it does

An end-to-end football analytics platform answering two questions:

1. **Player Recruitment Similarity** — who resembles a reference player, filtered by role, age, minutes and competition, with an interpretable explanation of *why* they are similar.
2. **Team Playing-Style Classification** — which teams play alike, grouped into data-driven style clusters with evidence-based labels.

Explicit non-goal: no match-result, scoreline or betting prediction. Results are descriptive — similarity does not mean equal quality or transfer suitability.

## Architecture

```
RAW (StatsBomb JSON) → STAGING (Parquet) → CURATED (DuckDB) → FEATURES (player/team matrices) → APP (Streamlit)
```

Planned stack: Python (pandas/Polars), Parquet + DuckDB, scikit-learn, Plotly + mplsoccer, Streamlit, pytest + Ruff + GitHub Actions.

## Quick start

```bash
# 1. Create the environment (Python 3.12+)
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 2. Download the raw StatsBomb layer for the configured dataset (PL 2015/16).
#    ~1.2 GB into data/raw/ (gitignored). Use --limit N for a quick sample.
.venv/Scripts/python scripts/download_data.py

# 3. Build the canonical staging layer (Parquet + DuckDB) and run quality checks.
.venv/Scripts/python scripts/build_staging.py

# 4. Build the player feature marts (role-aware per-90 / percentile metrics).
.venv/Scripts/python scripts/build_player_features.py

# 5. Query the similarity engine
.venv/Scripts/python scripts/find_similar.py "Kante" --top 5

# 6. Run the tests and the similarity validation
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/validate_similarity.py
```

## Data layers

| Layer | Location | Contents |
|---|---|---|
| Raw | `data/raw/` (gitignored) | Unchanged StatsBomb JSON + `manifest.json` |
| Staging | `data/staging/` (gitignored) | Canonical Parquet: `dim_matches`, `dim_players`, `fact_lineups`, `fact_events` |
| Curated | `data/curated.duckdb` (gitignored) | Canonical tables **and** feature marts in DuckDB, queried by the app |

**Feature marts** (built on the curated layer): `mart_player_match` (raw metric
counts per player-match) and `mart_player_season` (role-aware per-90,
possession-adjusted and percentile features for players with ≥ 600 minutes).

Build steps read raw and write staging/curated; nothing overwrites raw. See
[docs/METRIC_DICTIONARY.md](docs/METRIC_DICTIONARY.md) for metric definitions.

The dataset is defined once in [artifacts/feature_config.yml](artifacts/feature_config.yml); every step reads it from there.

## Data source & attribution

This project uses **StatsBomb Open Data**. See [DATA_SOURCES.md](DATA_SOURCES.md) for attribution and licensing details. Raw data is not redistributed in this repository.

## Methodology

[METHODOLOGY.md](METHODOLOGY.md) documents the modelling choices, validation results
and limitations. In short: features are standardized within role group, similarity is
weighted cosine on those z-scores, candidates are filtered *before* scoring, and every
result is explained by decomposing the weighted feature distance.

## License

Code is licensed under the [MIT License](LICENSE). Data is governed by the StatsBomb Open Data user agreement.
