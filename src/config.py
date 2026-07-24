"""Shared paths and access to the frozen analytical config.

Every module reads dataset keys, thresholds and role groups from here instead
of hard-coding them, so changing the dataset means editing one YAML file.
"""

from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "artifacts" / "feature_config.yml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def load_config() -> dict[str, Any]:
    """Load and return the parsed feature_config.yml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_keys() -> tuple[int, int]:
    """Return (competition_id, season_id) for the configured dataset."""
    dataset = load_config()["dataset"]
    return dataset["competition_id"], dataset["season_id"]
