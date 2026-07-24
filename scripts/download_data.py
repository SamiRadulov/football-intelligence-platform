"""Phase 1 ingestion CLI: download the raw StatsBomb layer for the configured dataset.

Usage (from the repo root):
    .venv/Scripts/python scripts/download_data.py            # full dataset
    .venv/Scripts/python scripts/download_data.py --limit 5  # quick smoke test
    .venv/Scripts/python scripts/download_data.py --force    # re-download everything

The dataset (competition_id, season_id) comes from artifacts/feature_config.yml,
so this script never hard-codes which competition to fetch.
"""

import argparse
import sys
from pathlib import Path

# Make `import src...` work when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR, load_config  # noqa: E402
from src.ingestion.raw_download import download_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="fetch at most N matches")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--ref", default="master", help="git ref/SHA of statsbomb/open-data")
    args = parser.parse_args()

    dataset = load_config()["dataset"]
    print(
        f"Downloading {dataset['competition_name']} {dataset['season_name']} "
        f"(competition_id={dataset['competition_id']}, season_id={dataset['season_id']})"
    )
    download_dataset(
        competition_id=dataset["competition_id"],
        season_id=dataset["season_id"],
        raw_dir=RAW_DIR,
        ref=args.ref,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
