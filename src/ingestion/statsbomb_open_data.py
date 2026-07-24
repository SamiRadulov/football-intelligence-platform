"""URL layout of the StatsBomb Open Data GitHub repository, plus a single-file
downloader.

The open data lives as static JSON files under a predictable path structure:

    data/competitions.json
    data/matches/{competition_id}/{season_id}.json
    data/lineups/{match_id}.json
    data/events/{match_id}.json

We download those files verbatim. `ref` pins which git revision to read; it
defaults to "master" but can be set to a commit SHA for full reproducibility.
"""

from pathlib import Path

import requests

# raw.githubusercontent.com serves file contents directly (no HTML wrapper).
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/{ref}/data"

REQUEST_TIMEOUT = 30  # seconds


def competitions_url(ref: str = "master") -> str:
    return f"{RAW_BASE.format(ref=ref)}/competitions.json"


def matches_url(competition_id: int, season_id: int, ref: str = "master") -> str:
    return f"{RAW_BASE.format(ref=ref)}/matches/{competition_id}/{season_id}.json"


def lineups_url(match_id: int, ref: str = "master") -> str:
    return f"{RAW_BASE.format(ref=ref)}/lineups/{match_id}.json"


def events_url(match_id: int, ref: str = "master") -> str:
    return f"{RAW_BASE.format(ref=ref)}/events/{match_id}.json"


def download_file(
    url: str,
    dest: Path,
    session: requests.Session | None = None,
    force: bool = False,
) -> int:
    """Download `url` to `dest`, returning the file size in bytes.

    Idempotent: if `dest` already exists and is non-empty, it is kept and its
    size returned, unless `force` is True. The download is written to a
    temporary file first and renamed on success, so an interrupted run never
    leaves a half-written JSON file behind.
    """
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest.stat().st_size

    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest.stat().st_size
