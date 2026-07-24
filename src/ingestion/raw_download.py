"""Download the full raw layer for one competition-season and record a manifest.

Layout produced under `raw_dir` (default data/raw/):

    competitions.json
    matches/{competition_id}/{season_id}.json
    lineups/{match_id}.json
    events/{match_id}.json
    manifest.json

The manifest is the audit trail: it records the exact source revision, when the
download ran, how many files of each kind were fetched, the total size, the list
of match IDs, and any files that failed to download.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import statsbomb_open_data as sbod


def _load_match_ids(matches_file: Path) -> list[int]:
    """Read the saved matches JSON and return its match IDs."""
    matches = json.loads(matches_file.read_text(encoding="utf-8"))
    return [m["match_id"] for m in matches]


def download_dataset(
    competition_id: int,
    season_id: int,
    raw_dir: Path,
    ref: str = "master",
    force: bool = False,
    limit: int | None = None,
    progress_every: int = 25,
) -> dict:
    """Download competitions, matches, lineups and events; return the manifest.

    `limit` caps how many matches are fetched (useful for a fast smoke test).
    `force` re-downloads files that already exist.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    total_bytes = 0
    failures: list[dict] = []

    # 1. Master competitions list (small; always refresh so it stays current).
    competitions_file = raw_dir / "competitions.json"
    total_bytes += sbod.download_file(
        sbod.competitions_url(ref), competitions_file, session, force=True
    )

    # 2. The match list for this competition-season.
    matches_file = raw_dir / "matches" / str(competition_id) / f"{season_id}.json"
    total_bytes += sbod.download_file(
        sbod.matches_url(competition_id, season_id, ref), matches_file, session, force=True
    )
    match_ids = _load_match_ids(matches_file)
    if limit is not None:
        match_ids = match_ids[:limit]
    print(f"Found {len(match_ids)} matches to fetch (ref={ref}).")

    # 3. Lineups and events, one file each per match.
    for index, match_id in enumerate(match_ids, start=1):
        for kind, url_fn in (("lineups", sbod.lineups_url), ("events", sbod.events_url)):
            dest = raw_dir / kind / f"{match_id}.json"
            try:
                total_bytes += sbod.download_file(url_fn(match_id, ref), dest, session, force)
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({"kind": kind, "match_id": match_id, "error": str(exc)})
                print(f"  ! {kind} {match_id}: {exc}")
        if index % progress_every == 0 or index == len(match_ids):
            print(f"  {index}/{len(match_ids)} matches done")

    manifest = {
        "dataset": {"competition_id": competition_id, "season_id": season_id},
        "source": {"base": "statsbomb/open-data", "ref": ref},
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "matches": len(match_ids),
            "lineups": len(match_ids) - sum(f["kind"] == "lineups" for f in failures),
            "events": len(match_ids) - sum(f["kind"] == "events" for f in failures),
        },
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1_000_000, 1),
        "match_ids": match_ids,
        "failures": failures,
    }
    manifest_file = raw_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_file} ({manifest['total_mb']} MB total).")
    if failures:
        print(f"WARNING: {len(failures)} file(s) failed — see manifest 'failures'.")
    return manifest
