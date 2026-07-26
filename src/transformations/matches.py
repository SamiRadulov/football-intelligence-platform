"""Build dim_matches: one clean row per match from the nested matches JSON."""

from __future__ import annotations

import pandas as pd


def build_dim_matches(matches_raw: list[dict]) -> pd.DataFrame:
    """Flatten the raw match list into one row per match.

    Only the fields we actually need downstream are kept; managers, 360
    metadata and other nested blocks are dropped here (they can be re-derived
    from the raw JSON if ever needed).
    """
    rows = []
    for m in matches_raw:
        rows.append(
            {
                "match_id": m["match_id"],
                "competition_id": m["competition"]["competition_id"],
                "season_id": m["season"]["season_id"],
                "match_date": m.get("match_date"),
                "match_week": m.get("match_week"),
                "home_team_id": m["home_team"]["home_team_id"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team_id": m["away_team"]["away_team_id"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
                "stadium": (m.get("stadium") or {}).get("name"),
                "referee": (m.get("referee") or {}).get("name"),
            }
        )
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df.sort_values("match_id").reset_index(drop=True)
