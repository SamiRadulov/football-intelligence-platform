"""Build dim_players: one row per unique player across all matches."""

from __future__ import annotations

import pandas as pd


def build_dim_players(all_lineups_raw: list[list[dict]]) -> pd.DataFrame:
    """Collect distinct players from every match's lineup file.

    A player appears in many matches; we keep one row keyed by the stable
    StatsBomb `player_id`. `nickname` (the common short name) and `country`
    are taken from the first lineup that provides them.
    """
    players: dict[int, dict] = {}
    for lineups_raw in all_lineups_raw:
        for team in lineups_raw:
            for player in team["lineup"]:
                pid = player["player_id"]
                if pid not in players:
                    players[pid] = {
                        "player_id": pid,
                        "player_name": player["player_name"],
                        "nickname": player.get("player_nickname"),
                        "country": (player.get("country") or {}).get("name"),
                    }
    return (
        pd.DataFrame(players.values())
        .sort_values("player_id")
        .reset_index(drop=True)
    )
