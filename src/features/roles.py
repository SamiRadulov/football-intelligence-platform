"""Map StatsBomb positions to broad role groups and assign each player a season role.

StatsBomb records ~25 granular positions ("Right Center Back", "Left Wing", ...).
For role-aware comparison we collapse these into the five outfield groups defined
in artifacts/feature_config.yml:

    CB    centre-backs
    FB    full-backs / wing-backs
    CM    central midfielders (incl. defensive/holding)
    AM_W  attacking midfielders / wingers / wide midfielders
    ST    strikers / centre-forwards

Goalkeepers (and anything unrecognised) return None and are excluded from the
player mart. A player's season role is the group they spent the most minutes in.
"""

from __future__ import annotations

import pandas as pd

# Exact wide-midfield names that we treat as wingers rather than central mids.
_WIDE_MIDFIELD = {"Left Midfield", "Right Midfield"}


def classify_position(position: str | None) -> str | None:
    """Return the role-group key for a StatsBomb position name, or None.

    Order matters: "Wing Back" must be caught as a full-back before the generic
    "Wing" rule, and "Center Back" before the generic "Back" rule.
    """
    if not position:
        return None
    name = position.strip()

    if name == "Goalkeeper":
        return None
    if "Back" in name:
        # Right/Left/Center Back -> CB; Right/Left Back and Wing Backs -> FB.
        return "CB" if "Center Back" in name else "FB"
    if "Forward" in name or "Striker" in name:
        return "ST"
    if "Wing" in name:  # Left/Right Wing (Wing Backs already handled above)
        return "AM_W"
    if "Attacking Midfield" in name:
        return "AM_W"
    if "Midfield" in name:
        return "AM_W" if name in _WIDE_MIDFIELD else "CM"
    return None


def assign_player_roles(fact_lineups: pd.DataFrame) -> pd.DataFrame:
    """Assign each player their season role from their per-match positions.

    Returns one row per player: role (the group with the most minutes),
    total minutes, matches played, and the single most-played position name.
    Players whose minutes are all in unclassifiable positions (goalkeepers)
    are dropped.
    """
    played = fact_lineups[fact_lineups["played"]].copy()
    played["role"] = played["position"].map(classify_position)

    classified = played[played["role"].notna()]

    # Minutes per (player, role) -> the role with the most minutes wins.
    role_minutes = (
        classified.groupby(["player_id", "role"])["minutes"].sum().reset_index()
    )
    best_role = (
        role_minutes.sort_values("minutes", ascending=False)
        .drop_duplicates("player_id")
        .set_index("player_id")["role"]
    )

    # Most-played single position label (for display/inspection).
    pos_minutes = (
        classified.groupby(["player_id", "position"])["minutes"].sum().reset_index()
    )
    best_position = (
        pos_minutes.sort_values("minutes", ascending=False)
        .drop_duplicates("player_id")
        .set_index("player_id")["position"]
    )

    totals = classified.groupby("player_id").agg(
        minutes=("minutes", "sum"),
        matches_played=("match_id", "nunique"),
    )

    out = totals.join(best_role.rename("role")).join(best_position.rename("primary_position"))
    return out.reset_index().sort_values("player_id").reset_index(drop=True)
