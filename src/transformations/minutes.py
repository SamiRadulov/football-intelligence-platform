"""Minutes played and per-match lineup rows, derived from lineups + events.

StatsBomb records each player's time on the pitch as one or more *position
spells* in lineups.json, e.g.

    {"position": "Left Wing", "from": "00:00", "to": "56:53",
     "start_reason": "Starting XI", "end_reason": "Tactical Shift"}

The "from"/"to" clocks are absolute cumulative match time (a second-half spell
reads "56:53", not "11:53"), and "to": null means the player was on until the
final whistle. That makes minutes a direct subtraction once we know when the
match ended. Red cards, tactical shifts and substitutions all show up as spell
boundaries, so this one rule handles them without special-casing.
"""

from __future__ import annotations

import pandas as pd

STARTER_REASON = "Starting XI"


def parse_clock(clock: str) -> float:
    """Convert a "MM:SS" (or "HH:MM:SS") match clock into minutes as a float."""
    parts = [int(p) for p in clock.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes + seconds / 60
    hours, minutes, seconds = parts
    return hours * 60 + minutes + seconds / 60


def match_end_minute(events_raw: list[dict]) -> float:
    """The final-whistle time in minutes, i.e. the latest event in the match.

    Used to close out spells with "to": null. Stoppage time is included, so a
    full match is ~90 plus added time and every player in that match is scaled
    consistently.
    """
    return max(e["minute"] + e["second"] / 60 for e in events_raw)


def _spell_bounds(spell: dict, match_end: float) -> tuple[float, float]:
    start = parse_clock(spell["from"])
    end = parse_clock(spell["to"]) if spell["to"] else match_end
    return start, end


def build_lineup_rows(
    match_id: int, lineups_raw: list[dict], match_end: float
) -> list[dict]:
    """Return one fact_lineups row per player in the matchday squad.

    Players who did not leave the bench have empty `positions`; they get 0
    minutes and `played=False`. The recorded `position` is the one the player
    spent the most time in (a mid-match tactical shift does not erase their
    original role, but the dominant position is the most representative label).
    """
    rows = []
    for team in lineups_raw:
        for player in team["lineup"]:
            positions = player["positions"]
            played = len(positions) > 0

            if played:
                is_starter = positions[0]["start_reason"] == STARTER_REASON

                # Sum each spell's on-pitch time. Summing (rather than
                # last_end - first_start) excludes gaps when a player leaves
                # and returns (temporary "Player Off" / "Player On"), and the
                # longest spell gives the representative position.
                minutes = 0.0
                best_position, best_duration = None, -1.0
                for spell in positions:
                    start, end = _spell_bounds(spell, match_end)
                    duration = max(0.0, end - start)
                    minutes += duration
                    if duration > best_duration:
                        best_duration, best_position = duration, spell["position"]

                start_minute = _spell_bounds(positions[0], match_end)[0]
                end_minute = _spell_bounds(positions[-1], match_end)[1]
                position = best_position
            else:
                minutes, is_starter = 0.0, False
                start_minute = end_minute = None
                position = None

            rows.append(
                {
                    "match_id": match_id,
                    "team_id": team["team_id"],
                    "team": team["team_name"],
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": position,
                    "is_starter": is_starter,
                    "played": played,
                    "minutes": round(minutes, 2),
                    "start_minute": None if start_minute is None else round(start_minute, 2),
                    "end_minute": None if end_minute is None else round(end_minute, 2),
                }
            )
    return rows


def build_fact_lineups(lineup_rows: list[dict]) -> pd.DataFrame:
    """Assemble all per-match lineup rows into the fact_lineups table."""
    return (
        pd.DataFrame(lineup_rows)
        .sort_values(["match_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )
