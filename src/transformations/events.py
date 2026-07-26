"""Flatten nested StatsBomb event JSON into the canonical fact_events table.

Raw events are deeply nested dicts whose shape depends on the event type (a
Pass carries a `pass` block, a Shot a `shot` block, etc.). We flatten with
`json_normalize`, split the `[x, y]` coordinate lists into separate columns,
and keep a curated set of fields — the ones the metric dictionary needs. Every
event type is retained as a row; type-specific columns are simply null for
events that don't use them.
"""

from __future__ import annotations

import pandas as pd

# Coordinate list fields -> (x_column, y_column). Split before renaming.
_LOCATION_FIELDS = {
    "location": ("location_x", "location_y"),
    "pass.end_location": ("pass_end_x", "pass_end_y"),
    "carry.end_location": ("carry_end_x", "carry_end_y"),
    "shot.end_location": ("shot_end_x", "shot_end_y"),
}

# Source (dotted, from json_normalize) -> canonical column name.
_RENAME = {
    "id": "event_id",
    "index": "event_index",
    "type.name": "type",
    "possession_team.name": "possession_team",
    "play_pattern.name": "play_pattern",
    "team.id": "team_id",
    "team.name": "team",
    "player.id": "player_id",
    "player.name": "player_name",
    "position.name": "position",
    "pass.length": "pass_length",
    "pass.angle": "pass_angle",
    "pass.height.name": "pass_height",
    "pass.recipient.id": "pass_recipient_id",
    "pass.recipient.name": "pass_recipient_name",
    "pass.outcome.name": "pass_outcome",
    "pass.type.name": "pass_type",
    "pass.body_part.name": "pass_body_part",
    "pass.cross": "pass_cross",
    "pass.switch": "pass_switch",
    "pass.through_ball": "pass_through_ball",
    "pass.shot_assist": "pass_shot_assist",
    "pass.goal_assist": "pass_goal_assist",
    "shot.statsbomb_xg": "shot_xg",
    "shot.outcome.name": "shot_outcome",
    "shot.type.name": "shot_type",
    "shot.technique.name": "shot_technique",
    "shot.body_part.name": "shot_body_part",
    "shot.key_pass_id": "shot_key_pass_id",
    "dribble.outcome.name": "dribble_outcome",
    "duel.type.name": "duel_type",
    "duel.outcome.name": "duel_outcome",
    "interception.outcome.name": "interception_outcome",
    "ball_receipt.outcome.name": "ball_receipt_outcome",
    "foul_committed.card.name": "foul_card",
    "bad_behaviour.card.name": "bad_behaviour_card",
}

# Flags that raw JSON only includes when True; missing means False.
_BOOLEAN_FLAGS = [
    "under_pressure",
    "counterpress",
    "out",
    "pass_cross",
    "pass_switch",
    "pass_through_ball",
    "pass_shot_assist",
    "pass_goal_assist",
]

# Coordinate/measure columns stored as floats.
_FLOAT_COLUMNS = [
    "location_x", "location_y", "duration", "pass_length", "pass_angle",
    "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y",
    "shot_xg", "shot_end_x", "shot_end_y",
]

# Identity/counter columns stored as nullable integers (player_id is null on
# events with no player, e.g. Half Start).
_INT_COLUMNS = [
    "match_id", "event_index", "period", "minute", "second", "possession",
    "team_id", "player_id", "pass_recipient_id",
]

# Final column order of fact_events.
CANONICAL_COLUMNS = (
    ["event_id", "match_id", "event_index", "period", "minute", "second", "timestamp",
     "type", "possession", "possession_team", "play_pattern", "team_id", "team",
     "player_id", "player_name", "position",
     "location_x", "location_y", "duration", "under_pressure", "counterpress", "out",
     "pass_length", "pass_angle", "pass_height", "pass_end_x", "pass_end_y",
     "pass_recipient_id", "pass_recipient_name", "pass_outcome", "pass_type",
     "pass_body_part", "pass_cross", "pass_switch", "pass_through_ball",
     "pass_shot_assist", "pass_goal_assist",
     "shot_xg", "shot_outcome", "shot_type", "shot_technique", "shot_body_part",
     "shot_end_x", "shot_end_y", "shot_key_pass_id",
     "carry_end_x", "carry_end_y",
     "dribble_outcome", "duel_type", "duel_outcome", "interception_outcome",
     "ball_receipt_outcome", "foul_card", "bad_behaviour_card", "related_events"]
)


def _split_locations(df: pd.DataFrame) -> pd.DataFrame:
    for src, (x_col, y_col) in _LOCATION_FIELDS.items():
        if src in df.columns:
            values = df[src]
            df[x_col] = values.apply(lambda v: v[0] if isinstance(v, list) else None)
            df[y_col] = values.apply(
                lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None
            )
    return df


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force a consistent dtype per column so every match's Parquet file shares
    one schema (required for DuckDB to scan them as a single table)."""
    for flag in _BOOLEAN_FLAGS:
        df[flag] = df[flag].fillna(False).astype(bool)
    for col in _FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # Everything else is textual; related_events is a list -> join to a string.
    df["related_events"] = df["related_events"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else None
    )
    string_cols = [
        c for c in df.columns
        if c not in _BOOLEAN_FLAGS and c not in _FLOAT_COLUMNS and c not in _INT_COLUMNS
    ]
    for col in string_cols:
        df[col] = df[col].astype("string")
    return df


def flatten_events(match_id: int, events_raw: list[dict]) -> pd.DataFrame:
    """Flatten one match's events into canonical fact_events columns."""
    df = pd.json_normalize(events_raw, sep=".")
    df["match_id"] = match_id
    df = _split_locations(df)
    df = df.rename(columns=_RENAME)

    # Keep only canonical columns that exist; add any missing as null.
    present = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df = df[present].reindex(columns=CANONICAL_COLUMNS)

    return _apply_dtypes(df)
