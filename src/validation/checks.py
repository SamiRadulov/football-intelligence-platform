"""Data-quality checks for the canonical staging tables.

Each check returns a CheckResult. `passed=False` with `hard=True` means the
build should fail; `hard=False` is a warning worth surfacing but not fatal
(e.g. a match with a red card legitimately breaks the team-minutes identity).

The checks encode the guarantees the rest of the platform relies on:
schema (required columns exist), uniqueness (keys are unique), referential
integrity (facts point at real dimensions) and football sanity (11 starters
per team, nobody plays longer than the match).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Minutes tolerance for the team-minutes reconciliation (added time rounding).
TEAM_MINUTES_TOLERANCE = 3.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    hard: bool = True


def _require_columns(df: pd.DataFrame, columns: set[str], table: str) -> CheckResult:
    missing = columns - set(df.columns)
    return CheckResult(
        name=f"schema:{table}",
        passed=not missing,
        detail="ok" if not missing else f"missing columns: {sorted(missing)}",
    )


def _unique_key(df: pd.DataFrame, keys: list[str], table: str) -> CheckResult:
    n_dupes = int(df.duplicated(subset=keys).sum())
    return CheckResult(
        name=f"unique:{table}:{'+'.join(keys)}",
        passed=n_dupes == 0,
        detail="ok" if n_dupes == 0 else f"{n_dupes} duplicate rows",
    )


def _foreign_key(
    child: pd.DataFrame, child_col: str, parent_ids: set, table: str
) -> CheckResult:
    values = child[child_col].dropna()
    orphans = set(values) - parent_ids
    return CheckResult(
        name=f"fk:{table}.{child_col}",
        passed=not orphans,
        detail="ok" if not orphans else f"{len(orphans)} unknown ids (e.g. {list(orphans)[:3]})",
    )


def run_all_checks(
    dim_matches: pd.DataFrame,
    dim_players: pd.DataFrame,
    fact_lineups: pd.DataFrame,
    fact_events: pd.DataFrame,
) -> list[CheckResult]:
    """Run every quality check and return the results in order."""
    results: list[CheckResult] = []
    match_ids = set(dim_matches["match_id"])
    player_ids = set(dim_players["player_id"])

    # 1. Schema: the columns other phases depend on must exist.
    results.append(_require_columns(dim_matches, {"match_id", "home_team", "away_team"}, "dim_matches"))
    results.append(_require_columns(dim_players, {"player_id", "player_name"}, "dim_players"))
    results.append(_require_columns(
        fact_lineups, {"match_id", "player_id", "minutes", "is_starter", "played"}, "fact_lineups"))
    results.append(_require_columns(
        fact_events, {"event_id", "match_id", "type", "player_id"}, "fact_events"))

    # 2. Uniqueness of keys.
    results.append(_unique_key(dim_matches, ["match_id"], "dim_matches"))
    results.append(_unique_key(dim_players, ["player_id"], "dim_players"))
    results.append(_unique_key(fact_lineups, ["match_id", "player_id"], "fact_lineups"))
    results.append(_unique_key(fact_events, ["event_id"], "fact_events"))

    # 3. Referential integrity: facts must point at known matches/players.
    results.append(_foreign_key(fact_lineups, "match_id", match_ids, "fact_lineups"))
    results.append(_foreign_key(fact_lineups, "player_id", player_ids, "fact_lineups"))
    results.append(_foreign_key(fact_events, "match_id", match_ids, "fact_events"))
    results.append(_foreign_key(fact_events, "player_id", player_ids, "fact_events"))

    # 4. Football sanity: exactly 11 starters per team per match.
    starters = fact_lineups[fact_lineups["is_starter"]]
    starter_counts = starters.groupby(["match_id", "team_id"]).size()
    bad_starter_groups = starter_counts[starter_counts != 11]
    results.append(CheckResult(
        name="minutes:11_starters_per_team",
        passed=bad_starter_groups.empty,
        detail="ok" if bad_starter_groups.empty else f"{len(bad_starter_groups)} team-matches without 11 starters",
    ))

    # 5. Nobody plays longer than their match lasted (+ small tolerance).
    match_length = (
        fact_events.groupby("match_id")
        .apply(lambda g: (g["minute"] + g["second"] / 60).max(), include_groups=False)
        .rename("match_length")
    )
    lineups_with_length = fact_lineups.merge(match_length, on="match_id", how="left")
    over = lineups_with_length[
        lineups_with_length["minutes"] > lineups_with_length["match_length"] + 1
    ]
    results.append(CheckResult(
        name="minutes:within_match_length",
        passed=over.empty,
        detail="ok" if over.empty else f"{len(over)} players exceed match length",
    ))

    # 6. Team-minutes reconciliation (warning): outfield play time should be
    #    ~11 x match length; red cards legitimately reduce it, so this warns.
    team_minutes = fact_lineups.groupby(["match_id", "team_id"])["minutes"].sum()
    expected = match_length * 11
    recon = team_minutes.reset_index().merge(expected, on="match_id", how="left")
    recon["diff"] = recon["minutes"] - recon["match_length"]
    off = recon[recon["diff"].abs() > TEAM_MINUTES_TOLERANCE]
    results.append(CheckResult(
        name="minutes:team_reconciliation",
        passed=off.empty,
        detail="ok" if off.empty else (
            f"{len(off)} team-matches differ >{TEAM_MINUTES_TOLERANCE}min "
            "(temporary player-off gaps, red cards, or contradictory source lineups)"
        ),
        hard=False,
    ))

    return results


def summarize(results: list[CheckResult]) -> tuple[bool, str]:
    """Return (all_hard_checks_passed, printable_report)."""
    lines = []
    hard_ok = True
    for r in results:
        status = "PASS" if r.passed else ("WARN" if not r.hard else "FAIL")
        if not r.passed and r.hard:
            hard_ok = False
        lines.append(f"  [{status}] {r.name}: {r.detail}")
    return hard_ok, "\n".join(lines)
