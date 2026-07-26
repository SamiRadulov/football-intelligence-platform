# Metric Dictionary (MVP — frozen in Phase 0)

Dataset: StatsBomb Open Data, Premier League 2015/16 (`competition_id=2`, `season_id=27`).

This is the frozen MVP metric set. Adding a metric later is allowed; silently changing
a definition is not — update this file and the methodology page together.

## Global rules

- **Minutes** are calculated from each player's position spells in `lineups.json`,
  never assumed to be 90. Each spell has an absolute cumulative match clock
  (`from`/`to`, where `to: null` = played to the final whistle). A player's minutes
  = the **sum of each spell's duration** (summing, not last−first, so a temporary
  "Player Off"/"Player On" gap is excluded), with the final whistle taken as the
  latest event time in that match. **Stoppage time is included**, so a full match is
  ~90 + added time (e.g. 94.4) and every player in a given match is scaled by the
  same match length. Substitutions, tactical shifts and red cards all appear as spell
  boundaries, so no event-by-event special-casing is needed.

## Known data-quality limitations

- **Team-minutes reconciliation** (a warning-level check): summed outfield minutes
  should equal 11 × match length. In ~15 of 760 team-matches (2%) they deviate by more
  than 3 minutes, for three genuine source-data reasons, none of them pipeline bugs:
  - *Negative* — a player leaves temporarily ("Player Off") without an immediate
    replacement, so the team really is short-handed for those minutes; or a red card
    reduces the team to ten.
  - *Positive* — the source lineup is internally inconsistent (e.g. a player tagged
    "Substitution - Off" who then returns via "Player On"), so StatsBomb records 12
    players on the pitch.
  The check surfaces these rather than hiding them. The per-match error is a few
  minutes, negligible once aggregated to season per-90 over the 600-minute threshold.
- **Per-90**: `metric_per90 = metric_count / minutes * 90`. Null when season minutes < 600
  (see `artifacts/feature_config.yml`).
- **Rates/percentages** are kept separate from counts and require a minimum number of
  attempts (e.g. pass completion needs ≥ 100 attempted passes) before they are shown.
- **Possession adjustment (padj)**: defensive counts are scaled by opponent possession
  share so players in low-possession teams are not automatically favoured:
  `padj_metric = metric * 0.5 / (1 - team_possession_share)` (sigmoid variants may be
  evaluated in Phase 3; any change is recorded here).
- **Winsorization**: features are clipped at the 1st/99th percentile before scaling;
  raw values are always what the user sees.
- **Standardization**: z-scores within role group; users see percentiles (within role,
  within this competition-season).

## Operational definitions (as implemented in Phase 3)

Pitch geometry uses StatsBomb units (120 × 80, the acting team always attacks
towards x = 120):
- **opponent goal centre** = (120, 40); distance-to-goal = `hypot(120 − x, 40 − y)`
- **penalty box** = `x ≥ 102 and 18 ≤ y ≤ 62`; **final third** = `x ≥ 80`

Thresholds and rules:
- **Forward pass**: completed, `end_x − start_x ≥ 5`.
- **Progressive pass / carry**: reduces distance-to-goal by `≥ 10`.
- **Long pass**: `pass_length ≥ 30`.
- **Final-third / box entry**: start outside the zone, end inside (so a pass that
  begins in the final third is not counted as an entry).
- **Open-play passes**: pass volume and completion exclude set pieces
  (throw-ins, corners, free kicks, goal kicks, kick-offs). A completed pass has a
  null `pass_outcome`.
- **Non-penalty shooting**: shots with `shot_type = 'Penalty'` are excluded from
  `shots_p90`, `npxg_p90` and `np_goals`.
- **Key pass / xG assisted**: a key pass has `pass_shot_assist = true`; xG assisted
  credits the assisting passer with the xG of the shot that its `shot_key_pass_id`
  points back to.
- **Turnovers** = miscontrols + dispossessions + incomplete passes made under pressure.
- **Aerials**: an aerial win is `aerial_won = true` (a flag nested under each event
  type — pass/clearance/shot/… — coalesced during flattening); an aerial loss is a
  `Duel` of type `Aerial Lost`.
- **Possession adjustment**: `padj_factor = 0.5 / (1 − possession_share)`, applied to
  `pressures/tackles/interceptions` per-90. `possession_share` is the team's share of
  all passes across its matches in the season. A team with 50% possession is unchanged;
  low-possession teams are scaled down, high-possession teams up, so defensive volume
  is comparable across styles.
- **Percentiles** are ranked **within role group** over qualifying players
  (≥ 600 minutes). Winsorization and z-score standardization are applied later, in the
  Phase 4 similarity feature matrix, not here.

## Player metrics

### Passing & security (all roles)
| Metric | Definition | Denominator |
|---|---|---|
| pass_attempts_p90 | Attempted passes | per 90 |
| pass_completion_pct | Completed / attempted passes | attempted passes (min 100) |
| forward_pass_share | Passes moving the ball ≥ 5 m toward opponent goal / attempted | attempted passes |
| long_pass_share | Passes ≥ 30 m / attempted | attempted passes |
| turnovers_p90 | Miscontrols + dispossessions + failed passes under pressure | per 90 |

### Progression (FB, CM, AM/W)
| Metric | Definition | Denominator |
|---|---|---|
| progressive_passes_p90 | Completed passes advancing the ball ≥ 10 m toward goal (≥ 5 m inside final third) | per 90 |
| progressive_carries_p90 | Carries advancing the ball ≥ 10 m toward goal (≥ 5 m inside final third) | per 90 |
| final_third_entries_p90 | Completed passes or carries entering the final third | per 90 |
| box_entries_p90 | Completed passes or carries entering the penalty area | per 90 |

### Creation (CM, AM/W, ST)
| Metric | Definition | Denominator |
|---|---|---|
| key_passes_p90 | Passes leading directly to a shot | per 90 |
| xg_assisted_p90 | Sum of xG of shots assisted | per 90 |
| crosses_into_box_p90 | Completed crosses received inside the penalty area | per 90 |
| through_balls_p90 | Completed through balls | per 90 |

### Shooting (AM/W, ST)
| Metric | Definition | Denominator |
|---|---|---|
| shots_p90 | Non-penalty shots | per 90 |
| npxg_p90 | Non-penalty xG (StatsBomb model) | per 90 |
| box_shot_share | Shots from inside the penalty area / all shots | shots (min 20) |
| npxg_per_shot | npxG / non-penalty shots — shot quality | shots (min 20) |
| goals_minus_xg | Non-penalty goals − npxG (display only, never a similarity feature) | season total |

### Defending (CB, FB, CM)
| Metric | Definition | Denominator |
|---|---|---|
| padj_pressures_p90 | Pressure events, possession-adjusted | per 90 |
| padj_tackles_p90 | Tackles (duel type), possession-adjusted | per 90 |
| padj_interceptions_p90 | Interceptions, possession-adjusted | per 90 |
| blocks_p90 | Blocked shots/passes | per 90 |
| recoveries_p90 | Ball recoveries | per 90 |
| def_action_height | Mean x-coordinate of defensive actions (0–120 scale) | — |

### Duels & aerial play (role-dependent)
| Metric | Definition | Denominator |
|---|---|---|
| dribble_success_pct | Completed / attempted dribbles | dribbles (min 20) |
| aerial_involvement_p90 | Aerial duels contested | per 90 |
| aerial_win_pct | Aerial duels won / contested | aerials (min 20) |

### Off-ball context (all roles)
| Metric | Definition | Denominator |
|---|---|---|
| touches_att_third_share | Touches in attacking third / all touches | touches |
| receptions_final_third_p90 | Ball receipts in final third | per 90 |
| pressured_actions_share | Actions under pressure / all actions | actions |

## Team style metrics (Module B)

| Metric | Definition |
|---|---|
| possession_share | Team passes / all passes in match |
| passes_per_possession | Completed passes per possession sequence |
| directness | Forward distance per pass / total pass distance |
| field_tilt | Team final-third passes / both teams' final-third passes |
| high_press_intensity | Pressures in attacking 40% of pitch, per opponent pass |
| def_action_height_team | Mean x of team defensive actions |
| counterattack_share | Shots within 15 s of regain / all shots |
| box_shot_share_team | Shots inside box / all shots |
| cross_share | Crosses / final-third entries |
| turnovers_p90_team | Possessions lost in own half per 90 |

Team features are built per match first, then aggregated to season mean **and standard
deviation** (variability is a style signal, not noise).

## Explicit exclusions

- No match-result, scoreline or betting metrics.
- No transfer-value metrics.
- `goals_minus_xg` is display-only: finishing over/under-performance is noisy in one
  season and must not drive similarity.
