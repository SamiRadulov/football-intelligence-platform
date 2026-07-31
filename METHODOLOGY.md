# Methodology

How the Football Intelligence Platform turns raw match events into the answers it
shows. Metric definitions live in [docs/METRIC_DICTIONARY.md](docs/METRIC_DICTIONARY.md);
this document covers the modelling choices and what validates them.

Dataset: StatsBomb Open Data, Premier League 2015/16 (380 matches, 1,313,773 events).

## Why this dataset

The platform needs event-level data with coordinates — pressures, carries,
progressive passes, zone occupation. Among freely available sources only StatsBomb
provides that depth; the affordable commercial APIs supply aggregated per-player
statistics and basic match events, which cannot support role-aware spatial features
or the team-style module. The choice was therefore **event depth over recency**: the
platform's value is the method, and the method requires raw touches. A full audit of
every available competition-season is in [docs/data_audit.csv](docs/data_audit.csv).

## Module A — Player Recruitment Similarity

### 1. Role groups

Outfield players are grouped into five roles (CB, FB, CM, AM_W, ST) by mapping
StatsBomb's ~25 granular positions and assigning each player the group they played
the most minutes in. Goalkeepers are excluded. Roles matter because every
subsequent step — standardization, percentiles, candidate filtering — happens
*within* role: 40 passes per 90 means something different for a centre-back than
for a winger.

### 2. Features

Each role has its own feature list and weights, version-controlled in
[artifacts/feature_config.yml](artifacts/feature_config.yml) rather than hard-coded,
so the analytical choices are reviewable and changeable without touching code.

Counts are summed across the season *before* being converted to rates — summing
then dividing once is the correct order. Volume metrics become per-90; rates carry
minimum-attempt guards (a pass completion % needs ≥ 100 attempts or it is null);
defensive volume is possession-adjusted so players in low-possession teams are not
flattered or penalised by their team's style.

Tactical zone features (Zone 14, halfspaces, wide channel) deliberately **combine
left and right** rather than splitting them, so the features are mirror-invariant:
a left-sided and right-sided player in the same role should read as similar.

### 3. Scaling

Within each role group:

1. **Winsorize** at the 1st/99th percentile, so one freak season cannot stretch the
   scale for everyone. Raw values are always what the user sees.
2. **Z-score**, giving every feature comparable spread.
3. **Impute** missing features to the role mean (z = 0) rather than dropping the
   player, while recording per-player **coverage** — the share of features actually
   present. Candidates below 70% coverage are excluded as too sparse to compare fairly.

### 4. Similarity

Weighted cosine similarity on the standardized vectors:

```
sim(a, b) = Σ(w · a · b) / ( √Σ(w · a²) · √Σ(w · b²) )
```

Because the inputs are z-scores centred on the role mean, this measures whether two
players deviate from their role's average in the same directions and proportions —
profile *shape* rather than raw output.

Two deliberate choices:

- **Filtering happens before scoring.** Candidates are restricted to the reference
  player's role and to the minutes/coverage thresholds before any similarity is
  computed. A centre-back must never surface for a winger query because of a
  coincidental statistical shape.
- **Every result is explainable.** The weighted per-feature distance is decomposed
  into the dimensions where the two players are closest and where they differ most,
  with raw values shown alongside, so a result can always be traced back to football.

PCA is used only for two-dimensional visualisation and diagnostics, never as the
similarity score itself.

### 5. Validation

Run with `scripts/validate_similarity.py`. Results on the full dataset
(344 qualifying players, top-10 lists):

| Check | Result | Reading |
|---|---|---|
| Match resampling (keep random 80%, 5 seeds) | **81.2% retained** (sd 0.7%) | Rankings reflect season-long style, not a handful of matches |
| Minutes threshold raised to 900 / 1200 | **98.0% / 98.4% retained** | The answers are not an artefact of the 600-minute cut-off |
| Role weights vs equal weights | 91.7% retained | The weights refine the result; the *feature selection* per role does most of the work |

**How overlap is measured.** Only baseline recommendations that could still be
returned are counted. Resampling and higher minutes thresholds push some players
below the qualifying threshold entirely, and scoring those as "lost
recommendations" would measure candidate-pool shrinkage rather than ranking
stability. An earlier version of this check did exactly that and understated
stability — most severely for the minutes threshold, where the raw figure (76%
at 1200 minutes) was almost entirely pool shrinkage rather than genuine
re-ranking.

The weight-sensitivity number is worth stating plainly: hand-tuned role weights
change roughly one result in twelve. They are a refinement, not the engine. Anyone
reviewing this should judge the role feature *lists* first.

**Face validity.** Spot-checks against the 2015/16 season return the expected
players: N'Golo Kanté → Idrissa Gueye (0.80), Jack Cork, Mousa Dembélé; Mesut Özil →
Kevin De Bruyne (0.85), David Silva, Dimitri Payet. Non-penalty xG leaders are
Agüero, Vardy and Kane; expected-assist leaders are De Bruyne and Özil; aerial
leaders are Kompany among centre-backs and Andy Carroll among strikers.

### 6. Limitations

- **Similarity is descriptive.** It means two players *do similar things*. It is not
  evidence of equal quality, tactical fit for a specific system, or transfer value.
- **One league, one season.** No cross-competition or cross-season normalisation is
  applied, because there is only one competition-season in scope.
- **No age filter.** StatsBomb Open Data does not publish player dates of birth, so
  the age criterion described in the product plan cannot be implemented from this
  source.
- **Team context is not removed.** A player in a dominant possession side will look
  different from the same player in a counter-attacking side. Possession adjustment
  reduces this for defensive volume only.
- **Finishing is excluded from similarity.** Non-penalty goals minus xG is shown for
  context but never used as a feature: one season of finishing variance is noise.
- **Known source-data quirks** are documented in the metric dictionary rather than
  silently corrected.

## Module B — Team Playing-Style Classification

### 1. Style features

28 style features are built **per team per match** across eight dimensions —
possession and circulation, progression and directness, territory, width and
crossing, pressing and defensive height, transitions, chance profile, and risk
and ball security. Definitions are in the metric dictionary.

Two implementation points worth understanding:

- **Opponent-relative metrics.** Possession share, field tilt and PPDA are only
  meaningful with both teams in view, so raw counts are computed per team and then
  joined to the opponent's row in the same match. Possession share and field tilt
  are validated to sum to exactly 1.0 across the two teams of every match.
- **The coordinate flip.** StatsBomb records events from the acting team's
  perspective, always attacking towards x = 120. Relating one team's pressing to
  the other's passing therefore requires mirroring the pitch: PPDA pairs our
  defensive actions at `x ≥ 48` with their passes at `x ≤ 72`.

Season aggregation keeps both the **mean** and the **match-to-match standard
deviation** of every feature. The standard deviation is a style signal in its own
right — forcing every match into a single identity would hide whether a side is
rigid or adaptable.

### 2. Face validity

The features recover the 2015/16 season without being told anything about it:

| Expectation | Result |
|---|---|
| Possession sides | Man Utd (.583), Arsenal (.575), Liverpool, Spurs, Man City top; West Brom (.402) and Sunderland bottom |
| Direct sides | West Brom, Leicester, Sunderland highest forward distance per pass |
| Pressing sides | Spurs (PPDA 1.92) and Liverpool (1.93) most aggressive; Sunderland and West Brom least |
| Leicester's title profile | 100th percentile for shots and final-third entries per 100 passes; 95th for directness, counters, width and crossing; 5th–15th for possession, pass completion and backward passes |
| Set-piece reliance | Crystal Palace and West Brom highest; league mean 25.5% |

### 3. Classification

Not yet implemented (Phase 6). This section will cover PCA, clustering, naming
clusters from their centroids, nearest-team similarity in the original feature
space, and cluster stability analysis.
