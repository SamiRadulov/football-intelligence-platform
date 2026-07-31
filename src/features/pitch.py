"""Shared pitch geometry: zones, distances and coordinate conventions.

StatsBomb coordinates are 120 x 80 and **always from the acting team's
perspective**, attacking towards x = 120. That convention matters when relating
two teams' events to each other: an action by team A at x is at pitch position
120 - x from team B's point of view (see `flip_x`).

Zone layout in the attacking half (y-bands use real pitch markings, since the
penalty area spans y 18-62):

    y  0 ---- 18 -------- 30 ---------- 50 -------- 62 ---- 80
       |  wide  | halfspace |  Zone 14   | halfspace |  wide |

Player and team features both import from here so the definitions cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_LENGTH, PITCH_WIDTH = 120.0, 80.0
GOAL_X, GOAL_Y = 120.0, 40.0

BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102.0, 18.0, 62.0
FINAL_THIRD_X = 80.0
ATT_HALF_X = 60.0

ZONE14_X_MIN, ZONE14_X_MAX = 78.0, 102.0
ZONE14_Y_MIN, ZONE14_Y_MAX = 30.0, 50.0
HALFSPACE_X_MIN, HALFSPACE_X_MAX = 60.0, 102.0

# Pressing zone: the attacking 60% of the pitch, used for PPDA.
PRESSING_ZONE_X = 48.0
# High regains: won back inside the opponent's defensive third-ish area.
HIGH_REGAIN_X = 72.0


def flip_x(x: pd.Series) -> pd.Series:
    """Convert an x-coordinate to the opposing team's frame of reference."""
    return PITCH_LENGTH - x


def dist_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    """Straight-line distance to the centre of the opponent's goal."""
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def in_box(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x >= BOX_X_MIN) & (y >= BOX_Y_MIN) & (y <= BOX_Y_MAX)


def in_final_third(x: pd.Series) -> pd.Series:
    return x >= FINAL_THIRD_X


def in_zone14(x: pd.Series, y: pd.Series) -> pd.Series:
    """Central pocket just outside the penalty area (the playmaker's zone)."""
    return (
        (x >= ZONE14_X_MIN) & (x < ZONE14_X_MAX)
        & (y >= ZONE14_Y_MIN) & (y <= ZONE14_Y_MAX)
    )


def in_halfspace(x: pd.Series, y: pd.Series) -> pd.Series:
    """Either channel between the penalty-area edge and the central strip.

    The two channels are combined rather than split left/right so the feature is
    mirror-invariant.
    """
    band = ((y >= BOX_Y_MIN) & (y < ZONE14_Y_MIN)) | (
        (y > ZONE14_Y_MAX) & (y <= BOX_Y_MAX)
    )
    return (x >= HALFSPACE_X_MIN) & (x < HALFSPACE_X_MAX) & band


def in_wide_channel(x: pd.Series, y: pd.Series) -> pd.Series:
    """Outside the width of the penalty area, in the attacking half."""
    return (x >= ATT_HALF_X) & ((y < BOX_Y_MIN) | (y > BOX_Y_MAX))
