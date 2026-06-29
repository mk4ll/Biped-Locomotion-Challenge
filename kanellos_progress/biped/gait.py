"""Per-terrain gait profiles.

One set of gait parameters does not suit every surface: stairs need a slow,
careful weight transfer and high foot clearance over the riser; inclines want
shorter, higher steps so the swing foot clears the rising ground and the CoM is
not over-committed; flat ground can stride out.  ``terrain_gait`` returns the
finetuned knobs for a given terrain (and robot), which the demo/eval scripts feed
to the :class:`WalkingController`.

Knobs:
  step_len         forward step length (m); on stairs this equals the tread depth
  t_ss, t_ds       single- / double-support durations (s)
  t_ds0            longer initial weight-shift (s)
  swing_clearance  swing-foot apex clearance (m)
"""
from __future__ import annotations

import numpy as np


def terrain_gait(terrain, robot_name: str = "g1") -> dict:
    name = getattr(terrain, "name", "flat")
    tall = robot_name == "talos"

    if name == "incline":
        ang = float(np.degrees(terrain.angle))
        # shorter strides + more clearance as it steepens; a touch slower
        step = max(0.10, 0.15 - 0.0035 * ang)
        return dict(step_len=round(step, 3),
                    t_ss=0.72, t_ds=0.22, t_ds0=0.85,
                    swing_clearance=0.07 + 0.0015 * ang)

    if name == "stairs":
        # grid-aligned: stride == tread depth; clear the riser; go slow and
        # spend longer in double support to shift weight up each step
        return dict(step_len=float(terrain.run),
                    t_ss=0.85, t_ds=0.28, t_ds0=0.95,
                    swing_clearance=round(float(terrain.rise) + 0.05, 3))

    # flat
    return dict(step_len=0.16 if tall else 0.15,
                t_ss=0.7, t_ds=0.2, t_ds0=0.8, swing_clearance=0.06)
