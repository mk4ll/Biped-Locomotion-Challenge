"""Swing-foot trajectory generation.

Minimum-jerk (quintic) interpolation in the horizontal plane plus a vertical
clearance bump that returns to the (possibly different) landing height at
touchdown -- so the same code handles flat ground and stair risers.

The bump is ``sin^2(pi u)`` rather than ``sin(pi u)``: its time-derivative is
zero at *both* ends, so the foot lifts off and (critically) touches down with
zero vertical velocity.  A plain ``sin`` bump lands at ~-h*pi/T m/s -- a hard
heel strike every step that shows up as a periodic jerk through the whole body.
"""
from __future__ import annotations

import numpy as np


def _quintic(u: float):
    """Min-jerk scalar s(u) and derivatives ds/du, d2s/du2 for u in [0, 1]."""
    u = min(max(u, 0.0), 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = 30 * u**2 - 60 * u**3 + 30 * u**4
    dds = 60 * u - 180 * u**2 + 120 * u**3
    return s, ds, dds


class SwingTrajectory:
    def __init__(self, p0, p1, duration: float, clearance: float = 0.06):
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.T = float(duration)
        self.h = float(clearance)

    def sample(self, t: float):
        """Return (pos, vel, acc) world-frame 3-vectors at time t in [0, T]."""
        T = self.T
        u = min(max(t / T, 0.0), 1.0)
        s, ds, dds = _quintic(u)
        # horizontal + linear vertical component
        pos = self.p0 + (self.p1 - self.p0) * s
        vel = (self.p1 - self.p0) * ds / T
        acc = (self.p1 - self.p0) * dds / T**2
        # vertical clearance bump: sin^2(pi u) -> zero value AND zero velocity at
        # both ends (soft liftoff and soft touchdown).  du/dt = 1/T.
        w = np.pi
        sp = np.sin(w * u)
        bump = self.h * sp * sp
        dbump = self.h * w / T * np.sin(2 * w * u)
        ddbump = self.h * 2 * w * w / T**2 * np.cos(2 * w * u)
        pos[2] += bump
        vel[2] += dbump
        acc[2] += ddbump
        return pos, vel, acc
