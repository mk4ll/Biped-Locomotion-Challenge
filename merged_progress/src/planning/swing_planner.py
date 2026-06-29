"""Swing-foot trajectory: smooth lift-and-place over a single-support phase.

Horizontal (x,y): smoothstep (zero velocity at both ends).
Vertical (z):     (1 - cos) bump to 'apex' (zero velocity at lift-off, apex,
                  and touchdown), returning to the target ground height.
"""
import numpy as np


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3, 6 * s - 6 * s**2   # value, d/ds


def swing_trajectory(s, p0, p1, apex, duration):
    """s in [0,1] (phase progress). Returns (pos[3], vel[3]) in world frame."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    f, df = _smoothstep(s)
    pos = np.empty(3); vel = np.empty(3)
    # horizontal
    pos[:2] = p0[:2] + f * (p1[:2] - p0[:2])
    vel[:2] = df * (p1[:2] - p0[:2]) / max(duration, 1e-6)
    # vertical: base interpolates p0.z->p1.z, plus (1-cos) apex bump
    base_z = p0[2] + f * (p1[2] - p0[2])
    dbase_z = df * (p1[2] - p0[2]) / max(duration, 1e-6)
    bump = 0.5 * apex * (1 - np.cos(2 * np.pi * s))
    dbump = 0.5 * apex * 2 * np.pi * np.sin(2 * np.pi * s) / max(duration, 1e-6)
    pos[2] = base_z + bump
    vel[2] = dbase_z + dbump
    return pos, vel
