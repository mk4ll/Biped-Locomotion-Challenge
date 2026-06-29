"""Gait finite-state machine: phases + contact schedule.

Phase sequence:  DS_init -> SS -> DS -> SS -> ... -> DS_final
A phase is a dict:
  {'type': 'DS'|'SS', 't0', 't1', 'dur',
   'support': 'double'|'left'|'right',
   'swing': None|'left'|'right',
   'swing_from': xyz, 'swing_to': xyz,         # SS only
   'zmp_from': xy, 'zmp_to': xy}               # ZMP reference endpoints
The ZMP ramps linearly from zmp_from->zmp_to across the phase (constant in SS).
"""
import numpy as np


def phase_at(timeline, t):
    """Return (phase, s) where s in [0,1] is progress within the phase."""
    for ph in timeline:
        if ph["t0"] <= t < ph["t1"]:
            s = (t - ph["t0"]) / ph["dur"] if ph["dur"] > 0 else 0.0
            return ph, s
    ph = timeline[-1]
    return ph, 1.0


def support_mode(phase):
    """Map a phase to the WBC ContactSet mode."""
    return phase["support"]


def total_duration(timeline):
    return timeline[-1]["t1"]
