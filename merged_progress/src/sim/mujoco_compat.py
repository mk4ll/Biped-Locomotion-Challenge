"""Compatibility shims for MuJoCo spec API differences across versions.

MuJoCo 3.x changed several MjsJoint / MjsGeom attribute signatures so that
what was a plain scalar now expects a numpy array (or vice-versa).  Docker
environments often ship a newer MuJoCo than the local dev machine.

Usage
-----
    from src.sim.mujoco_compat import spec_set

    spec_set(joint, "damping", 6.0)   # works on any MuJoCo 2.x / 3.x
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Attributes that changed from scalar → 1-D ndarray in MuJoCo 3.x spec API.
# Each entry maps attribute_name → default ndarray shape used as fallback.
# ---------------------------------------------------------------------------
_JOINT_ARRAY_ATTRS = {
    "damping":   (3,),
    "armature":  (1,),
    "stiffness": (1,),
    "springref": (1,),
    "margin":    (1,),
}


def spec_set(obj, attr: str, value):
    """Set a MuJoCo spec object attribute in a cross-version-safe way.

    If the plain assignment raises TypeError (new API expects array), we
    automatically wrap the scalar in a zero-padded numpy array and retry.

    Parameters
    ----------
    obj:   Any MuJoCo spec element (MjsJoint, MjsGeom, MjsBody, …)
    attr:  Attribute name as a string, e.g. "damping"
    value: The value to assign (scalar, list, or ndarray)
    """
    try:
        setattr(obj, attr, value)
    except TypeError:
        # New spec API expects an ndarray for this attribute.
        shape = _JOINT_ARRAY_ATTRS.get(attr, (3,))
        arr = np.zeros(shape, dtype=np.float64)
        v = np.asarray(value, dtype=np.float64).ravel()
        arr[:len(v)] = v
        setattr(obj, attr, arr)
