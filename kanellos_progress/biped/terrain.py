"""Terrain definitions for the walking benchmark.

A ``Terrain`` does two jobs:

1. ``apply(spec)`` mutates the MuJoCo model at load time (tilt the ground plane,
   add stair boxes, ...).
2. ``height(x, y)`` / ``normal(x, y)`` give the controller terrain awareness so
   footsteps land flat and friction cones are expressed in the *surface* frame.

The incline case is the important one: a world-frame friction cone on a slope
lets the QP command physically impossible forces and the robot slides, so the
controller must rotate each foot's cone to ``normal(x, y)``.
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

import mujoco
import numpy as np


class Terrain:
    name: str = "terrain"

    def apply(self, spec) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def height(self, x: float, y: float) -> float:
        return 0.0

    def normal(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def footstep_x(self, stance_x: float, swing_x: float,
                   nominal_step: float) -> float:
        """Forward x for the next footstep.

        Default: advance the *stance* foot by the nominal step length (the swing
        foot leap-frogs ahead of the support).  Stairs override this to land each
        foot on a tread centre so the sole never overhangs a riser.
        """
        return stance_x + nominal_step


@dataclasses.dataclass
class Flat(Terrain):
    name: str = "flat"
    # upright cylindrical obstacles: tuple of (x, y, radius)
    obstacles: tuple = ()
    # flat visual markers (e.g. the goal): tuple of (x, y)
    markers: tuple = ()

    def apply(self, spec) -> None:
        # The Menagerie scene already provides a flat floor plane; add obstacles.
        for i, (ox, oy, rr) in enumerate(self.obstacles):
            g = spec.worldbody.add_geom()
            g.name = f"obstacle_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [rr, 0.6, 0.0]          # radius, half-height
            g.pos = [ox, oy, 0.6]
            g.rgba = [0.85, 0.35, 0.2, 1.0]
            g.friction = [0.8, 0.005, 0.0001]
        for i, (mx, my) in enumerate(self.markers):
            g = spec.worldbody.add_geom()
            g.name = f"marker_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [0.16, 0.005, 0.0]      # flat green disk on the floor
            g.pos = [mx, my, 0.006]
            g.rgba = [0.2, 0.8, 0.3, 0.9]
            g.contype = 0
            g.conaffinity = 0                # visual only, no collision
        return None


@dataclasses.dataclass
class Incline(Terrain):
    """Constant slope, uphill toward +x. ``angle`` in radians."""
    angle: float = np.deg2rad(8.0)
    name: str = "incline"

    def _floor_geom(self, spec):
        for g in spec.worldbody.geoms:
            if g.name == "floor":
                return g
        return None

    def apply(self, spec) -> None:
        g = self._floor_geom(spec)
        if g is None:
            raise RuntimeError("no 'floor' geom to tilt")
        a = self.angle
        # Rotation about +y by -angle: surface normal -> [-sin a, 0, cos a],
        # so the plane is z = tan(a) * x (uphill toward +x).
        g.quat = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]

    def height(self, x: float, y: float) -> float:
        return float(np.tan(self.angle) * x)

    def normal(self, x: float, y: float) -> np.ndarray:
        a = self.angle
        return np.array([-np.sin(a), 0.0, np.cos(a)])


@dataclasses.dataclass
class Stairs(Terrain):
    """Flat treads stepping up toward +x. Each step ``rise`` high, ``run`` deep."""
    rise: float = 0.05
    run: float = 0.30
    n_steps: int = 6
    x0: float = 0.5  # x where the first riser begins
    width: float = 1.5
    name: str = "stairs"

    top_run: float = 1.2   # depth of the flat landing past the last riser

    def apply(self, spec) -> None:
        for i in range(self.n_steps):
            h = (i + 1) * self.rise
            cx = self.x0 + self.run * (i + 0.5)
            g = spec.worldbody.add_geom()
            g.name = f"stair_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            # half-sizes; box top sits at height h, extends downward to the floor
            g.size = [self.run / 2, self.width / 2, h / 2]
            g.pos = [cx, 0.0, h / 2]
            g.rgba = [0.45, 0.45, 0.5, 1.0]
            g.friction = [0.8, 0.005, 0.0001]
        # top landing: a real platform past the last riser so the robot has
        # somewhere to stand (height() promises a flat top -- give it geometry)
        h = self.n_steps * self.rise
        g = spec.worldbody.add_geom()
        g.name = "stair_top"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.top_run / 2, self.width / 2, h / 2]
        g.pos = [self.x0 + self.n_steps * self.run + self.top_run / 2, 0.0, h / 2]
        g.rgba = [0.4, 0.42, 0.48, 1.0]
        g.friction = [0.8, 0.005, 0.0001]

    def height(self, x: float, y: float) -> float:
        if x < self.x0:
            return 0.0
        idx = int((x - self.x0) // self.run)
        if idx >= self.n_steps:
            return self.n_steps * self.rise
        return (idx + 1) * self.rise

    def normal(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    # ---- footstep planning -------------------------------------------------
    def tread_index(self, x: float) -> int:
        """Index of the tread surface under x (-1 = ground before the stairs)."""
        if x < self.x0:
            return -1
        return int((x - self.x0) // self.run)

    def tread_center_x(self, i: int) -> float:
        """x of the centre of tread ``i`` (i = 0 is the first step up)."""
        return self.x0 + (i + 0.5) * self.run

    def footstep_x(self, stance_x: float, swing_x: float,
                   nominal_step: float) -> float:
        """Land each foot on a tread centre (feet-together-per-tread gait).

        The target is computed from the *swing* foot's own tread so the two feet
        climb one tread at a time and meet on each step -- steps stay ~one tread
        deep instead of doubling.  On the ground the robot walks normally and
        snaps onto tread-0 when a step would reach the stairs; on the top
        platform it resumes flat steps.
        """
        sj = self.tread_index(swing_x)
        if sj < 0:                                  # swing foot on the ground
            if swing_x + nominal_step >= self.x0 - 0.05:
                return self.tread_center_x(0)       # snap onto the first tread
            return swing_x + nominal_step
        if sj >= self.n_steps - 1:                  # on/over the top: walk flat
            return swing_x + nominal_step
        return self.tread_center_x(sj + 1)          # climb to the next tread


def make(name: str, **kw) -> Terrain:
    return {"flat": Flat, "incline": Incline, "stairs": Stairs}[name](**kw)
