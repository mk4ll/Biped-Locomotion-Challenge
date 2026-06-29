"""Terrain abstraction for terrain-aware locomotion (merged).

Adapted into the offline/online/WBC pipeline of this project. A Terrain does two
jobs (the second is what makes incline/stairs walking work, not bolted on):

1. ``apply(spec)``  -- mutate the MuJoCo model at load time (tilt the floor, add
   stair boxes / obstacles) via mjSpec.
2. ``height(x,y)`` / ``normal(x,y)`` / ``footstep_x(...)`` -- give the planner &
   controller terrain awareness so footsteps land flat ON the surface and the WBC
   friction cones are expressed in the SURFACE frame (not world-z).

Credit: terrain-aware design adapted from Kanellos' `biped/terrain.py`; folded
into this repo's planning layer.
"""
from __future__ import annotations
import dataclasses
import mujoco
import numpy as np


class Terrain:
    name = "terrain"

    def apply(self, spec) -> None:
        raise NotImplementedError

    def height(self, x: float, y: float) -> float:
        return 0.0

    def normal(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def footstep_x(self, stance_x: float, swing_x: float, nominal_step: float) -> float:
        """Forward x of the next footstep (swing leap-frogs the stance foot)."""
        return stance_x + nominal_step

    def surface_R(self, x: float, y: float) -> np.ndarray:
        """World<-surface rotation whose 3rd column is the surface normal."""
        n = self.normal(x, y)
        n = n / (np.linalg.norm(n) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(np.array([0, 1.0, 0]), n)
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(ref, n)
        t1 /= np.linalg.norm(t1) + 1e-12
        t2 = np.cross(n, t1)
        return np.column_stack([t1, t2, n])


@dataclasses.dataclass
class Flat(Terrain):
    name: str = "flat"
    obstacles: tuple = ()        # (x, y, radius) cylinders for navigation
    markers: tuple = ()          # (x, y) flat visual goal disks

    def apply(self, spec) -> None:
        for i, (ox, oy, rr) in enumerate(self.obstacles):
            self._add_table(spec, i, ox, oy, rr)
        for i, (mx, my) in enumerate(self.markers):
            g = spec.worldbody.add_geom()
            g.name = f"marker_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [0.16, 0.005, 0.0]
            g.pos = [mx, my, 0.006]
            g.rgba = [0.2, 0.8, 0.3, 0.9]
            g.contype = 0
            g.conaffinity = 0

    @staticmethod
    def _add_table(spec, i, ox, oy, rr, h=0.50):
        """A real-looking table: round top + 4 legs (collidable footprint = rr)."""
        top = spec.worldbody.add_geom()
        top.name = f"table_{i}_top"
        top.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        top.size = [rr, 0.022, 0.0]
        top.pos = [ox, oy, h]
        top.rgba = [0.55, 0.36, 0.20, 1.0]               # wood
        top.friction = [0.8, 0.005, 0.0001]
        d = rr * 0.62
        for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            leg = spec.worldbody.add_geom()
            leg.name = f"table_{i}_leg{k}"
            leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            leg.size = [0.025, h / 2, 0.0]
            leg.pos = [ox + sx * d, oy + sy * d, h / 2]
            leg.rgba = [0.40, 0.26, 0.14, 1.0]
            leg.friction = [0.8, 0.005, 0.0001]


@dataclasses.dataclass
class Incline(Terrain):
    """Constant slope, uphill toward +x. ``angle`` in radians."""
    angle: float = np.deg2rad(8.0)
    name: str = "incline"

    def _floor(self, spec):
        for g in spec.worldbody.geoms:
            if g.name == "floor":
                return g
        return None

    def apply(self, spec) -> None:
        g = self._floor(spec)
        if g is None:
            raise RuntimeError("no 'floor' geom to tilt")
        a = self.angle
        # rotate about +y by -angle => normal [-sin a,0,cos a], plane z = tan(a)*x
        g.quat = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]
        # Steep inclines need higher friction to prevent slipping;
        # scale linearly from 0.8 at 0° to 1.5 at 30°+.
        g.friction = [min(0.8 + 2.33 * a, 1.5), 0.005, 0.0001]

    def height(self, x, y):
        return float(np.tan(self.angle) * x)

    def normal(self, x, y):
        a = self.angle
        return np.array([-np.sin(a), 0.0, np.cos(a)])


@dataclasses.dataclass
class Stairs(Terrain):
    """Flat treads stepping up toward +x."""
    rise: float = 0.05
    run: float = 0.30
    n_steps: int = 6
    x0: float = 0.5
    width: float = 1.5
    top_run: float = 1.2
    name: str = "stairs"

    def apply(self, spec) -> None:
        for i in range(self.n_steps):
            h = (i + 1) * self.rise
            cx = self.x0 + self.run * (i + 0.5)
            g = spec.worldbody.add_geom()
            g.name = f"stair_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [self.run / 2, self.width / 2, h / 2]
            g.pos = [cx, 0.0, h / 2]
            g.rgba = [0.45, 0.45, 0.5, 1.0]
            g.friction = [0.8, 0.005, 0.0001]
        h = self.n_steps * self.rise
        g = spec.worldbody.add_geom()
        g.name = "stair_top"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.top_run / 2, self.width / 2, h / 2]
        g.pos = [self.x0 + self.n_steps * self.run + self.top_run / 2, 0.0, h / 2]
        g.rgba = [0.4, 0.42, 0.48, 1.0]
        g.friction = [0.8, 0.005, 0.0001]

    def height(self, x, y):
        if x < self.x0:
            return 0.0
        idx = int((x - self.x0) // self.run)
        if idx >= self.n_steps:
            return self.n_steps * self.rise
        return (idx + 1) * self.rise

    def normal(self, x, y):
        return np.array([0.0, 0.0, 1.0])

    def tread_index(self, x):
        if x < self.x0:
            return -1
        return int((x - self.x0) // self.run)

    def tread_center_x(self, i):
        return self.x0 + (i + 0.5) * self.run

    def footstep_x(self, stance_x, swing_x, nominal_step):
        sj = self.tread_index(swing_x)
        if sj < 0:
            if swing_x + nominal_step >= self.x0 - 0.05:
                return self.tread_center_x(0)
            return swing_x + nominal_step
        if sj >= self.n_steps - 1:
            return swing_x + nominal_step
        return self.tread_center_x(sj + 1)


def make_terrain(name: str, **kw) -> Terrain:
    return {"flat": Flat, "incline": Incline, "stairs": Stairs}[name](**kw)
