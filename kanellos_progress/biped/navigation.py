"""Go-to-goal navigation with obstacle avoidance -> velocity command.

A thin planning layer on top of the omnidirectional walker.  Given the robot's
world pose and a set of circular obstacles, an artificial-potential field gives a
desired world-frame travel direction (attraction to the goal + repulsion from
obstacles); that direction is turned into a body-frame ``(vx, vy, vyaw)`` command
with a *turn-to-face* policy -- the robot yaws toward where it wants to go and
walks forward, the way a person does, rather than crab-walking sideways.

    nav = Navigator(goal=(3.0, 0.5), obstacles=[(1.2, 0.0, 0.25)])
    (vx, vy, vyaw), done = nav.command(pos_xy, heading)
"""
from __future__ import annotations

import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Navigator:
    def __init__(self, goal, obstacles=None, v_max=0.22, vyaw_max=0.12,
                 goal_tol=0.2, slow_radius=0.6, influence=0.55,
                 k_repel=0.5, k_swirl=1.2, k_yaw=1.5, robot_radius=0.20):
        self.goal = np.asarray(goal, float)
        # obstacles as (x, y, radius)
        self.obstacles = [np.asarray(o, float) for o in (obstacles or [])]
        self.v_max = v_max
        self.vyaw_max = vyaw_max
        self.goal_tol = goal_tol
        self.slow_radius = slow_radius
        self.influence = influence
        self.k_repel = k_repel
        self.k_swirl = k_swirl
        self.k_yaw = k_yaw
        self.robot_radius = robot_radius

    def field(self, pos):
        """World-frame desired travel direction (unit-ish) + distance to goal."""
        pos = np.asarray(pos, float)
        to_goal = self.goal - pos
        dist = float(np.linalg.norm(to_goal))
        attract = to_goal / (dist + 1e-9)
        repel = np.zeros(2)
        for o in self.obstacles:
            d = pos - o[:2]
            dd = float(np.linalg.norm(d))
            dhat = d / (dd + 1e-9)
            clear = dd - o[2] - self.robot_radius        # surface clearance
            if clear < self.influence:
                # push away, stronger the closer we are (clamped)
                mag = self.k_repel * (1.0 / max(clear, 0.05) - 1.0 / self.influence)
                repel += mag * dhat
                # tangential swirl so we circle *around* the obstacle instead of
                # stalling head-on (breaks the potential-field local minimum when
                # the obstacle sits between us and the goal); pick the goal side
                tang = np.array([-dhat[1], dhat[0]])
                if np.dot(tang, attract) < 0:
                    tang = -tang
                repel += self.k_swirl * mag * tang
        v = attract + repel
        n = np.linalg.norm(v)
        return (v / n if n > 1e-6 else attract), dist

    def command(self, pos, heading):
        """Return ((vx, vy, vyaw), done) -- a body-frame velocity command.

        Navigation is done by *omnidirectional translation* (the robot keeps its
        heading and walks forward/sideways toward the field direction), because
        translation is far more stable than turning for this gait.  Heading is
        only nudged when it drifts well off the travel direction.
        """
        direction, dist = self.field(pos)
        if dist < self.goal_tol:
            return (0.0, 0.0, 0.0), True
        # desired travel direction expressed in the body frame
        c, s = np.cos(-heading), np.sin(-heading)
        vb = np.array([c * direction[0] - s * direction[1],
                       s * direction[0] + c * direction[1]])
        speed = self.v_max * min(1.0, dist / self.slow_radius)
        vx = speed * vb[0]
        vy = float(np.clip(speed * vb[1], -0.6 * self.v_max, 0.6 * self.v_max))
        # gentle heading correction only past a wide deadband (avoid fragile turns)
        herr = _wrap(np.arctan2(direction[1], direction[0]) - heading)
        vyaw = (float(np.clip(self.k_yaw * herr, -self.vyaw_max, self.vyaw_max))
                if abs(herr) > np.radians(40) else 0.0)
        return (vx, vy, vyaw), False
