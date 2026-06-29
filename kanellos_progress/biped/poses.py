"""Nominal standing postures.

The Menagerie 'stand' keyframe has locked (straight) knees, which gives the CoM
almost no vertical/lateral control authority and makes the leg Jacobian nearly
singular.  Every controller here instead starts from a *bent-knee* crouch -- the
standard walking stance -- with the base height solved so the soles rest on the
terrain.
"""
from __future__ import annotations

import mujoco
import numpy as np


def crouch_qpos(robot, terrain, scale=1.0, crouch=None):
    """Return a qpos for a symmetric bent-knee stance, feet flat on the terrain.

    Arms/torso keep the keyframe pose; the legs are bent according to the
    robot's ``leg_crouch`` spec (leg-group index -> angle, optionally ``scale``d);
    the base height is solved by dropping the body until the lowest sole touches
    the ground.  ``crouch`` overrides the config spec when given.
    """
    m, d = robot.m, robot.d
    qpos = robot.keyframe_qpos()
    spec = crouch if crouch is not None else robot.config.leg_crouch

    for side in ("left", "right"):
        grp = robot.groups[f"leg_{side}"]
        for idx, val in spec.items():
            qpos[grp.qadr[idx]] = val * scale

    # Solve base height so the lowest foot corner sits on the terrain surface.
    qpos[3:7] = [1, 0, 0, 0]  # upright base
    d.qpos[:] = qpos
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    corners = np.vstack([robot.foot_corner_points("left"),
                         robot.foot_corner_points("right")])
    # terrain height under each corner; lift base so the deepest one is flush
    gaps = corners[:, 2] - np.array([terrain.height(c[0], c[1]) for c in corners])
    qpos[2] -= gaps.min()
    return qpos


def reset_to_crouch(robot, terrain, **kw):
    qpos = crouch_qpos(robot, terrain, **kw)
    robot.d.qpos[:] = qpos
    robot.d.qvel[:] = 0
    mujoco.mj_forward(robot.m, robot.d)
    return qpos
