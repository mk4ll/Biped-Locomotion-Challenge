"""Footstep planner: alternating steps + gait timeline.

Produces a list of footsteps (alternating left/right lanes, advancing in +x)
and the phase timeline consumed by the FSM / DCM / swing planners.
A kinematic clamp limits the per-step advance to a safe maximum.
"""
import numpy as np


class FootstepPlanner:
    def __init__(self, params):
        g = params["gait"]
        self.n_steps = g["n_steps"]
        self.step_length = g["step_length"]
        self.swing_apex = g["swing_apex"]
        self.t_ss = g["t_ss"]
        self.t_ds = g["t_ds"]
        self.t_ds_init = g["t_ds_init"]
        self.t_ds_final = g["t_ds_final"]
        self.first_swing = g["first_swing"]
        self.max_step = 1.5 * self.step_length     # kinematic clamp

    def plan(self, init_left, init_right, tan_slope=0.0):
        """init_left/right: xyz of the two foot sites. tan_slope: ground slope in +x.
        Foot landing heights follow the slope: z(x) = x*tan_slope + z0.
        Returns (footsteps, timeline)."""
        L = np.array(init_left, float)
        R = np.array(init_right, float)
        # slope reference so the initial feet lie on the surface
        z0 = 0.5 * (L[2] + R[2]) - 0.5 * (L[0] + R[0]) * tan_slope

        def slope_z(x):
            return x * tan_slope + z0
        feet = {"left": L.copy(), "right": R.copy()}

        footsteps = [
            {"foot": "left", "pos": L.copy()},
            {"foot": "right", "pos": R.copy()},
        ]
        timeline = []
        t = 0.0

        def add(ph):
            ph["t0"] = t
            ph["dur"] = ph["dur"]
            ph["t1"] = t + ph["dur"]
            timeline.append(ph)
            return ph["t1"]

        # initial double support
        t = add({"type": "DS", "dur": self.t_ds_init, "support": "double",
                 "swing": None, "zmp_from": 0.5 * (L[:2] + R[:2]),
                 "zmp_to": 0.5 * (L[:2] + R[:2])})

        swing = self.first_swing
        for k in range(self.n_steps):
            support = "left" if swing == "right" else "right"
            advance = min(self.step_length, self.max_step)
            target = feet[swing].copy()
            target[0] = feet[support][0] + advance         # step past the stance foot
            target[2] = slope_z(target[0])                 # land on the slope
            # SS: ZMP sits at the support foot.
            t = add({"type": "SS", "dur": self.t_ss, "support": support,
                     "swing": swing, "swing_from": feet[swing].copy(),
                     "swing_to": target.copy(),
                     "zmp_from": feet[support][:2].copy(),
                     "zmp_to": feet[support][:2].copy()})
            feet[swing] = target
            footsteps.append({"foot": swing, "pos": target.copy()})
            # DS: ZMP ramps from old support foot to the new stance (the foot just placed).
            next_support = swing       # the foot just placed becomes next support
            t = add({"type": "DS", "dur": self.t_ds, "support": "double",
                     "swing": None, "zmp_from": feet[support][:2].copy(),
                     "zmp_to": feet[next_support][:2].copy()})
            swing = "left" if swing == "right" else "right"

        # final settle: ZMP to mid-feet
        mid = 0.5 * (feet["left"][:2] + feet["right"][:2])
        t = add({"type": "DS", "dur": self.t_ds_final, "support": "double",
                 "swing": None, "zmp_from": timeline[-1]["zmp_to"].copy(),
                 "zmp_to": mid})
        return footsteps, timeline
