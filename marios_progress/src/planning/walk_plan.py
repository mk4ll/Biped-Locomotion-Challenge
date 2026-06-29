"""Orchestrates footsteps + FSM + DCM CoM + swing into a single reference(t).

This is the planner output consumed by the WBC in Stage 4. It is completely
independent of the controller (DESCRIPTION.md s2): it only needs the initial
foot positions and CoM height.
"""
import numpy as np

from src.planning.footstep_planner import FootstepPlanner
from src.planning.com_planner import DCMPlanner
from src.planning.swing_planner import swing_trajectory
from src.planning.fsm import phase_at, total_duration


class WalkPlan:
    def __init__(self, params, init_left, init_right, com0, com_height=None,
                 gravity=9.81, incline_rad=0.0):
        self.params = params
        self.z = com_height if com_height is not None else com0[2]
        self.tan_slope = np.tan(incline_rad)
        # CoM world-z follows the slope: z(x) = x*tan_slope + z0_com
        self.z0_com = com0[2] - com0[0] * self.tan_slope
        self.fp = FootstepPlanner(params)
        self.footsteps, self.timeline = self.fp.plan(init_left, init_right,
                                                     tan_slope=self.tan_slope)
        # DCM uses the vertical CoM height above the (gravity) frame; mild slope ok.
        self.dcm = DCMPlanner(params, self.z, gravity)
        self.traj = self.dcm.generate(self.timeline, np.asarray(com0)[:2])
        self.dt = self.traj["t"][1] - self.traj["t"][0]
        self.duration = total_duration(self.timeline)

    # -- sampled trajectory access -------------------------------------------
    def _idx(self, t):
        return int(np.clip(round(t / self.dt), 0, len(self.traj["t"]) - 1))

    def reference(self, t):
        """Full reference at time t for the WBC."""
        i = self._idx(t)
        com_x = self.traj["com"][i, 0]
        com_z = com_x * self.tan_slope + self.z0_com           # follow the slope
        com = np.array([com_x, self.traj["com"][i, 1], com_z])
        com_vx = self.traj["com_vel"][i, 0]
        com_vel = np.array([com_vx, self.traj["com_vel"][i, 1],
                            com_vx * self.tan_slope])           # vertical rate up-slope
        zmp = self.traj["zmp"][i].copy()
        dcm = self.traj["dcm"][i].copy()
        ph, s = phase_at(self.timeline, t)
        ref = {"com": com, "com_vel": com_vel, "zmp": zmp, "dcm": dcm,
               "omega": self.traj["omega"], "progress": s,
               "support": ph["support"], "phase": ph["type"],
               "swing": ph["swing"], "swing_pos": None, "swing_vel": None}
        if ph["type"] == "SS":
            pos, vel = swing_trajectory(s, ph["swing_from"], ph["swing_to"],
                                        self.fp.swing_apex, ph["dur"])
            ref["swing_pos"] = pos
            ref["swing_vel"] = vel
        return ref
