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
                 gravity=9.81, terrain=None, velocity=None, path=None):
        self.params = params
        self.terrain = terrain
        # nominal CoM height ABOVE the local surface (regulated relative to terrain)
        h0 = terrain.height(com0[0], 0.0) if terrain is not None else 0.0
        self.z_above = (com_height if com_height is not None else com0[2]) - h0
        self.fp = FootstepPlanner(params)
        if path is not None:
            self.footsteps, self.timeline = self.fp.plan_path(
                path, init_left, init_right, terrain=terrain)
        elif velocity is not None:
            vx, vy, vyaw = velocity
            self.footsteps, self.timeline = self.fp.plan_velocity(
                init_left, init_right, vx, vy, vyaw, terrain=terrain)
        else:
            self.footsteps, self.timeline = self.fp.plan(init_left, init_right,
                                                         terrain=terrain)
        # DCM uses the vertical CoM height above the surface.
        # On sloped terrain, extract the slope angle so the planner can apply the
        # gravity-projection bias (slope_bias_x = z_com * tan(alpha)).
        slope_angle = getattr(terrain, 'angle', 0.0) if terrain is not None else 0.0
        self.dcm = DCMPlanner(params, self.z_above, gravity, slope_angle=slope_angle)
        self.traj = self.dcm.generate(self.timeline, np.asarray(com0)[:2])
        self.dt = self.traj["t"][1] - self.traj["t"][0]
        self.duration = total_duration(self.timeline)

    def _ground(self, x):
        return self.terrain.height(x, 0.0) if self.terrain is not None else 0.0

    # -- sampled trajectory access -------------------------------------------
    def _idx(self, t):
        return int(np.clip(round(t / self.dt), 0, len(self.traj["t"]) - 1))

    def reference(self, t):
        """Full reference at time t for the WBC."""
        i = self._idx(t)
        com_x = self.traj["com"][i, 0]
        com_z = self._ground(com_x) + self.z_above             # follow the terrain
        com = np.array([com_x, self.traj["com"][i, 1], com_z])
        com_vx = self.traj["com_vel"][i, 0]
        # vertical CoM rate from terrain slope dh/dx (finite diff, robust on stairs)
        dh = (self._ground(com_x + 1e-3) - self._ground(com_x - 1e-3)) / 2e-3
        com_vel = np.array([com_vx, self.traj["com_vel"][i, 1], com_vx * dh])
        zmp = self.traj["zmp"][i].copy()
        dcm = self.traj["dcm"][i].copy()
        ph, s = phase_at(self.timeline, t)
        ref = {"com": com, "com_vel": com_vel, "zmp": zmp, "dcm": dcm,
               "omega": self.traj["omega"], "progress": s,
               "support": ph["support"], "phase": ph["type"],
               "swing": ph["swing"], "swing_pos": None, "swing_vel": None,
               "swing_to": None,
               "heading": ph.get("heading", 0.0)}
        if ph["type"] == "SS":
            pos, vel = swing_trajectory(s, ph["swing_from"], ph["swing_to"],
                                        self.fp.swing_apex, ph["dur"])
            ref["swing_pos"] = pos
            ref["swing_vel"] = vel
            ref["swing_to"] = np.asarray(ph["swing_to"], float)   # nominal landing (xy+z)
        return ref

    def eval(self, t):
        """Return {zmp: xy, xi: xy (reference DCM)} at time t for the MPC."""
        i = self._idx(t)
        return {"zmp": self.traj["zmp"][i].copy(), "xi": self.traj["dcm"][i].copy()}

    def replan_touchdown(self, t, landed_side, actual_foot_pos):
        """Event-driven replanning: called when the swing foot (landed_side) touches down
        at actual_foot_pos (xyz).  Updates the remaining timeline so future footsteps
        are consistent with the actual robot state.

        Only updates phases AFTER the current time t to avoid modifying completed steps.

        Updates made:
          1. swing_from for the next SS phase that swings landed_side — the foot must
             depart from its actual position, not the originally planned one.
             swing_to (the landing target) is kept unchanged; it was computed by the
             step planner for balance and should not be shifted by tracking error.
          2. The DCM/CoM trajectory is regenerated from t onward so the CoM reference
             stays consistent with any ZMP changes.  The past CoM entries (before t)
             are preserved to avoid a discontinuous reference jump.
        """
        actual = np.asarray(actual_foot_pos, float)

        # Update swing_from for the NEXT future SS where landed_side swings.
        # Only update the immediately-next phase; further-future phases are corrected
        # at their own touchdown events, avoiding cascading errors.
        for ph in self.timeline:
            if ph["t1"] <= t:
                continue  # past phase, skip
            if ph["type"] == "SS" and ph["swing"] == landed_side:
                ph["swing_from"] = actual.tolist()
                # swing_to is deliberately left unchanged: the planned landing
                # target was chosen for balance and we should still aim for it.
                break

        # Regenerate DCM/CoM from the current CoM position onward.
        # The backward DCM pass depends only on ZMP (unchanged), so the DCM is the
        # same as before; the forward CoM pass is restarted from the actual CoM at t
        # to keep numerical consistency.  We splice: entries before idx_now are kept
        # intact to avoid a discontinuous reference jump in the controller.
        idx_now = self._idx(t)
        com_now = self.traj["com"][idx_now].copy()
        new_traj = self.dcm.generate(self.timeline, com_now)
        self.traj["dcm"] = new_traj["dcm"]   # backward pass: full array correct
        self.traj["zmp"] = new_traj["zmp"]   # same ZMP grid (unchanged timeline)
        n_future = len(self.traj["com"]) - idx_now
        self.traj["com"][idx_now:]     = new_traj["com"][:n_future]
        self.traj["com_vel"][idx_now:] = new_traj["com_vel"][:n_future]

    def support_box(self, t, foot_half, margin=0.0):
        """Axis-aligned CoP feasibility box at time t: returns (lo[2], hi[2]).

        During SS: tight rectangle around the stance foot.
        During DS: bounding box of both foot centers (conservative, always feasible).
        foot_half: (half_x, half_y) of the contact patch.
        """
        ph, _ = phase_at(self.timeline, t)
        fh = np.maximum(np.asarray(foot_half, float) - margin, 0.0)
        if ph["support"] == "double":
            z_from = np.asarray(ph["zmp_from"], float)
            z_to = np.asarray(ph["zmp_to"], float)
            lo = np.minimum(z_from, z_to) - fh
            hi = np.maximum(z_from, z_to) + fh
        else:
            foot_pos = np.asarray(ph["zmp_from"], float)
            lo = foot_pos - fh
            hi = foot_pos + fh
        return lo, hi
