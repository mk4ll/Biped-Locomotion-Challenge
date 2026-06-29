"""Walking controller: DCM tracking + swing foot + whole-body QP.

Each tick:
  1. query the WalkPlan for the reference DCM, nominal ZMP, active contacts and
     swing-foot target at the current gait time;
  2. measure the DCM  ξ = com + v_com/ω  and apply the DCM feedback law
        p_cmd = p_ref + k_dcm (ξ_meas - ξ_ref);
  3. turn that into a desired CoM acceleration via the LIPM
        a_com_xy = ω² (com - p_cmd),   a_com_z = PD(height);
  4. assemble WBC tasks (CoM, pelvis orientation, swing foot, posture) and the
     stance contact set, solve the QP, and return joint torques.

Contacts follow the planned schedule (time-based); the short double-support
overlaps make that robust.  Contact-event gating belongs to Phase 4.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from ..wbc import Task, Contact, WholeBodyQP
from ..walk_plan import plan_walk, plan_walk_velocity
from ..dcm_mpc import DCMPreviewMPC, MPCWeights


@dataclasses.dataclass
class WalkGains:
    k_dcm: float = 2.0
    com_z_kp: float = 120.0
    com_z_kd: float = 22.0
    ori_kp: float = 200.0
    ori_kd: float = 28.0
    swing_kp: float = 400.0
    swing_kd: float = 40.0
    swing_ori_kp: float = 250.0
    swing_ori_kd: float = 30.0
    post_kp: float = 60.0
    post_kd: float = 12.0
    w_com: float = 200.0
    w_ori: float = 4.0
    w_swing: float = 60.0
    w_swing_ori: float = 8.0
    w_post: float = 0.2


def quat_from_normal(n, heading=0.0):
    """Quaternion of a frame whose z-axis is the surface normal n, pointing along
    ``heading`` (rad about world z).

    Lands the swing foot flat on a slope/tread *and* yawed to the walking
    direction (for turning / sideways steps).
    """
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    x_ref = np.array([np.cos(heading), np.sin(heading), 0.0])
    x_axis = x_ref - (x_ref @ n) * n
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(n, x_axis)
    R = np.column_stack([x_axis, y_axis, n])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def yaw_quat(theta):
    """Upright pelvis quaternion at heading ``theta`` (rad about world z)."""
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])


class WalkingController:
    def __init__(self, robot, terrain, q_nom, gains: WalkGains = None,
                 mu: float = 0.5, reactive: bool = False, k_step: float = 1.0,
                 step_adjust_max: float = 0.18, arm_swing: bool = True,
                 k_arm: float = 1.0, mpc: bool = False,
                 mpc_horizon: float = 1.6, mpc_dt: float = 0.1,
                 mpc_weights: MPCWeights = None, velocity=None, **plan_kw):
        self.r = robot
        self.terrain = terrain
        self.g = gains or WalkGains()
        self.mu = mu
        self.qp = WholeBodyQP(robot)
        self.q_nom = q_nom.copy()
        self.quat_des = q_nom[3:7].copy()
        # omnidirectional velocity command (vx, vy, vyaw); None -> straight gait
        self.velocity = None if velocity is None else np.array(velocity, float)
        self._cur_yaw = 0.0
        self.S = np.zeros((robot.nu, robot.nv))
        for k, vadr in enumerate(robot.act_vadr):
            self.S[k, vadr] = 1.0

        # reactive (capture-point) footstep adjustment
        self.reactive = reactive
        self.k_step = k_step
        self.step_adjust_max = step_adjust_max
        self.react_ramp_start = 0.5    # fraction of swing before adjustment acts
        self.react_deadband = 0.015    # m, ignore tracking noise below this
        self._prev_phase = None
        self._committed_delta = np.zeros(2)

        # DCM preview MPC (optimises the CoP over a horizon instead of the
        # one-step proportional feedback law); built after the plan exists.
        self.use_mpc = mpc
        self._mpc_kw = dict(horizon=mpc_horizon, dt=mpc_dt, weights=mpc_weights)
        self.mpc = None
        self._t = 0.0
        # re-solve the MPC at ~50 Hz and hold the CoP setpoint between solves
        # (the WBC still runs every tick); keeps the live viewer real-time.
        self.mpc_solve_dt = 0.02
        self._mpc_last_t = None
        self._mpc_p_cmd = None

        # natural arm swing: drive each shoulder pitch off the contralateral hip
        # (only when the robot's shoulder/hip axes map to a clean pitch swing)
        self.arm_swing = arm_swing and robot.config.arm_swing_enabled
        self.k_arm = k_arm
        g = robot.groups
        ai, hi = robot.config.arm_swing_idx, robot.config.hip_pitch_idx
        self._lsh_q = g["arm_left"].qadr[ai]    # left shoulder pitch (qpos adr)
        self._rsh_q = g["arm_right"].qadr[ai]
        self._lhip_q = g["leg_left"].qadr[hi]   # left hip pitch
        self._rhip_q = g["leg_right"].qadr[hi]

        com0 = robot.com()
        self.z_des = float(com0[2])
        # low-pass for the CoM-height target (smooths contact-switch / riser steps)
        dt = float(robot.m.opt.timestep)
        self._z_alpha = 1.0 - np.exp(-dt / 0.08)   # ~80 ms time constant
        self._z_des_filt = None
        self._plan_kw = plan_kw
        self._build_plan()

        if self.use_mpc:
            # full foot half-extent (the commanded CoP is still clipped to the
            # measured support polygon in _com_task, so this is not optimistic)
            cp = robot.foot_corner_points("left")
            foot_half = (np.ptp(cp[:, 0]) / 2, np.ptp(cp[:, 1]) / 2)
            self.mpc = DCMPreviewMPC(self.omega, foot_half=foot_half,
                                     **self._mpc_kw)

    def _pelvis_yaw(self):
        q = self.r.pelvis_quat()
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def _build_plan(self):
        """(Re)build the footstep/DCM plan from the robot's current foot poses."""
        r = self.r
        foot_xy = {s: r.foot_pos(s)[:2].copy() for s in ("left", "right")}
        foot_z = min(r.foot_pos("left")[2], r.foot_pos("right")[2])
        self.z_des = float(r.com()[2])
        # nominal CoM height *above the local ground* -- held constant while the
        # absolute target follows the terrain (so legs don't compress uphill).
        self.com_h = self.z_des - foot_z
        if self.velocity is not None:
            vx, vy, vyaw = self.velocity
            self.plan = plan_walk_velocity(
                foot_xy, foot_z, com_height=self.com_h,
                vx=vx, vy=vy, vyaw=vyaw, theta0=self._pelvis_yaw(),
                terrain=self.terrain, **self._plan_kw)
        else:
            self.plan = plan_walk(foot_xy, foot_z, com_height=self.com_h,
                                  terrain=self.terrain, **self._plan_kw)
        self.omega = self.plan.omega

    def set_velocity(self, vx, vy, vyaw):
        """Update the omnidirectional command (applied on the next replan)."""
        self.velocity = np.array([vx, vy, vyaw], float)

    def restart(self):
        """Rebuild the plan from the current state (call when a plan finishes,
        with the robot settled in double support, to keep walking)."""
        self._build_plan()
        self._prev_phase = None
        self._committed_delta = np.zeros(2)
        self._z_des_filt = None
        if self.mpc is not None:
            self.mpc.reset()
            self._mpc_last_t = None
            self._mpc_p_cmd = None

    def _reactive_adjust(self, s):
        """Capture-point footstep adjustment: shift the swing target by the DCM
        error, and commit it to the plan when the foot lands (SS->DS)."""
        cur = s["phase"]
        if self._prev_phase is not None and cur != self._prev_phase:
            if (self.plan.phases[self._prev_phase].kind == "SS"
                    and self.plan.phases[cur].kind == "DS"):
                self.plan.shift_future(cur, self._committed_delta)
            self._committed_delta = np.zeros(2)
        self._prev_phase = cur

        if s["swing"] is None:
            return s
        # Capture-point law: predict the DCM forward to touchdown and compare to
        # the nominal DCM at touchdown.  delta is identically zero in nominal
        # walking (so no drift) and only reacts to genuine deviations.
        ph = self.plan.phases[s["phase"]]
        T, tau = ph.duration, s["tau"]
        u = s["zmp"]                                  # stance ZMP during SS
        com = self.r.com()
        xi_meas = com[:2] + (self.r.com_jacobian() @ self.r.d.qvel)[:2] / self.omega
        xi_pred_td = u + np.exp(self.omega * (T - tau)) * (xi_meas - u)
        xi_nom_td = u + np.exp(self.omega * T) * (ph.xi_start - u)
        err = xi_pred_td - xi_nom_td
        # deadband: ignore normal tracking noise
        err = np.sign(err) * np.maximum(np.abs(err) - self.react_deadband, 0.0)
        # ramp in over the latter part of swing (short, reliable prediction
        # horizon) so nominal early-swing prediction noise is not amplified
        r0 = self.react_ramp_start
        ramp = np.clip((tau / T - r0) / max(1e-3, 1.0 - r0 - 0.1), 0.0, 1.0)
        delta = ramp * np.clip(self.k_step * err,
                               -self.step_adjust_max, self.step_adjust_max)
        self._committed_delta = delta
        s = dict(s)
        sw = dict(s["swing"])
        sw["pos"] = sw["pos"].copy()
        sw["pos"][:2] += delta
        sw["pos"][2] = self.terrain.height(sw["pos"][0], sw["pos"][1])
        s["swing"] = sw
        return s

    # ---- tasks ---------------------------------------------------------
    def _support_bounds(self, contact_sides, margin=0.02):
        """Axis-aligned bounds of the current support polygon (xy), with margin.

        The commanded ZMP is clamped to this so the CoM task never chases a CoP
        the feet can't produce -- lateral DCM is only correctable in DS anyway.
        """
        pts = np.vstack([self.r.foot_corner_points(s) for s in contact_sides])[:, :2]
        return pts.min(0) - margin, pts.max(0) + margin

    def _com_task(self, s, contact_sides):
        Jcom = self.r.com_jacobian()
        com = self.r.com()
        vcom = self.r.com_vel(Jcom)
        xi = com[:2] + vcom[:2] / self.omega
        if self.mpc is not None:
            # receding-horizon CoP (already support-constrained over the horizon),
            # re-solved at mpc_solve_dt and held between solves
            if (self._mpc_p_cmd is None or self._mpc_last_t is None
                    or self._t - self._mpc_last_t >= self.mpc_solve_dt):
                self._mpc_p_cmd = self.mpc.solve(self.plan, self._t, xi)
                self._mpc_last_t = self._t
            p_cmd = self._mpc_p_cmd
        else:
            # one-step proportional DCM feedback law
            p_cmd = s["zmp"] + self.g.k_dcm * (xi - s["xi"])
        lo, hi = self._support_bounds(contact_sides)
        p_cmd = np.clip(p_cmd, lo, hi)
        a_xy = self.omega ** 2 * (com[:2] - p_cmd)
        # CoM height target follows the local ground (mean of contact-foot
        # heights).  Low-pass it: the raw target jumps when the contact set
        # switches (SS<->DS, and each riser on stairs) -- a step in z_des is a
        # vertical jolt through the whole body.  The filter makes the CoM rise
        # smoothly between footholds.
        ground = float(np.mean([self.r.foot_pos(side)[2] for side in contact_sides]))
        z_target = ground + self.com_h
        if self._z_des_filt is None:
            self._z_des_filt = z_target
        self._z_des_filt += self._z_alpha * (z_target - self._z_des_filt)
        self.z_des = self._z_des_filt
        a_z = self.g.com_z_kp * (self.z_des - com[2]) - self.g.com_z_kd * vcom[2]
        return Task(Jcom, np.array([a_xy[0], a_xy[1], a_z]), self.g.w_com), xi, p_cmd

    def _orientation_task(self):
        Jr = self.r.body_angular_jacobian(self.r.pelvis_id)
        omega = Jr @ self.r.d.qvel
        # turn the pelvis to the walking heading (yaw); upright otherwise
        quat_des = (yaw_quat(self._cur_yaw) if self.velocity is not None
                    else self.quat_des)
        err = np.zeros(3)
        mujoco.mju_subQuat(err, quat_des, self.r.pelvis_quat())
        acc = self.g.ori_kp * err - self.g.ori_kd * omega
        return Task(Jr, acc, self.g.w_ori)

    def _posture_task(self):
        q, qd = self.r.d.qpos, self.r.d.qvel
        q_tgt = self.q_nom.copy()
        if self.arm_swing:
            # each arm swings opposite its contralateral leg (counter-rotation)
            q_tgt[self._lsh_q] = self.q_nom[self._lsh_q] + \
                self.k_arm * (q[self._rhip_q] - self.q_nom[self._rhip_q])
            q_tgt[self._rsh_q] = self.q_nom[self._rsh_q] + \
                self.k_arm * (q[self._lhip_q] - self.q_nom[self._lhip_q])
        acc = np.zeros(self.r.nv)
        for grp in self.r.groups.values():
            for qa, va in zip(grp.qadr, grp.vadr):
                acc[va] = self.g.post_kp * (q_tgt[qa] - q[qa]) - self.g.post_kd * qd[va]
        return Task(self.S, acc[self.r.act_vadr], self.g.w_post)

    def _swing_task(self, swing):
        side = swing["side"]
        Jp, Jr = self.r.site_jacobian(side)
        pos = self.r.foot_pos(side)
        bid = self.r.foot_body[side]
        v = self.r.d.qvel
        Jdv = self.r.point_jacobian_dot(pos, bid) @ v
        vel = Jp @ v
        acc_des = (swing["acc"]
                   + self.g.swing_kp * (swing["pos"] - pos)
                   + self.g.swing_kd * (swing["vel"] - vel))
        tasks = [Task(Jp, acc_des - Jdv, self.g.w_swing)]

        # orientation: land flat on the terrain (foot z-axis -> surface normal),
        # yawed to the walking heading
        nrm = self.terrain.normal(swing["pos"][0], swing["pos"][1])
        quat_des = quat_from_normal(nrm, self._cur_yaw)
        quat_cur = np.zeros(4)
        mujoco.mju_mat2Quat(quat_cur, self.r.foot_rot(side).flatten())
        err = np.zeros(3)
        mujoco.mju_subQuat(err, quat_des, quat_cur)
        omega_f = Jr @ v
        acc_ori = self.g.swing_ori_kp * err - self.g.swing_ori_kd * omega_f
        tasks.append(Task(Jr, acc_ori, self.g.w_swing_ori))
        return tasks

    def _contacts(self, contact_sides):
        out = []
        for side in contact_sides:
            bid = self.r.foot_body[side]
            for p in self.r.foot_corner_points(side):
                nrm = self.terrain.normal(p[0], p[1])
                out.append(Contact(p, bid, nrm, self.mu))
        return out

    # ---- main ----------------------------------------------------------
    def __call__(self, t):
        self._t = t
        s = self.plan.eval(t)
        self._cur_yaw = s.get("yaw", 0.0)
        if self.reactive:
            s = self._reactive_adjust(s)
        com_task, xi, p_cmd = self._com_task(s, s["contacts"])
        tasks = [com_task, self._orientation_task(), self._posture_task()]
        if s["swing"] is not None:
            tasks.extend(self._swing_task(s["swing"]))
        sol = self.qp.solve(tasks, self._contacts(s["contacts"]))
        if sol is None:
            return np.zeros(self.r.nu), {"plan": s, "xi": xi, "p_cmd": p_cmd, "ok": False}
        sol.update({"plan": s, "xi": xi, "p_cmd": p_cmd, "ok": True})
        return sol["tau"], sol
