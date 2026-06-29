"""Online DCM walker with step-timing + footstep adjustment (capture-point).

Unlike :class:`WalkingController` (which tracks a *precomputed* DCM/footstep
plan), this controller generates the gait online and is **event-driven**: every
single-support phase, a small QP (:mod:`biped.step_timing`) re-decides both where
the next foot lands *and* when it lands, from the measured DCM.  That "when" is
the piece a fixed-time plan cannot do -- under a hard lateral push the robot
steps down sooner to catch itself.

State machine: DS0 (initial weight shift) -> SS -> DS -> SS -> ...  During SS the
centre of pressure sits at the stance foot (DCM feedback, clamped to the foot),
the swing foot tracks a re-timed min-jerk arc to the QP's target, and the phase
ends on contact / at the QP's chosen touchdown.  The whole-body QP turns the CoM,
swing-foot, pelvis and posture tasks into joint torques, exactly as elsewhere.
"""
from __future__ import annotations

import numpy as np

from ..wbc import Task, Contact, WholeBodyQP
from ..swing import SwingTrajectory
from ..step_timing import StepTimingQP, StepTimingGains
from .walking import WalkGains, quat_from_normal, yaw_quat

import mujoco


class OnlineWalkingController:
    def __init__(self, robot, terrain, q_nom, gains: WalkGains = None,
                 mu: float = 0.5, step_len: float = 0.15, T_step: float = 0.7,
                 t_ds: float = 0.12, t_ds0: float = 0.8, clearance: float = 0.07,
                 first_swing: str = "right", arm_swing: bool = True,
                 k_arm: float = 1.0, timing: bool = True,
                 velocity=None, max_dx: float = 0.18, max_dy: float = 0.10,
                 step_gains: StepTimingGains = None):
        self.r = robot
        self.terrain = terrain
        self.g = gains or WalkGains()
        self.mu = mu
        self.qp = WholeBodyQP(robot)
        self.q_nom = q_nom.copy()
        self.quat_des = q_nom[3:7].copy()
        self.S = np.zeros((robot.nu, robot.nv))
        for k, vadr in enumerate(robot.act_vadr):
            self.S[k, vadr] = 1.0

        self.step_len = step_len
        self.T_step = T_step
        self.t_ds = t_ds
        self.t_ds0 = t_ds0
        self.clearance = clearance
        self.timing = timing
        self.max_dx = max_dx
        self.max_dy = max_dy
        # continuously-updatable body-frame velocity command (vx, vy, vyaw).
        # Default matches the forward step_len so behaviour is unchanged.
        Tc = T_step + t_ds
        if velocity is None:
            self.vx, self.vy, self.vyaw = step_len / Tc, 0.0, 0.0
        else:
            self.vx, self.vy, self.vyaw = (float(v) for v in velocity)
        self.theta = 0.0               # accumulated heading (for turning)

        # arm swing (contralateral hip coupling), config-driven indices
        self.arm_swing = arm_swing and robot.config.arm_swing_enabled
        self.k_arm = k_arm
        g = robot.groups
        ai, hi = robot.config.arm_swing_idx, robot.config.hip_pitch_idx
        self._lsh_q = g["arm_left"].qadr[ai]
        self._rsh_q = g["arm_right"].qadr[ai]
        self._lhip_q = g["leg_left"].qadr[hi]
        self._rhip_q = g["leg_right"].qadr[hi]

        # geometry / LIPM
        com0 = robot.com()
        foot_z = min(robot.foot_pos("left")[2], robot.foot_pos("right")[2])
        self.com_h = float(com0[2]) - foot_z
        self.omega = float(np.sqrt(9.81 / self.com_h))
        self.foot_y = {s: float(robot.foot_pos(s)[1]) for s in ("left", "right")}
        self.W = abs(self.foot_y["left"] - self.foot_y["right"])
        self.stq = StepTimingQP(self.omega, T_step, step_gains or StepTimingGains())

        dt = float(robot.m.opt.timestep)
        self._z_alpha = 1.0 - np.exp(-dt / 0.08)
        self._z_des_filt = float(com0[2])
        self.z_des = float(com0[2])

        self.first_swing = first_swing
        self.restart()

    # ------------------------------------------------------------------ #
    def restart(self):
        r = self.r
        self.phase = "DS0"
        self.t0 = 0.0                  # phase start time
        self.stance = "left" if self.first_swing == "right" else "right"
        self.swing = self.first_swing
        self.cop0 = 0.5 * (r.foot_pos("left")[:2] + r.foot_pos("right")[:2])
        self.cop_target = r.foot_pos(self.stance)[:2].copy()
        self.swing_from = r.foot_pos(self.swing).copy()
        self.swing_to = None
        self.T_eff = self.T_step
        self._last_swing_pos = None

    # ---- helpers (shared task builders) --------------------------------
    def _support_bounds(self, sides, margin=0.02):
        pts = np.vstack([self.r.foot_corner_points(s) for s in sides])[:, :2]
        return pts.min(0) - margin, pts.max(0) + margin

    def _z_des(self, sides):
        ground = float(np.mean([self.r.foot_pos(s)[2] for s in sides]))
        target = ground + self.com_h
        self._z_des_filt += self._z_alpha * (target - self._z_des_filt)
        self.z_des = self._z_des_filt
        return self.z_des

    def _com_task(self, cop_cmd, sides):
        r = self.r
        Jcom = r.com_jacobian()
        com = r.com()
        vcom = r.com_vel(Jcom)
        lo, hi = self._support_bounds(sides)
        cop = np.clip(cop_cmd, lo, hi)
        a_xy = self.omega ** 2 * (com[:2] - cop)
        z = self._z_des(sides)
        a_z = self.g.com_z_kp * (z - com[2]) - self.g.com_z_kd * vcom[2]
        xi = com[:2] + vcom[:2] / self.omega
        return Task(Jcom, np.array([a_xy[0], a_xy[1], a_z]), self.g.w_com), xi

    def _orientation_task(self):
        r = self.r
        Jr = r.body_angular_jacobian(r.pelvis_id)
        omega = Jr @ r.d.qvel
        err = np.zeros(3)
        quat_des = yaw_quat(self.theta) if self.vyaw != 0.0 else self.quat_des
        mujoco.mju_subQuat(err, quat_des, r.pelvis_quat())
        return Task(Jr, self.g.ori_kp * err - self.g.ori_kd * omega, self.g.w_ori)

    def _posture_task(self):
        r = self.r
        q, qd = r.d.qpos, r.d.qvel
        q_tgt = self.q_nom.copy()
        if self.arm_swing:
            q_tgt[self._lsh_q] = self.q_nom[self._lsh_q] + \
                self.k_arm * (q[self._rhip_q] - self.q_nom[self._rhip_q])
            q_tgt[self._rsh_q] = self.q_nom[self._rsh_q] + \
                self.k_arm * (q[self._lhip_q] - self.q_nom[self._lhip_q])
        acc = np.zeros(r.nv)
        for grp in r.groups.values():
            for qa, va in zip(grp.qadr, grp.vadr):
                acc[va] = self.g.post_kp * (q_tgt[qa] - q[qa]) - self.g.post_kd * qd[va]
        return Task(self.S, acc[r.act_vadr], self.g.w_post)

    def _swing_task(self, pos_d, vel_d, acc_d):
        r = self.r
        side = self.swing
        Jp, Jr = r.site_jacobian(side)
        pos = r.foot_pos(side)
        bid = r.foot_body[side]
        v = r.d.qvel
        Jdv = r.point_jacobian_dot(pos, bid) @ v
        acc = acc_d + self.g.swing_kp * (pos_d - pos) + self.g.swing_kd * (vel_d - Jp @ v)
        tasks = [Task(Jp, acc - Jdv, self.g.w_swing)]
        nrm = self.terrain.normal(pos_d[0], pos_d[1])
        quat_des = quat_from_normal(nrm, self.theta)
        quat_cur = np.zeros(4)
        mujoco.mju_mat2Quat(quat_cur, r.foot_rot(side).flatten())
        err = np.zeros(3)
        mujoco.mju_subQuat(err, quat_des, quat_cur)
        acc_o = self.g.swing_ori_kp * err - self.g.swing_ori_kd * (Jr @ v)
        tasks.append(Task(Jr, acc_o, self.g.w_swing_ori))
        return tasks

    def _contacts(self, sides):
        out = []
        for side in sides:
            bid = self.r.foot_body[side]
            for p in self.r.foot_corner_points(side):
                out.append(Contact(p, bid, self.terrain.normal(p[0], p[1]), self.mu))
        return out

    # ---- main ----------------------------------------------------------
    def set_velocity(self, vx, vy, vyaw=0.0):
        """Update the body-frame velocity command (takes effect next step -- no
        replanning/restart; the gait flows continuously)."""
        self.vx, self.vy, self.vyaw = float(vx), float(vy), float(vyaw)

    def _step_disp(self):
        """Per-step torso displacement (dx, dy) and turn dpsi, clamped."""
        Tc = self.T_step + self.t_ds
        dx = float(np.clip(self.vx * Tc, -self.max_dx, self.max_dx))
        dy = float(np.clip(self.vy * Tc, -self.max_dy, self.max_dy))
        dpsi = float(np.clip(self.vyaw * Tc, -0.18, 0.18))
        return dx, dy, dpsi

    def _b_offset(self, stance):
        """Nominal DCM start-of-step offset (world frame) for the command."""
        tau = np.exp(self.omega * self.T_step)
        dx, dy, _ = self._step_disp()
        sgn = 1.0 if stance == "right" else -1.0
        b = np.array([dx / (tau - 1), dy / (tau - 1) + sgn * self.W / (tau + 1)])
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([c * b[0] - s * b[1], s * b[0] + c * b[1]])

    def _nominal_next(self):
        """Nominal next footstep for the current velocity command + heading."""
        u = self.r.foot_pos(self.stance)[:2]
        dx, dy, dpsi = self._step_disp()
        th = self.theta + dpsi
        c, s = np.cos(th), np.sin(th)
        adv = np.array([c * dx - s * dy, s * dx + c * dy])      # torso advance
        dlat = self.foot_y[self.swing] - self.foot_y[self.stance]   # +/- W
        lat = np.array([-s * dlat, c * dlat])                  # swing-side offset
        return u + adv + lat

    def __call__(self, t):
        r = self.r
        te = t - self.t0
        other = {"left": "right", "right": "left"}

        if self.phase == "DS0":
            sides = ("left", "right")
            com = r.com(); vcom = r.com_vel()
            xi = com[:2] + vcom[:2] / self.omega
            # hand the DCM off at (stance + offset) so the first SS starts with
            # the DCM already leaning toward the swing side (stabilizing law, k>1)
            b = self._b_offset(self.stance)
            goal = r.foot_pos(self.stance)[:2] + b
            cop_cmd = goal + self.g.k_dcm * (xi - goal)
            com_task, xi = self._com_task(cop_cmd, sides)
            tasks = [com_task, self._orientation_task(), self._posture_task()]
            sol = self.qp.solve(tasks, self._contacts(sides))
            if te >= self.t_ds0:
                self._begin_swing(t)
            return self._finish(sol, xi)

        if self.phase == "SS":
            u_cur = r.foot_pos(self.stance)[:2].copy()
            com = r.com(); vcom = r.com_vel()
            xi = com[:2] + vcom[:2] / self.omega
            # --- step timing + footstep QP ---
            b = self._b_offset(self.stance)
            if self.timing:
                u_next, t_rem = self.stq.solve(xi, u_cur, te, self._nominal_next(), b)
            else:
                u_next, t_rem = self._nominal_next(), max(self.T_step - te, 0.05)
            # commit timing: allow stepping sooner (push), never keep delaying
            desired_T = te + t_rem
            if desired_T < self.T_eff - 0.02:
                self.T_eff = max(desired_T, te + 0.08)
            self.swing_to = np.array([u_next[0], u_next[1],
                                      self.terrain.height(u_next[0], u_next[1])])
            # re-timed swing arc from lift-off pose to the (moving) target
            traj = SwingTrajectory(self.swing_from, self.swing_to,
                                   max(self.T_eff, te + 1e-3), self.clearance)
            pos_d, vel_d, acc_d = traj.sample(te)
            # DCM feedback CoP at the stance foot
            xi_ref = u_cur + b * np.exp(self.omega * te)
            cop = u_cur + self.g.k_dcm * (xi - xi_ref)
            com_task, xi = self._com_task(cop, (self.stance,))
            tasks = [com_task, self._orientation_task(), self._posture_task()]
            tasks += self._swing_task(pos_d, vel_d, acc_d)
            sol = self.qp.solve(tasks, self._contacts((self.stance,)))
            # transition at the planned touchdown (foot has the full swing to
            # reach its target; an early contact cut the step short -> landed
            # short -> lateral capture failed)
            if te >= self.T_eff:
                self._begin_ds(t, u_next)
            return self._finish(sol, xi)

        # DS: position the DCM at (next-stance + offset) for the upcoming step.
        # The just-landed foot (self.swing) becomes the next stance.
        sides = ("left", "right")
        com = r.com(); vcom = r.com_vel()
        xi = com[:2] + vcom[:2] / self.omega
        b = self._b_offset(self.swing)
        goal = self.cop_target + b
        cop_cmd = goal + self.g.k_dcm * (xi - goal)
        com_task, xi = self._com_task(cop_cmd, sides)
        tasks = [com_task, self._orientation_task(), self._posture_task()]
        sol = self.qp.solve(tasks, self._contacts(sides))
        if te >= self.t_ds:
            self.stance, self.swing = self.swing, self.stance
            self._begin_swing(t)
        return self._finish(sol, xi)

    def _begin_swing(self, t):
        self.phase = "SS"
        self.t0 = t
        self.T_eff = self.T_step
        self.swing_from = self.r.foot_pos(self.swing).copy()
        self.swing_to = None
        # advance the commanded heading by this step's turn
        _, _, dpsi = self._step_disp()
        self.theta += dpsi

    def _begin_ds(self, t, u_next):
        self.phase = "DS"
        self.t0 = t
        self.cop0 = self.r.foot_pos(self.stance)[:2].copy()
        self.cop_target = u_next.copy()

    def _finish(self, sol, xi):
        if sol is None:
            return np.zeros(self.r.nu), {"xi": xi, "phase": self.phase, "ok": False}
        sol.update({"xi": xi, "phase": self.phase, "ok": True})
        return sol["tau"], sol
