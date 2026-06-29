"""Phase 1 controller: double-support standing balance + lateral weight shift.

Builds three weighted WBC tasks each tick:
  * CoM        -- track a (possibly moving) CoM reference, z held constant
  * pelvis     -- keep the torso/pelvis upright
  * posture    -- regularize actuated joints toward a nominal configuration

Both feet are stance contacts (4 corner points each).  Getting this stable
proves the contact handling, dynamics and QP are correct before any stepping.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from ..wbc import Task, Contact, WholeBodyQP


@dataclasses.dataclass
class StandingGains:
    com_kp: float = 120.0
    com_kd: float = 22.0
    ori_kp: float = 200.0
    ori_kd: float = 28.0
    post_kp: float = 60.0
    post_kd: float = 12.0
    w_com: float = 10.0
    w_ori: float = 4.0
    w_post: float = 0.2


class StandingController:
    def __init__(self, robot, terrain, gains: StandingGains = None, mu: float = 0.5,
                 q_nom=None):
        self.r = robot
        self.terrain = terrain
        self.g = gains or StandingGains()
        self.mu = mu
        self.qp = WholeBodyQP(robot)
        # nominal posture for the regularization task (defaults to keyframe)
        self.q_nom = (q_nom.copy() if q_nom is not None
                      else robot.keyframe_qpos())
        self.quat_des = self._upright_quat()
        # selection matrix for actuated joints (posture task)
        self.S = np.zeros((robot.nu, robot.nv))
        for k, vadr in enumerate(robot.act_vadr):
            self.S[k, vadr] = 1.0
        self.com_des = None   # xy target, set on first call
        self.z_des = None     # height target, captured once

    def _upright_quat(self):
        return self.r.keyframe_qpos()[3:7].copy()

    def set_com_target(self, xy):
        self.com_des = np.array(xy, dtype=float)

    # ---- task builders -------------------------------------------------
    def _com_task(self, com_des_xyz):
        Jcom = self.r.com_jacobian()
        com = self.r.com()
        vcom = self.r.com_vel(Jcom)
        acc = self.g.com_kp * (com_des_xyz - com) - self.g.com_kd * vcom
        return Task(Jcom, acc, self.g.w_com), Jcom

    def _orientation_task(self):
        Jr = self.r.body_angular_jacobian(self.r.pelvis_id)
        omega = Jr @ self.r.d.qvel
        err = np.zeros(3)
        mujoco.mju_subQuat(err, self.quat_des, self.r.pelvis_quat())
        acc = self.g.ori_kp * err - self.g.ori_kd * omega
        return Task(Jr, acc, self.g.w_ori)

    def _posture_task(self):
        q = self.r.d.qpos
        qd = self.r.d.qvel
        acc = np.zeros(self.r.nv)
        for grp in self.r.groups.values():
            for qa, va in zip(grp.qadr, grp.vadr):
                acc[va] = self.g.post_kp * (self.q_nom[qa] - q[qa]) - self.g.post_kd * qd[va]
        return Task(self.S, acc[self.r.act_vadr], self.g.w_post)

    def _contacts(self):
        out = []
        for side in ("left", "right"):
            bid = self.r.foot_body[side]
            for p in self.r.foot_corner_points(side):
                nrm = self.terrain.normal(p[0], p[1])
                out.append(Contact(p, bid, nrm, self.mu))
        return out

    # ---- main ----------------------------------------------------------
    def __call__(self):
        c = self.r.com()
        if self.com_des is None:
            self.com_des = c[:2].copy()
        if self.z_des is None:
            self.z_des = c[2]
        com_des = np.array([self.com_des[0], self.com_des[1], self.z_des])

        com_task, _ = self._com_task(com_des)
        tasks = [com_task, self._orientation_task(), self._posture_task()]
        sol = self.qp.solve(tasks, self._contacts())
        if sol is None:
            return np.zeros(self.r.nu), None
        return sol["tau"], sol
