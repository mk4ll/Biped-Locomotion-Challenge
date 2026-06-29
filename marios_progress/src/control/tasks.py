"""Acceleration-level WBC tasks.

Each task produces a linear residual in the joint accelerations v̇:
    r = J_task v̇ - b_task ,   b_task = a_des - J̇_task v
so that minimizing ‖r‖² drives the task acceleration to
    a_des = a_ref + Kp (x_ref - x) + Kd (ẋ_ref - ẋ).

Each task returns (J, b, weight): J is (m x nv), b is (m,). The QP stacks
w * ‖J v̇ - b‖². Task-space drift J̇_task v is approximated as 0 for CoM /
orientation / posture (the PD feedback absorbs it); the stance-contact drift
J̇_c v IS computed analytically (see model_terms) because it enters an
equality constraint.
"""
import numpy as np
import mujoco

from src.utils.math_utils import quat_to_mat, orientation_error


class CoMTask:
    """Track a world CoM position. J via mj_jacSubtreeCom on the root body."""

    def __init__(self, terms, root_body_id, kp, kd, weight):
        self.terms = terms
        self.root = root_body_id
        self.kp, self.kd, self.weight = kp, kd, weight
        self._jacp = np.zeros((3, terms.nv))
        self.p_ref = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.a_ref = np.zeros(3)

    def com(self, data):
        return data.subtree_com[self.root].copy()

    def jacobian(self, data):
        mujoco.mj_jacSubtreeCom(self.terms.model, data, self._jacp, self.root)
        return self._jacp.copy()

    def compute(self, data):
        J = self.jacobian(data)
        p = self.com(data)
        v = J @ data.qvel
        a_des = self.a_ref + self.kp * (self.p_ref - p) + self.kd * (self.v_ref - v)
        return J, a_des, self.weight  # J̇v ~ 0


class OrientationTask:
    """Keep a body's orientation at R_des (default: upright w.r.t. gravity)."""

    def __init__(self, terms, body_id, kp, kd, weight, R_des=None):
        self.terms = terms
        self.body = body_id
        self.kp, self.kd, self.weight = kp, kd, weight
        self.R_des = np.eye(3) if R_des is None else R_des
        self._jacp = np.zeros((3, terms.nv))
        self._jacr = np.zeros((3, terms.nv))

    def jacobian(self, data):
        mujoco.mj_jacBody(self.terms.model, data, self._jacp, self._jacr, self.body)
        return self._jacr.copy()

    def compute(self, data):
        Jr = self.jacobian(data)
        R = data.xmat[self.body].reshape(3, 3)
        e = orientation_error(R, self.R_des)          # world rotation vector
        w = Jr @ data.qvel
        a_des = self.kp * e + self.kd * (-w)
        return Jr, a_des, self.weight


class PostureTask:
    """Regularize actuated joints toward a nominal pose q_nom (drives v̇ on those dofs)."""

    def __init__(self, terms, q_nom_actuated, kp, kd, weight):
        self.terms = terms
        self.act_dof = terms.act_dof
        self.q_nom = np.asarray(q_nom_actuated)
        self.kp, self.kd, self.weight = kp, kd, weight
        nu, nv = terms.nu, terms.nv
        self.J = np.zeros((nu, nv))
        for i, d in enumerate(self.act_dof):
            self.J[i, d] = 1.0
        # qpos address per actuated dof (hinge: qpos = dof index shifted by 1).
        self.qadr = np.array([terms.model.jnt_qposadr[terms.model.dof_jntid[d]]
                              for d in self.act_dof])

    def compute(self, data):
        q = data.qpos[self.qadr]
        dq = data.qvel[self.act_dof]
        a_des = self.kp * (self.q_nom - q) + self.kd * (-dq)
        return self.J, a_des, self.weight


class FootTask:
    """Track a swing-foot pose: 3D position + (optionally) keep the foot flat.

    With track_orientation=True the task is 6D ([lin; ang]) and holds the foot
    orientation at R_des (default = the foot's nominal/flat orientation), so the
    swing foot lands flat instead of on an edge.
    """

    def __init__(self, terms, site_id, kp, kd, weight,
                 track_orientation=False, kp_ori=None, kd_ori=None, R_des=None):
        self.terms = terms
        self.site = site_id
        self.kp, self.kd, self.weight = kp, kd, weight
        self.track_ori = track_orientation
        self.kp_ori = kp if kp_ori is None else kp_ori
        self.kd_ori = kd if kd_ori is None else kd_ori
        self.R_des = R_des
        self.p_ref = np.zeros(3)
        self.v_ref = np.zeros(3)
        self.a_ref = np.zeros(3)

    def compute(self, data):
        p = data.site_xpos[self.site].copy()
        if not self.track_ori:
            J = self.terms.site_jacobian(data, self.site)
            v = J @ data.qvel
            a_des = self.a_ref + self.kp * (self.p_ref - p) + self.kd * (self.v_ref - v)
            return J, a_des, self.weight
        # 6D: position + orientation
        J6 = self.terms.site_jacobian6(data, self.site)         # [lin; ang]
        v6 = J6 @ data.qvel
        a_lin = self.a_ref + self.kp * (self.p_ref - p) + self.kd * (self.v_ref - v6[:3])
        R = data.site_xmat[self.site].reshape(3, 3)
        R_des = R if self.R_des is None else self.R_des
        e_ang = orientation_error(R, R_des)
        a_ang = self.kp_ori * e_ang + self.kd_ori * (-v6[3:])
        return J6, np.concatenate([a_lin, a_ang]), self.weight
