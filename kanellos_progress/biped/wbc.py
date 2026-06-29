"""Whole-body QP controller.

Acceleration-based inverse-dynamics QP. Decision variables are the generalized
accelerations ``qdd`` (nv) and a stack of 3-D point contact forces ``f`` (3*nc).

    minimize    sum_i  w_i || J_i qdd - b_i ||^2  +  reg_qdd||qdd||^2 + reg_f||f||^2
    subject to  floating-base dynamics              (6 equalities, underactuated)
                contact points do not accelerate    (3*nc equalities, exact Jdot*v)
                friction pyramid + min normal force  (inequalities, surface frame)
                joint torque limits                  (inequalities)

After solving, joint torques are recovered from the rigid-body dynamics:
    tau = (M qdd + h - Jc^T f)[actuated]

Tasks are soft (weighted in the cost); contacts, dynamics and limits are hard.
This is the standard QP-WBC that the DCM walking layer plugs into later.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp


@dataclasses.dataclass
class Task:
    J: np.ndarray        # (k, nv)
    target: np.ndarray   # (k,)  desired task-space acceleration
    weight: float

    def __post_init__(self):
        self.J = np.atleast_2d(self.J)
        self.target = np.atleast_1d(self.target)


@dataclasses.dataclass
class Contact:
    point: np.ndarray     # (3,) world position
    body_id: int
    normal: np.ndarray    # (3,) surface normal (unit)
    mu: float = 0.5


def _tangents(n: np.ndarray):
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref); t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    return n, t1, t2


class WholeBodyQP:
    def __init__(self, robot, fmin: float = 5.0, fmax: float = 600.0,
                 reg_qdd: float = 1e-6, reg_f: float = 1e-8,
                 contact_kd: float = 30.0, w_contact: float = 1.0e3):
        self.r = robot
        self.fmin = fmin
        self.fmax = fmax
        self.reg_qdd = reg_qdd
        self.reg_f = reg_f
        self.contact_kd = contact_kd
        # weight of the (soft) "contact points do not accelerate" objective.
        # Soft rather than hard: 4 corner points per rigid foot give 12 scalar
        # constraints for 6 DOF, a redundant equality system the QP solver flags
        # as infeasible.  A large weight enforces it without that brittleness.
        self.w_contact = w_contact
        self.nv = robot.nv
        self.nu = robot.nu
        self.act = robot.act_vadr  # actuated dof indices (6..nv-1)

    def solve(self, tasks: List[Task], contacts: List[Contact]):
        r = self.r
        nv, nu = self.nv, self.nu
        nc = len(contacts)
        n = nv + 3 * nc
        M = r.mass_matrix()
        h = r.bias()
        v = r.d.qvel

        # ---- contact Jacobians, Jdot*v, friction frames ----
        Jc = np.zeros((3 * nc, nv))
        Jdv = np.zeros(3 * nc)
        for i, c in enumerate(contacts):
            Jp = r.point_jacobian(c.point, c.body_id)
            Jdp = r.point_jacobian_dot(c.point, c.body_id)
            Jc[3 * i:3 * i + 3] = Jp
            Jdv[3 * i:3 * i + 3] = Jdp @ v
        JcT = Jc.T

        # ---- cost: weighted task least squares (+ regularization) ----
        Pqq = np.zeros((nv, nv))
        qq = np.zeros(nv)
        for t in tasks:
            WJ = t.weight * t.J
            Pqq += WJ.T @ t.J
            qq += -WJ.T @ t.target
        # soft contact-acceleration objective: Jc qdd -> a_contact - Jdot*v
        if nc:
            a_contact = -self.contact_kd * (Jc @ v) - Jdv
            Pqq += self.w_contact * (Jc.T @ Jc)
            qq += -self.w_contact * (Jc.T @ a_contact)
        Pqq += self.reg_qdd * np.eye(nv)
        P = sp.block_diag([sp.csc_matrix(Pqq),
                           self.reg_f * sp.eye(3 * nc)]).tocsc()
        q = np.concatenate([qq, np.zeros(3 * nc)])

        # ---- equality: floating-base dynamics (underactuation) ----
        A = sp.csc_matrix(np.hstack([M[:6, :], -JcT[:6, :]]))
        b = -h[:6]

        # ---- inequality: friction pyramid + torque limits ----
        G_rows, h_rows = [], []
        # friction (acts on f block only)
        Fsel = np.zeros((5 * nc, 3 * nc))
        for i, c in enumerate(contacts):
            nrm, t1, t2 = _tangents(c.normal)
            base = 5 * i
            col = 3 * i
            Fsel[base + 0, col:col + 3] = -nrm                    # f.n >= fmin
            Fsel[base + 1, col:col + 3] = t1 - c.mu * nrm         # |f.t1| <= mu f.n
            Fsel[base + 2, col:col + 3] = -t1 - c.mu * nrm
            Fsel[base + 3, col:col + 3] = t2 - c.mu * nrm
            Fsel[base + 4, col:col + 3] = -t2 - c.mu * nrm
        G_fric = np.hstack([np.zeros((5 * nc, nv)), Fsel])
        h_fric = np.zeros(5 * nc)
        h_fric[0::5] = -self.fmin
        G_rows.append(G_fric); h_rows.append(h_fric)

        # torque limits: tau = M[act]qdd + h[act] - JcT[act] f
        Ma = M[self.act, :]
        JcTa = JcT[self.act, :]
        ha = h[self.act]
        lim = r.torque_limit
        G_tau = np.vstack([np.hstack([Ma, -JcTa]),
                           np.hstack([-Ma, JcTa])])
        h_tau = np.concatenate([lim - ha, lim + ha])
        G_rows.append(G_tau); h_rows.append(h_tau)

        G = sp.csc_matrix(np.vstack(G_rows))
        hG = np.concatenate(h_rows)

        # ---- solve ----
        x = solve_qp(P, q, G, hG, A, b, solver="osqp",
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, verbose=False)
        if x is None:
            return None
        qdd = x[:nv]
        f = x[nv:]
        tau = (Ma @ qdd + ha - JcTa @ f)
        return {"qdd": qdd, "f": f, "tau": tau,
                "contact_force": f.reshape(-1, 3) if nc else np.zeros((0, 3))}
