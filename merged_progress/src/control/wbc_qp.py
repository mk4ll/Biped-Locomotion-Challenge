"""Whole-Body Control as a QP Inverse Dynamics problem (Lecture 9).

Decision variables:  x = [v̇ (nv) ; τ (nu) ; f (3*ncp)]
Objective:           min Σ_i w_i ‖J_i v̇ − b_i‖²  + regularization
Equalities:
   dynamics:   [M  −Sᵀ  −J_cᵀ] x = −h
   contacts:   [J_c  0  0] x = −J̇_c v          (stance feet, no acceleration)
Inequalities:
   friction cones (pyramid) on f ;  τ_min ≤ τ ≤ τ_max
Solved each control step with qpsolvers (quadprog), with a safe fallback if
the problem is infeasible.
"""
import numpy as np
from qpsolvers import solve_qp

from src.dynamics.contacts import friction_pyramid


class WBCQP:
    def __init__(self, terms, params):
        self.terms = terms
        nv, nu = terms.nv, terms.nu
        self.nv, self.nu = nv, nu
        w = params["wbc"]
        self.mu = w["friction_mu"]
        self.f_min = w["f_min"]
        self.solver = w["solver"]
        self.solver_fallback = w.get("solver_fallback", "osqp")
        self.reg_a = w["reg"]["qddot"]
        self.reg_tau = w["reg"]["torque"]
        self.reg_f = w["reg"]["force"]
        # Torque limits from the model ctrlrange (actuator order).
        self.tau_lo = terms.model.actuator_ctrlrange[:, 0].copy()
        self.tau_hi = terms.model.actuator_ctrlrange[:, 1].copy()
        self.last_tau = np.zeros(nu)
        self.n_infeasible = 0

    def solve(self, data, tasks, stance_feet, R_surface=None, fallback_tau=None):
        """Solve the WBC QP.

        stance_feet: list of {'site': foot_site_id, 'corners': [corner_site_ids]}.
        The rigid 6D contact constraint is applied at each foot 'site'; contact
        FORCES live at the 'corners' (for friction cones / CoP). This keeps the
        contact equality full-rank (6 per foot) while retaining a force per corner.
        Returns dict with tau, qddot, f, ok.
        """
        T = self.terms
        nv, nu = self.nv, self.nu
        force_ids = [c for foot in stance_feet for c in foot["corners"]]
        foot_sites = [foot["site"] for foot in stance_feet]
        ncp = len(force_ids)
        nf = 3 * ncp
        n = nv + nu + nf
        ia = slice(0, nv)
        it = slice(nv, nv + nu)
        iff = slice(nv + nu, n)

        M = T.mass_matrix(data)
        h = T.bias(data)
        # Force-mapping Jacobian: 3D translational Jacobian per corner point.
        Jf = T.contact_jacobian(data, force_ids)                  # (nf, nv)
        # Contact constraint Jacobian: 6D per stance foot (rigid, full-rank).
        Jcon = (np.vstack([T.site_jacobian6(data, s) for s in foot_sites])
                if foot_sites else np.zeros((0, nv)))
        Jdcon = T.site_jdot_v6(data, foot_sites)                  # (6*nfeet,)

        # ---- objective ----
        P = np.zeros((n, n))
        q = np.zeros(n)
        P[ia, ia] += self.reg_a * np.eye(nv)
        P[it, it] += self.reg_tau * np.eye(nu)
        P[iff, iff] += self.reg_f * np.eye(nf)
        for task in tasks:
            J, b, wt = task.compute(data)
            P[ia, ia] += wt * (J.T @ J)
            q[ia] += -wt * (J.T @ b)
        P = 0.5 * (P + P.T)

        # ---- equalities ----
        # dynamics: M v̇ - Sᵀ τ - J_fᵀ f = -h
        A_dyn = np.zeros((nv, n))
        A_dyn[:, ia] = M
        A_dyn[:, it] = -T.S.T
        A_dyn[:, iff] = -Jf.T
        b_dyn = -h
        # contacts (rigid foot): J_con v̇ = -J̇v   (6 rows per stance foot)
        A_con = np.zeros((Jcon.shape[0], n))
        A_con[:, ia] = Jcon
        b_con = -Jdcon
        A = np.vstack([A_dyn, A_con])
        b = np.concatenate([b_dyn, b_con])

        # ---- inequalities: friction cones on f ----
        Gc, hc = friction_pyramid(ncp, self.mu, self.f_min, R_surface)
        G = np.zeros((Gc.shape[0], n))
        G[:, iff] = Gc
        hineq = hc

        # ---- bounds: torque limits ----
        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)
        lb[it] = self.tau_lo
        ub[it] = self.tau_hi

        x = solve_qp(P, q, G=G, h=hineq, A=A, b=b, lb=lb, ub=ub, solver=self.solver)
        if x is None:
            x = self._fallback_solve(P, q, G, hineq, A, b, lb, ub)

        if x is None:
            self.n_infeasible += 1
            tau = self.last_tau if fallback_tau is None else fallback_tau
            return {"tau": tau, "qddot": np.zeros(nv), "f": np.zeros(nf), "ok": False}

        tau = x[it].copy()
        self.last_tau = tau
        return {"tau": tau, "qddot": x[ia].copy(), "f": x[iff].copy(), "ok": True}

    def _fallback_solve(self, P, q, G, hineq, A, b, lb, ub):
        """Relax: try fallback solver, then drop friction lower bound margin."""
        x = solve_qp(P, q, G=G, h=hineq, A=A, b=b, lb=lb, ub=ub,
                     solver=self.solver_fallback)
        if x is not None:
            return x
        # last resort: increase regularization for conditioning
        Pr = P + 1e-3 * np.eye(P.shape[0])
        return solve_qp(Pr, q, G=G, h=hineq, A=A, b=b, lb=lb, ub=ub,
                        solver=self.solver_fallback)
