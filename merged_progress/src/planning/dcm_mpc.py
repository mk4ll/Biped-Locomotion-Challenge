"""DCM preview model-predictive control (ported from Kanellos; adapted for merged pipeline).

Replaces the one-step proportional DCM feedback law with a receding-horizon QP:
at every tick it optimises the centre-of-pressure (CoP/ZMP) sequence that

  * drives the *measured* DCM back onto its reference trajectory,
  * keeps the CoP near the reference ZMP,
  * stays inside the *planned* support polygon at EVERY preview step (hard
    inequality) — so it anticipates an upcoming single-support phase and shifts
    weight early, which a one-step law cannot.

Formulation (deviation form, per axis — LIPM decouples in x/y):
    delta_{k+1} = a delta_k + (1-a) dp_k ,   a = exp(omega*h) > 1

Discount trick: weight per step by a^{-2k} to cancel the a^N growth that
otherwise wrecks QP conditioning (OSQP "solved inaccurate"). Full-horizon
constraints are kept at full strength — that is where anticipation lives.

    cost = Σ_k Q_k delta_k² + w_p Σ dp_k² + w_dp Σ (dp_k - dp_{k-1})²
    s.t.  lo_k - p_ref_k <= dp_k <= hi_k - p_ref_k    (support polygon)

Hessian is constant (built once, sparse); only gradient + box bounds change
each tick. We command the first CoP  p_0 = p_ref_0 + dp_0.

Reference: Kanellos biped/dcm_mpc.py (Phase 7); Englsberger 2015 (DCM).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp


@dataclasses.dataclass
class MPCWeights:
    w_xi: float = 1.0    # DCM-deviation tracking
    w_p: float = 4e-2    # stay near reference ZMP (sets effective gain)
    w_dp: float = 2e-2   # CoP smoothness (rate penalty)


class DCMPreviewMPC:
    """Receding-horizon CoP optimiser for DCM tracking with support limits."""

    def __init__(self, omega: float, horizon: float = 1.2, dt: float = 0.02,
                 weights: MPCWeights | None = None, foot_half=(0.09, 0.04),
                 box_margin: float = 0.0):
        self.omega = float(omega)
        self.h = float(dt)
        self.N = max(2, int(round(horizon / dt)))
        self.w = weights or MPCWeights()
        self.foot_half = np.asarray(foot_half, float)
        self.box_margin = box_margin
        self._dp_prev = None
        self._build_constant_matrices()

    def _build_constant_matrices(self):
        N = self.N
        a = np.exp(self.omega * self.h)
        self.a = a

        self.Phi = np.array([a ** (k + 1) for k in range(N)])

        Gamma = np.zeros((N, N))
        for m in range(N):
            for j in range(m + 1):
                Gamma[m, j] = a ** (m - j) * (1.0 - a)
        self.Gamma = Gamma

        self.Qdiag = self.w.w_xi * a ** (-2.0 * np.arange(1, N + 1))

        D = np.eye(N)
        for k in range(1, N):
            D[k, k - 1] = -1.0
        self.D = D

        w = self.w
        Q = np.diag(self.Qdiag)
        Pd = 2.0 * (Gamma.T @ Q @ Gamma
                    + w.w_p * np.eye(N)
                    + w.w_dp * D.T @ D)
        Pd += 1e-9 * np.eye(N)
        self.P = sp.csc_matrix(Pd)
        self._GtQ = Gamma.T @ Q
        self._Dt = D.T

    def reset(self):
        self._dp_prev = None

    def _solve_axis(self, d0, p_ref, lo, hi, dp_prev):
        """QP for a single axis; returns optimal CoP-deviation sequence (N,)."""
        w = self.w
        g = np.zeros(self.N)
        g[0] = dp_prev
        q = 2.0 * (self._GtQ @ (self.Phi * d0) - w.w_dp * self._Dt @ g)
        lb = np.minimum(lo - p_ref, 0.0)
        ub = np.maximum(hi - p_ref, 0.0)
        return solve_qp(self.P, q, lb=lb, ub=ub, solver="osqp",
                        eps_abs=1e-6, eps_rel=1e-6, max_iter=8000,
                        polish=False, verbose=False)

    def solve(self, plan, t_now: float, xi_meas):
        """Return the CoP to apply now (xy).

        ``plan`` is a WalkPlan with eval(t) -> {zmp, xi} and
        support_box(t, foot_half, margin) -> (lo, hi).
        ``xi_meas`` is the measured DCM (xy, 2-vector).
        """
        N, h = self.N, self.h
        p_ref = np.zeros((N, 2))
        lo = np.zeros((N, 2))
        hi = np.zeros((N, 2))
        for k in range(N):
            tk = t_now + k * h
            s = plan.eval(tk)
            p_ref[k] = s["zmp"]
            blo, bhi = plan.support_box(tk, self.foot_half, margin=-self.box_margin)
            lo[k], hi[k] = blo, bhi

        xi_ref_now = plan.eval(t_now)["xi"]
        d0 = np.asarray(xi_meas, float) - xi_ref_now

        dp_prev = self._dp_prev if self._dp_prev is not None else np.zeros(2)
        p_cmd = np.zeros(2)
        dp_now = np.zeros(2)
        for ax in range(2):
            dp = self._solve_axis(d0[ax], p_ref[:, ax], lo[:, ax], hi[:, ax], dp_prev[ax])
            if dp is None:
                p_cmd[ax] = float(np.clip(p_ref[0, ax], lo[0, ax], hi[0, ax]))
            else:
                dp_now[ax] = float(dp[0])
                p_cmd[ax] = float(np.clip(p_ref[0, ax] + dp[0], lo[0, ax], hi[0, ax]))
        self._dp_prev = dp_now
        return p_cmd
