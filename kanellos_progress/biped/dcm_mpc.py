"""DCM preview model-predictive control (capture-point MPC).

This replaces the one-step proportional DCM feedback law
``p_cmd = p_ref + k_dcm (xi - xi_ref)`` with a receding-horizon optimisation:
at every tick it solves a small QP over the next ``horizon`` seconds for the
centre-of-pressure (CoP / ZMP) sequence that

  * drives the *measured* divergent component of motion (DCM) back onto its
    reference trajectory,
  * keeps the CoP near the reference ZMP (so it does not waste control
    authority), and
  * stays inside the *planned* support polygon at **every** preview step
    (a hard inequality) -- this is the part a one-step law cannot do: the MPC
    sees that an upcoming single-support phase shrinks the support box and
    starts shifting weight early.

Formulation (deviation form, per axis -- the LIPM/DCM decouples in x and y).
The reference DCM ``xi_ref`` already satisfies the dynamics under the reference
ZMP, so we optimise the *deviation* ``delta = xi - xi_ref`` driven by the CoP
deviation ``dp = p - p_ref``:

    delta_{k+1} = a delta_k + (1 - a) dp_k ,    a = exp(omega * h)   (> 1)

with ``delta_0 = xi_meas - xi_ref(t_now)``.  Condensed over the horizon,

    Delta = Phi delta_0 + Gamma dp ,   dp = [dp_0, ..., dp_{N-1}]

Because ``a > 1`` the open-loop DCM is *unstable*, so a uniform tracking weight
would amplify the far-horizon terms by ``a^N`` and wreck the QP conditioning
(OSQP returns "solved inaccurate").  We therefore **discount** the per-step
tracking weight by ``a^{-2k}`` -- exactly cancelling that growth, which is also
the physically right thing to do (on an unstable system, near-term deviations
are what blow up, so weight them more).  The horizon *constraints* are kept at
full strength; that is where the MPC's anticipation comes from.

    cost = sum_k Q_k delta_k^2  +  w_p sum_k dp_k^2  +  w_dp sum_k (dp_k - dp_{k-1})^2
    s.t.  lo_k - p_ref_k  <=  dp_k  <=  hi_k - p_ref_k        (support polygon)

with ``Q_k = w_xi a^{-2k}``.  The Hessian is constant (built once as a sparse
csc matrix); only the gradient and the box bounds change per tick.  We command
the first CoP ``p_0 = p_ref_0 + dp_0``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp


@dataclasses.dataclass
class MPCWeights:
    w_xi: float = 1.0      # DCM-deviation tracking
    w_p: float = 4e-2      # stay near reference ZMP (sets the effective gain)
    w_dp: float = 2e-2     # CoP smoothness (rate)


class DCMPreviewMPC:
    """Receding-horizon CoP optimiser for DCM tracking with support limits."""

    def __init__(self, omega: float, horizon: float = 1.2, dt: float = 0.1,
                 weights: MPCWeights = None, foot_half=(0.09, 0.04),
                 box_margin: float = 0.0):
        self.omega = float(omega)
        self.h = float(dt)
        self.N = max(2, int(round(horizon / dt)))
        self.w = weights or MPCWeights()
        self.foot_half = np.asarray(foot_half, float)
        self.box_margin = box_margin
        self._dp_prev = None     # last CoP deviation (xy) for the rate term

        self._build_constant_matrices()

    # -- one-time prediction matrices (depend only on omega, h, N, weights) --
    def _build_constant_matrices(self):
        N = self.N
        a = np.exp(self.omega * self.h)
        self.a = a

        # Phi[k] = a^{k+1}  (predict delta_1 .. delta_N from delta_0)
        self.Phi = np.array([a ** (k + 1) for k in range(N)])

        # Gamma[m, j] = a^{m-j} (1 - a)  for j <= m, else 0   (rows = delta_1..N)
        Gamma = np.zeros((N, N))
        for m in range(N):
            for j in range(m + 1):
                Gamma[m, j] = a ** (m - j) * (1.0 - a)
        self.Gamma = Gamma

        # discount that cancels the a^{2k} growth of the unstable DCM mode
        self.Qdiag = self.w.w_xi * a ** (-2.0 * np.arange(1, N + 1))

        # first-difference operator for CoP smoothness (dp_k - dp_{k-1})
        D = np.eye(N)
        for k in range(1, N):
            D[k, k - 1] = -1.0
        self.D = D

        w = self.w
        Q = np.diag(self.Qdiag)
        # constant Hessian: P = 2 (Gamma^T Q Gamma + w_p I + w_dp D^T D)
        Pd = 2.0 * (Gamma.T @ Q @ Gamma
                    + w.w_p * np.eye(N)
                    + w.w_dp * D.T @ D)
        Pd += 1e-9 * np.eye(N)            # strict PD for the solver
        self.P = sp.csc_matrix(Pd)
        self._GtQ = Gamma.T @ Q           # reused in the gradient
        self._DtD_target = D.T            # reused in the gradient

    # ------------------------------------------------------------------ #
    def reset(self):
        self._dp_prev = None

    def _solve_axis(self, d0, p_ref, lo, hi, dp_prev):
        """One QP for a single axis; returns the optimal CoP-deviation seq (N,)."""
        w = self.w
        g = np.zeros(self.N)
        g[0] = dp_prev                       # rate target: dp_0 close to dp_prev
        # gradient: q = 2 (Gamma^T Q Phi d0 - w_dp D^T g)
        q = 2.0 * (self._GtQ @ (self.Phi * d0) - w.w_dp * self.D.T @ g)
        lb = lo - p_ref                      # dp box from the support polygon
        ub = hi - p_ref
        # guard against an inverted box (margin wider than the foot): allow dp=0
        lb = np.minimum(lb, 0.0)
        ub = np.maximum(ub, 0.0)
        dp = solve_qp(self.P, q, lb=lb, ub=ub, solver="osqp",
                      eps_abs=1e-6, eps_rel=1e-6, max_iter=8000,
                      polish=False, verbose=False)
        return dp

    def solve(self, plan, t_now: float, xi_meas):
        """Optimise the CoP over the horizon; return the CoP to apply now (xy).

        ``plan`` provides the reference DCM/ZMP and the support box at each
        preview time; ``xi_meas`` is the measured DCM (xy).
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

        d0 = np.asarray(xi_meas, float) - plan.eval(t_now)["xi"]   # deviation now

        dp_prev = self._dp_prev if self._dp_prev is not None else np.zeros(2)
        p_cmd = np.zeros(2)
        dp_now = np.zeros(2)
        for ax in range(2):
            dp = self._solve_axis(d0[ax], p_ref[:, ax],
                                  lo[:, ax], hi[:, ax], dp_prev[ax])
            if dp is None:                  # infeasible -> reference ZMP, clamped
                p_cmd[ax] = float(np.clip(p_ref[0, ax], lo[0, ax], hi[0, ax]))
            else:
                dp_now[ax] = float(dp[0])
                p_cmd[ax] = float(np.clip(p_ref[0, ax] + dp[0],
                                          lo[0, ax], hi[0, ax]))
        self._dp_prev = dp_now
        return p_cmd
