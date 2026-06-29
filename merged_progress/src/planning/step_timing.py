"""Step timing + footstep adjustment QP (capture-point, Khadiv et al. 2016).

The classic DCM controller fixes *where* and *when* the next step lands ahead of
time.  Under a hard lateral push that is not enough: the DCM runs away faster
than a pre-planned step can catch.  The fix is to make **both** the next footstep
location and the **step timing** joint decision variables, solved online every tick
from the *measured* DCM.

Model.  During single support with the CoP held at the stance foot ``u``, the
DCM obeys ``xi(t) = u + (xi0 - u) e^{w t}``.  With ``tau = e^{w t_rem}`` (timing
variable), the DCM at end-of-step is ``xi_eos = u + (xi_meas - u) tau``.  For
balance after stepping, the next footstep ``u_next`` must sit a nominal offset
``b`` behind the end-of-step DCM:

    u_next + b = u + (xi_meas - u) * tau                       (DCM constraint)

This equation is *linear* in ``[u_next, tau]``, so we solve a tiny 3-variable QP
each tick. A push drives xi_meas outward; the QP both shifts u_next toward it
and, when the footstep alone can't reach, *shrinks* tau (steps sooner).

Nominal offsets for a periodic gait:
    b_x = L / (tau_nom - 1)          (sagittal, step length L)
    b_y = +/- W / (tau_nom + 1)      (lateral, foot-separation W; sign per stance)

Ported from kanellos_progress/biped/step_timing.py — same algorithm, same solver.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from qpsolvers import solve_qp


@dataclasses.dataclass
class StepTimingGains:
    w_foot: float = 1.0      # stay near the nominal footstep
    w_time: float = 4.0      # stay near the nominal timing (prefer foot motion)
    max_dx: float = 0.16     # +/- reachable footstep range about nominal [m]
    max_dy: float = 0.10
    t_min: float = 0.12      # soonest a step may land from now [s]
    t_max: float = 1.20      # latest a step may land from now [s]


class StepTimingQP:
    """Online step-timing + footstep-placement QP.

    Parameters
    ----------
    omega:
        sqrt(g / z_com) — natural DCM frequency [rad/s].
    T_nom:
        Nominal single-support duration [s] (= gait.t_ss).
    gains:
        Cost weights and kinematic limits (StepTimingGains).
    """

    def __init__(self, omega: float, T_nom: float, gains: StepTimingGains = None):
        self.w = float(omega)
        self.T_nom = float(T_nom)
        self.g = gains or StepTimingGains()

    def nominal_offset(self, L: float, W: float, stance_side: str) -> np.ndarray:
        """Nominal DCM offset b for a periodic gait (start-of-step convention).

        Parameters
        ----------
        L: step length [m]  W: lateral foot separation [m]
        stance_side: 'left' or 'right'
        """
        tau = np.exp(self.w * self.T_nom)
        b_x = L / (tau - 1.0)
        s = 1.0 if stance_side == "right" else -1.0   # toward swing side
        b_y = s * W / (tau + 1.0)
        return np.array([b_x, b_y])

    def solve(self, xi, u_cur, t_elapsed: float, u_nom, b_nom):
        """Return (u_next [xy], t_remaining [s]).

        Parameters
        ----------
        xi:        measured DCM [x, y] (2,)
        u_cur:     current stance foot position [x, y] (2,)
        t_elapsed: seconds elapsed since SS started [s]
        u_nom:     nominal swing landing position [x, y] (2,) from the plan
        b_nom:     nominal DCM offset (2,) from ``nominal_offset()``
        """
        w, gains = self.w, self.g
        xi = np.asarray(xi, float)
        u_cur = np.asarray(u_cur, float)
        u_nom = np.asarray(u_nom, float)
        d = xi - u_cur      # DCM-minus-stance-foot: grows outward under a push

        t_rem_nom = max(self.T_nom - t_elapsed, gains.t_min)
        tau_nom = np.exp(w * t_rem_nom)
        tau_min = np.exp(w * gains.t_min)
        tau_max = np.exp(w * gains.t_max)

        # decision variables z = [u_next_x, u_next_y, tau]
        P = np.diag([2 * gains.w_foot, 2 * gains.w_foot, 2 * gains.w_time])
        q = np.array([-2 * gains.w_foot * u_nom[0],
                      -2 * gains.w_foot * u_nom[1],
                      -2 * gains.w_time * tau_nom])
        # equality: u_next - d*tau = u_cur - b_nom
        A = np.array([[1.0, 0.0, -d[0]],
                      [0.0, 1.0, -d[1]]])
        bvec = u_cur - np.asarray(b_nom, float)
        lb = np.array([u_nom[0] - gains.max_dx, u_nom[1] - gains.max_dy, tau_min])
        ub = np.array([u_nom[0] + gains.max_dx, u_nom[1] + gains.max_dy, tau_max])

        z = solve_qp(P, q, A=A, b=bvec, lb=lb, ub=ub, solver="osqp",
                     eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, verbose=False)
        if z is None:
            return u_nom.copy(), t_rem_nom   # fallback: nominal plan

        u_next = z[:2]
        tau = float(np.clip(z[2], tau_min, tau_max))
        t_remaining = np.log(tau) / w
        return u_next, t_remaining
