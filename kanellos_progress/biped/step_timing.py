"""Step timing + footstep adjustment QP (capture-point, Khadiv et al. 2016).

The classic DCM controller fixes *where* and *when* the next step lands ahead of
time.  Under a hard lateral push that is not enough: the divergent component of
motion (DCM) runs away faster than a pre-planned step can catch.  The fix is to
make **both** the next footstep location and the **step timing** decision
variables, solved online every tick from the *measured* DCM.

Model.  During single support with the centre of pressure held at the stance
foot ``u``, the DCM obeys ``xi(t) = u + (xi0 - u) e^{w t}``.  With ``t`` the time
still remaining in the step and ``tau = e^{w t}`` (the "timing" variable), the
DCM at the end of the step is ``xi_eos = u + (xi_meas - u) tau``.  For the robot
to be balanced after stepping, the next footstep ``u_next`` must sit a nominal
offset ``b`` behind the end-of-step DCM:

    u_next + b = u + (xi_meas - u) * tau                         (DCM constraint)

That equation is *linear* in the unknowns ``[u_next, tau]`` (everything else is
measured), so we solve a tiny QP each tick:

    min  w_foot ||u_next - u_nom||^2 + w_time (tau - tau_nom)^2
    s.t. u_next - (xi_meas - u) tau = u - b                      (x and y rows)
         u_next in a kinematically reachable box
         tau in [tau_min, tau_max]                               (timing limits)

A push drives ``xi_meas`` outward; the QP both shifts ``u_next`` toward it and,
when the footstep alone can't reach, *shrinks* ``tau`` (steps down sooner).  The
recovered timing is ``t_remaining = ln(tau) / w``.

Nominal offsets for a periodic gait (derived from the DCM recursion):
    b_x = L / (tau_nom - 1)            (sagittal, step length L)
    b_y = +/- W / (tau_nom + 1)        (lateral, stance width W; sign per stance)
"""
from __future__ import annotations

import dataclasses

import numpy as np
from qpsolvers import solve_qp


@dataclasses.dataclass
class StepTimingGains:
    w_foot: float = 1.0      # stay near the nominal footstep
    w_time: float = 4.0      # stay near the nominal timing (prefer foot motion)
    max_dx: float = 0.16     # +/- reachable footstep range about nominal (m)
    max_dy: float = 0.10
    t_min: float = 0.12      # soonest a step may land from now (s)
    t_max: float = 1.20      # latest a step may land from now (s)


class StepTimingQP:
    def __init__(self, omega: float, T_nom: float, gains: StepTimingGains = None):
        self.w = float(omega)
        self.T_nom = float(T_nom)
        self.g = gains or StepTimingGains()

    def nominal_offset(self, L: float, W: float, stance_side: str):
        """Nominal DCM offset b for a periodic gait (start-of-step convention)."""
        tau = np.exp(self.w * self.T_nom)
        b_x = L / (tau - 1.0)
        # lateral offset points toward the swing side (centre); sign by stance
        s = 1.0 if stance_side == "right" else -1.0
        b_y = s * W / (tau + 1.0)
        return np.array([b_x, b_y])

    def solve(self, xi, u_cur, t_elapsed, u_nom, b_nom):
        """Return (u_next xy, t_remaining s).

        xi, u_cur, u_nom, b_nom : (2,) arrays.  t_elapsed : seconds into the step.
        """
        w, g = self.w, self.g
        xi = np.asarray(xi, float)
        u_cur = np.asarray(u_cur, float)
        u_nom = np.asarray(u_nom, float)
        d = xi - u_cur                      # known DCM-minus-CoP vector

        # nominal timing for the *remaining* time
        t_rem_nom = max(self.T_nom - t_elapsed, g.t_min)
        tau_nom = np.exp(w * t_rem_nom)
        tau_min = np.exp(w * g.t_min)
        tau_max = np.exp(w * g.t_max)

        # variables z = [u_next_x, u_next_y, tau]
        P = np.diag([2 * g.w_foot, 2 * g.w_foot, 2 * g.w_time])
        q = np.array([-2 * g.w_foot * u_nom[0],
                      -2 * g.w_foot * u_nom[1],
                      -2 * g.w_time * tau_nom])
        # equality: u_next - d*tau = u_cur - b_nom
        A = np.array([[1.0, 0.0, -d[0]],
                      [0.0, 1.0, -d[1]]])
        bvec = u_cur - np.asarray(b_nom, float)
        lb = np.array([u_nom[0] - g.max_dx, u_nom[1] - g.max_dy, tau_min])
        ub = np.array([u_nom[0] + g.max_dx, u_nom[1] + g.max_dy, tau_max])

        z = solve_qp(P, q, A=A, b=bvec, lb=lb, ub=ub, solver="osqp",
                     eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, verbose=False)
        if z is None:                       # fall back to the nominal plan
            return u_nom.copy(), t_rem_nom
        u_next = z[:2]
        tau = float(np.clip(z[2], tau_min, tau_max))
        t_remaining = np.log(tau) / w
        return u_next, t_remaining
