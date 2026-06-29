"""Stage 1 -- gravity compensation (static Inverse Dynamics).

At static equilibrium (v = 0, v̇ = 0) the floating-base dynamics reduce to
    Sᵀ τ + Σ_c J_cᵀ f_c = h ,      h = qfrc_bias = g(q) .
The 6 unactuated base rows must be met by contact forces alone:
    (J_cᵀ f)[base] = h[base] .
We choose contact forces closest to an even vertical weight distribution
(equality-constrained least squares), then read off the joint torques from
the actuated rows:
    τ = (h − J_cᵀ f)[actuated dofs] .

Recomputed every control step from the current state, so it is quasi-static
feedback and naturally rejects small numerical drift.
"""
import numpy as np


class GravityCompensator:
    def __init__(self, terms, total_mass, gravity=9.81, reg=1e-6,
                 q_nom=None, hold_kp=0.0, hold_kd=0.0):
        self.terms = terms
        self.weight = total_mass * gravity
        self.reg = reg
        # Optional posture hold (Stage 1 'hold_posture'): a joint-space PD that
        # keeps the robot in its nominal pose. Needed to hold a bent-knee crouch
        # still (pure feedforward gravity comp would slowly drift there).
        self.q_nom = q_nom
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd
        if q_nom is not None:
            self.qadr = np.array([terms.model.jnt_qposadr[terms.model.dof_jntid[d]]
                                  for d in terms.act_dof])

    def compute(self, data, active_site_ids=None):
        """Return (tau [nu], f [3*ncp]) holding the robot in static balance."""
        T = self.terms
        h = T.bias(data)                                   # (nv,)
        Jc = T.contact_jacobian(data, active_site_ids)     # (3*ncp, nv)
        ncp = Jc.shape[0] // 3

        base = np.arange(6)                                # unactuated base dofs
        A = Jc[:, base].T                                  # (6, 3*ncp)
        b = h[base]                                        # (6,)

        # Nominal: support total weight evenly, vertical (world +z).
        f0 = np.zeros(3 * ncp)
        f0[2::3] = self.weight / max(ncp, 1)

        # Equality-constrained least squares: min ||f - f0|| s.t. A f = b.
        # f = f0 + Aᵀ (A Aᵀ + reg I)^-1 (b - A f0).
        AAT = A @ A.T + self.reg * np.eye(A.shape[0])
        lam = np.linalg.solve(AAT, b - A @ f0)
        f = f0 + A.T @ lam

        # Actuated rows give the joint torques.
        gen_contact = Jc.T @ f                             # (nv,)
        tau = (h - gen_contact)[T.act_dof]                 # (nu,)

        if self.q_nom is not None and (self.hold_kp or self.hold_kd):
            q = data.qpos[self.qadr]
            dq = data.qvel[T.act_dof]
            tau = tau + self.hold_kp * (self.q_nom - q) + self.hold_kd * (-dq)
        return tau, f

    def residual(self, data, tau, f, active_site_ids=None):
        """Dynamics residual ‖Sᵀτ + Jᵀf − h‖ (should be ~0)."""
        T = self.terms
        h = T.bias(data)
        Jc = T.contact_jacobian(data, active_site_ids)
        lhs = T.S.T @ tau + Jc.T @ f
        return np.linalg.norm(lhs - h)
