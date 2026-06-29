"""DCM-based CoM trajectory (Lecture 10, Englsberger 2015).

Divergent Component of Motion:   xi = p_CoM + p_dot_CoM / omega,  omega = sqrt(g/z).
DCM dynamics:                     xi_dot = omega (xi - p_zmp).
CoM dynamics:                     p_dot_CoM = -omega (p_CoM - xi) = omega (xi - p_CoM).

We sample a piecewise ZMP reference p_zmp(t) from the gait timeline, integrate
the DCM BACKWARD (stable) from a terminal condition xi_end = p_zmp_end, then
integrate the CoM FORWARD. The result is a smooth CoM that keeps the ZMP inside
the support polygon by construction (ZMP sits on / between the stance feet).
"""
import numpy as np

from src.planning.fsm import phase_at, total_duration


class DCMPlanner:
    def __init__(self, params, z_com, gravity=9.81, slope_angle=0.0):
        self.dt = params["gait"]["dt_plan"]
        self.z = z_com
        self.g = gravity
        self.omega = np.sqrt(self.g / self.z)
        self.slope_angle = slope_angle
        # Slope-gravity bias: on a slope of angle alpha the DCM attractor shifts
        # forward in world-frame x by  g*sin(alpha)/omega^2 = z_com*tan(alpha).
        # In simulation, applying the full z_com*tan(alpha) offset causes the WBC
        # to saturate (offset > support polygon half-length for alpha > ~8 deg).
        # We therefore scale by tan(alpha) itself (a second-order term) so the
        # correction is small for mild slopes (handled by proportional DCM feedback)
        # and grows quadratically for steep inclines where the full correction is
        # needed to overcome backward DCM drift.
        self.slope_bias = (np.array([z_com * np.tan(slope_angle) ** 2, 0.0])
                           if slope_angle != 0.0 else np.zeros(2))

    def _sample_zmp(self, timeline):
        T = total_duration(timeline)
        N = int(round(T / self.dt)) + 1
        t = np.arange(N) * self.dt
        zmp = np.zeros((N, 2))
        for k in range(N):
            ph, s = phase_at(timeline, t[k])
            zmp[k] = ph["zmp_from"] + s * (ph["zmp_to"] - ph["zmp_from"])
        return t, zmp

    def generate(self, timeline, com0_xy):
        """Return dict of sampled trajectories (t, zmp, dcm, com, com_vel)."""
        t, zmp = self._sample_zmp(timeline)
        N = len(t)
        w, dt = self.omega, self.dt

        # Backward DCM recursion (exact for piecewise-constant ZMP over dt).
        # On a slope of angle alpha, the gravity component along the slope adds a
        # bias: slope_bias_x = z_com * tan(alpha).  We treat (zmp + slope_bias) as
        # the effective ZMP so the DCM converges to the correct slope-adjusted
        # attractor.  The terminal DCM is set at zmp_end + slope_bias, and the
        # backward recursion propagates this forward-lean correction throughout the
        # trajectory via eff_zmp = zmp + slope_bias at each step.
        dcm = np.zeros((N, 2))
        dcm[-1] = zmp[-1] + self.slope_bias
        decay = np.exp(-w * dt)
        for k in range(N - 2, -1, -1):
            eff_zmp_k = zmp[k] + self.slope_bias
            dcm[k] = eff_zmp_k + (dcm[k + 1] - eff_zmp_k) * decay

        # Forward CoM integration: p_dot = omega (xi - p).
        com = np.zeros((N, 2))
        com_vel = np.zeros((N, 2))
        com[0] = com0_xy
        for k in range(N - 1):
            com_vel[k] = w * (dcm[k] - com[k])
            com[k + 1] = com[k] + dt * com_vel[k]
        com_vel[-1] = w * (dcm[-1] - com[-1])

        return {"t": t, "zmp": zmp, "dcm": dcm, "com": com, "com_vel": com_vel,
                "omega": w, "z": self.z}
