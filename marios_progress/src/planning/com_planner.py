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
    def __init__(self, params, z_com, gravity=9.81):
        self.dt = params["gait"]["dt_plan"]
        self.z = z_com
        self.g = gravity
        self.omega = np.sqrt(self.g / self.z)

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
        dcm = np.zeros((N, 2))
        dcm[-1] = zmp[-1]
        decay = np.exp(-w * dt)
        for k in range(N - 2, -1, -1):
            dcm[k] = zmp[k] + (dcm[k + 1] - zmp[k]) * decay

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
