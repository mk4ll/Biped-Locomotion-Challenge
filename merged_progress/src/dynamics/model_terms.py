"""Floating-base dynamics terms extracted from MuJoCo.

Provides, per control step:
  M(q)          mass matrix              (nv x nv)   via mj_fullM
  h(q,v)        bias = C(q,v)v + g(q)    (nv,)       via data.qfrc_bias
  S             selection matrix         (nu x nv)
  J_c           stacked contact Jacobian (3*ncp x nv) at foot corner points
  Jdot_v        (J̇_c v)                 (3*ncp,)    drift term

All quantities follow:  M v̇ + h = Sᵀ τ + Σ_c J_cᵀ f_c .
Contact points are the foot corner SITES (world-frame 3D point contacts).
"""
import mujoco
import numpy as np


class ModelTerms:
    def __init__(self, model, contact_sites):
        """contact_sites: list of site names used as point contacts."""
        self.model = model
        self.nv = model.nv
        self.nu = model.nu
        self.site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                         for s in contact_sites]
        self.site_names = list(contact_sites)
        # Selection matrix (actuator -> dof).
        self.S = np.zeros((self.nu, self.nv))
        self.act_dof = np.empty(self.nu, dtype=int)
        for a in range(self.nu):
            dof = model.jnt_dofadr[model.actuator_trnid[a, 0]]
            self.S[a, dof] = 1.0
            self.act_dof[a] = dof
        # Reusable buffers for mj_jac.
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    # -- inertia / bias -------------------------------------------------------
    def mass_matrix(self, data) -> np.ndarray:
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, data.qM)
        return M

    def bias(self, data) -> np.ndarray:
        # qfrc_bias = C(q,v) v + g(q)  (Coriolis/centrifugal + gravity).
        return data.qfrc_bias.copy()

    # -- contacts -------------------------------------------------------------
    def site_jacobian(self, data, site_id) -> np.ndarray:
        """Translational Jacobian (3 x nv) of a site in world frame."""
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, site_id)
        return self._jacp.copy()

    def contact_jacobian(self, data, active_site_ids=None) -> np.ndarray:
        """Stack 3xnv translational Jacobians of the given contact sites.

        active_site_ids: subset of self.site_ids in contact (default: all).
        Returns (3*n x nv).
        """
        ids = self.site_ids if active_site_ids is None else active_site_ids
        if len(ids) == 0:
            return np.zeros((0, self.nv))
        return np.vstack([self.site_jacobian(data, sid) for sid in ids])

    def _site_accels(self, data, site_ids):
        """Raw spatial site accelerations [ang; lin] with q̈=0 AND gravity=0.

        Spatial site acceleration = J q̈ + J̇ v (+ gravity offset). MuJoCo's
        mj_objectAcceleration includes the world gravity offset in cacc, so we
        must zero BOTH q̈ and gravity to isolate the pure convective term J̇ v.
        Returns list of length-6 arrays [ang(3); lin(3)] (MuJoCo ordering).
        """
        saved_qacc = data.qacc.copy()
        saved_g = self.model.opt.gravity.copy()
        data.qacc[:] = 0.0
        self.model.opt.gravity[:] = 0.0
        mujoco.mj_rnePostConstraint(self.model, data)
        out = []
        acc = np.zeros(6)
        for sid in site_ids:
            mujoco.mj_objectAcceleration(self.model, data, mujoco.mjtObj.mjOBJ_SITE,
                                         sid, acc, 0)  # world frame; [ang; lin]
            out.append(acc.copy())
        data.qacc[:] = saved_qacc
        self.model.opt.gravity[:] = saved_g
        mujoco.mj_rnePostConstraint(self.model, data)  # restore cacc
        return out

    def contact_jdot_v(self, data, active_site_ids=None) -> np.ndarray:
        """Drift term J̇_c v (linear) for each contact point (3*n,). v=0 -> 0."""
        ids = self.site_ids if active_site_ids is None else active_site_ids
        if len(ids) == 0:
            return np.zeros(0)
        return np.concatenate([a[3:6] for a in self._site_accels(data, ids)])

    def site_pos(self, data, site_id) -> np.ndarray:
        return data.site_xpos[site_id].copy()

    # -- full 6D site Jacobian / drift (for rigid-foot contact constraint) -----
    def site_jacobian6(self, data, site_id) -> np.ndarray:
        """6D site Jacobian [lin(3); ang(3)] x nv."""
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, site_id)
        return np.vstack([self._jacp.copy(), self._jacr.copy()])

    def site_jdot_v6(self, data, site_ids) -> np.ndarray:
        """Stacked 6D drift J̇v = [lin; ang] per site (6*n,). v=0 -> 0."""
        if len(site_ids) == 0:
            return np.zeros(0)
        # _site_accels returns [ang; lin]; reorder to [lin; ang] to match jac6.
        return np.concatenate([np.concatenate([a[3:6], a[0:3]])
                               for a in self._site_accels(data, site_ids)])
