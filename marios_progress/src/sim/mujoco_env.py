"""MuJoCo environment wrapper: load, reset, step, apply torques, viewer."""
from pathlib import Path
import mujoco
import numpy as np

from src.utils.config import load_params, resolve


class MujocoEnv:
    """Thin wrapper around an MjModel/MjData pair driven by joint torques."""

    def __init__(self, scene_path: str | Path, keyframe: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(resolve(scene_path)))
        self.data = mujoco.MjData(self.model)
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.keyframe = keyframe
        # Selection matrix S (nu x nv): actuator a drives dof of its joint.
        self.S = self._build_selection_matrix()
        # Map actuator index -> dof index (for fast torque scatter).
        self.act_dof = np.array(
            [self.model.jnt_dofadr[self.model.actuator_trnid[a, 0]]
             for a in range(self.nu)], dtype=int)
        self.reset()

    # -- construction helpers -------------------------------------------------
    def _build_selection_matrix(self) -> np.ndarray:
        S = np.zeros((self.nu, self.nv))
        for a in range(self.nu):
            jnt = self.model.actuator_trnid[a, 0]
            S[a, self.model.jnt_dofadr[jnt]] = 1.0
        return S

    # -- lifecycle ------------------------------------------------------------
    def reset(self):
        if self.keyframe is not None:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe)
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def set_torques(self, tau: np.ndarray):
        """tau is length nu (joint torques in actuator order). Clamped to ctrlrange."""
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(tau, lo, hi)

    def step(self, tau: np.ndarray | None = None):
        if tau is not None:
            self.set_torques(tau)
        mujoco.mj_step(self.model, self.data)
        return self.data

    def apply_external_force(self, body_id, force_world, torque_world=None):
        """Apply an external wrench (world frame) on a body via xfrc_applied."""
        self.data.xfrc_applied[body_id, :3] = force_world
        if torque_world is not None:
            self.data.xfrc_applied[body_id, 3:] = torque_world

    def clear_external_forces(self):
        self.data.xfrc_applied[:] = 0.0

    @property
    def dt(self) -> float:
        return self.model.opt.timestep


def make_env_from_params(scene_key: str = "scene_flat") -> MujocoEnv:
    params = load_params()
    return MujocoEnv(params["model"][scene_key], keyframe=params["model"]["keyframe"])
