"""MuJoCo environment wrapper: load, reset, step, apply torques, viewer."""
from pathlib import Path
import mujoco
import numpy as np

from src.utils.config import load_params, resolve


class MujocoEnv:
    """Thin wrapper around an MjModel/MjData pair driven by joint torques."""

    def __init__(self, scene_path: str | Path = None, keyframe: str | None = None,
                 terrain=None, model=None):
        # Load via mjSpec so a Terrain can mutate the model at load time
        # (tilt the floor, add stair boxes / obstacles). terrain=None => flat XML.
        # A prebuilt ``model`` (e.g. from robots.load_robot_model) bypasses loading.
        self.terrain = terrain
        if model is not None:
            self.model = model
        elif terrain is not None:
            spec = mujoco.MjSpec.from_file(str(resolve(scene_path)))
            terrain.apply(spec)
            self.model = spec.compile()
        else:
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


def make_env_from_params(scene_key: str = "scene_flat", terrain=None) -> MujocoEnv:
    params = load_params()
    return MujocoEnv(params["model"][scene_key], keyframe=params["model"]["keyframe"],
                     terrain=terrain)


def make_robot_env(robot: str = "g1", terrain=None, scene_key: str = "scene_flat",
                   decorate=None):
    """Robot-agnostic env factory (G1 or Talos). Returns (env, mcfg).

    G1 uses its baked 'stand' crouch keyframe; Talos is loaded with contact-corner
    sites injected and placed into a bent-knee crouch with the feet on the ground.
    ``decorate(spec, mcfg)`` optionally attaches extra geoms (e.g. tray + frappe).
    """
    from src.sim.robots import load_robot_model
    params = load_params()
    model, mcfg = load_robot_model(robot, params, terrain=terrain, scene_key=scene_key,
                                   decorate=decorate)
    env = MujocoEnv(model=model, keyframe=mcfg["keyframe"], terrain=terrain)
    if robot == "talos":
        _init_talos_crouch(env, mcfg)
    return env, mcfg


def _init_talos_crouch(env, mcfg):
    """Put Talos into a bent-knee crouch with feet flat on the ground."""
    m, d = env.model, env.data
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    for j, ang in mcfg["crouch"].items():
        adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        d.qpos[adr] = ang
    mujoco.mj_forward(m, d)
    # drop the base so the lowest foot corner just touches the ground (z=0.005)
    corners = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, c)
               for side in ("left", "right") for c in mcfg["feet"][side]["corners"]]
    d.qpos[2] -= (min(d.site_xpos[c][2] for c in corners) - 0.005)
    mujoco.mj_forward(m, d)
