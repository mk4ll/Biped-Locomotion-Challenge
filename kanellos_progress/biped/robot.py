"""Humanoid model loading and a thin state/kinematics wrapper (G1 and Talos).

The Menagerie G1 ships with *position* actuators (PD servos); for a torque-based
whole-body QP we rewrite every actuator to a unit-gear ``motor`` via ``mjSpec``.
Talos already ships a torque (``motor``) scene, so no conversion is needed there.
In both cases per-joint torque limits (``jnt_actfrcrange``) are copied onto the
actuators (motors do not inherit them).

A :class:`RobotConfig` captures everything that differs between robots -- scene
path, the floating-base/root body, foot bodies/sites, joint-name groups, the
crouch pose, the home keyframe and the foot collision-geometry kind -- so the
controllers, planner and WBC are entirely robot-agnostic.

``Robot`` is a stateless-ish facade over ``(MjModel, MjData)`` exposing the
quantities the controllers need: mass matrix, bias forces, CoM and CoM Jacobian,
foot poses/Jacobians (and their time-derivative products), contact points, and
named index groups for legs / waist / arms / extras.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

import mujoco
import numpy as np


# --------------------------------------------------------------------------- #
# Robot configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RobotConfig:
    name: str
    scene_path: str
    root_body: str                       # floating-base body (CoM/orientation)
    foot_body: Dict[str, str]            # side -> ankle/foot body name
    foot_site: Dict[str, str]            # side -> foot reference site name
    leg_joints: Dict[str, List[str]]     # side -> ordered leg joint names
    arm_joints: Dict[str, List[str]]     # side -> ordered arm joint names
    waist_joints: List[str]              # waist / torso joints
    extra_joints: List[str]              # other actuated joints to hold (head...)
    leg_crouch: Dict[int, float]         # leg-group index -> crouch angle
    keyframe: object = 0                 # home keyframe (name or id)
    convert_to_motor: bool = True        # rewrite position servos -> torque motors
    foot_geom: str = "sphere"            # "sphere" (corner geoms) or "box"
    # arm swing (contralateral hip coupling); leg/arm-group *indices*
    arm_swing_enabled: bool = True
    arm_swing_idx: int = 0               # arm-group index of the swing joint
    hip_pitch_idx: int = 0               # leg-group index of hip pitch


# ---- G1 (Unitree, 29 DoF, position servos -> torque) ----------------------
_G1_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_G1_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
           "wrist_roll", "wrist_pitch", "wrist_yaw"]
_G1_WAIST = ["waist_yaw", "waist_roll", "waist_pitch"]

G1_CONFIG = RobotConfig(
    name="g1",
    scene_path="mujoco_menagerie/unitree_g1/scene.xml",
    root_body="pelvis",
    foot_body={"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"},
    foot_site={"left": "left_foot", "right": "right_foot"},
    leg_joints={s: [f"{s}_{j}_joint" for j in _G1_LEG] for s in ("left", "right")},
    arm_joints={s: [f"{s}_{j}_joint" for j in _G1_ARM] for s in ("left", "right")},
    waist_joints=[f"{w}_joint" for w in _G1_WAIST],
    extra_joints=[],
    leg_crouch={0: -0.40, 3: 0.80, 4: -0.42},   # hip_pitch, knee, ankle_pitch
    keyframe="stand",
    convert_to_motor=True,
    foot_geom="sphere",
    arm_swing_enabled=True,
    arm_swing_idx=0,        # shoulder_pitch
    hip_pitch_idx=0,        # hip_pitch
)

# ---- Talos (PAL Robotics, 32 actuated DoF, ships torque motors) ------------
TALOS_CONFIG = RobotConfig(
    name="talos",
    scene_path="mujoco_menagerie/pal_talos/scene_motor.xml",
    root_body="base_link",
    foot_body={"left": "leg_left_6_link", "right": "leg_right_6_link"},
    foot_site={"left": "left_foot", "right": "right_foot"},
    leg_joints={s: [f"leg_{s}_{i}_joint" for i in range(1, 7)]
                for s in ("left", "right")},
    arm_joints={s: [f"arm_{s}_{i}_joint" for i in range(1, 8)]
                for s in ("left", "right")},
    waist_joints=["torso_1_joint", "torso_2_joint"],
    extra_joints=["head_1_joint", "head_2_joint",
                  "gripper_left_joint", "gripper_right_joint"],
    # leg order: 1 hip_yaw, 2 hip_roll, 3 hip_pitch, 4 knee, 5 ankle_pitch, 6 ankle_roll
    leg_crouch={2: -0.40, 3: 0.80, 4: -0.42},
    keyframe=0,
    convert_to_motor=False,    # scene_motor.xml is already torque-controlled
    foot_geom="box",
    arm_swing_enabled=False,   # Talos shoulder axes don't map to a clean pitch swing
)


CONFIGS = {"g1": G1_CONFIG, "talos": TALOS_CONFIG}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _build_spec(config: RobotConfig, terrain=None) -> "mujoco.MjSpec":
    """Load the scene, optionally convert actuators to torque motors, add terrain."""
    spec = mujoco.MjSpec.from_file(config.scene_path)
    if config.convert_to_motor:
        for a in spec.actuators:
            a.set_to_motor()  # force = ctrl, gear = 1
    if terrain is not None:
        terrain.apply(spec)
    return spec


def load_model(config: RobotConfig, terrain=None):
    """Return a compiled (model, data) torque-controlled robot on the terrain."""
    spec = _build_spec(config, terrain)
    model = spec.compile()
    data = mujoco.MjData(model)
    # Motors don't inherit jnt_actfrcrange, so re-apply per-actuator force limits.
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        lo, hi = model.jnt_actfrcrange[jid]
        if hi > lo:
            model.actuator_forcerange[i] = [lo, hi]
            model.actuator_forcelimited[i] = 1
    return model, data


def load_g1(scene_path: str = None, terrain=None):
    """Return a compiled (model, data) torque-controlled G1 on the given terrain."""
    cfg = G1_CONFIG if scene_path is None else dataclasses.replace(
        G1_CONFIG, scene_path=scene_path)
    return load_model(cfg, terrain)


def load_talos(terrain=None):
    """Return a compiled (model, data) torque-controlled Talos on the terrain."""
    return load_model(TALOS_CONFIG, terrain)


# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class JointGroups:
    """Index maps for a set of joints (in MjModel ordering).

    ``qadr``  : index into qpos (configuration)
    ``vadr``  : index into qvel / qacc / generalized forces
    ``actid`` : actuator id driving each joint (or -1 if unactuated)
    """
    name: List[str]
    qadr: np.ndarray
    vadr: np.ndarray
    actid: np.ndarray


class Robot:
    def __init__(self, model, data, config: RobotConfig = G1_CONFIG):
        self.m = model
        self.d = data
        self.config = config
        self.nv = model.nv
        self.nu = model.nu

        self.pelvis_id = model.body(config.root_body).id   # floating-base body
        self.foot_body = {s: model.body(config.foot_body[s]).id
                          for s in ("left", "right")}
        self.foot_site = {s: model.site(config.foot_site[s]).id
                          for s in ("left", "right")}

        # Actuated dofs, in actuator order: act_vadr[k] is the dof of actuator k.
        # (Do not assume "everything past the floating base" -- Talos has passive
        #  coupled gripper joints, so the actuated set is a strict subset.)
        self._dof_to_actid = self._actuator_for_dof()
        self.act_vadr = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]]
                                  for i in range(self.nu)])
        # truly unactuated dofs (floating base; coupled joints are driven by
        # equality forces and are *not* part of the WBC underactuation rows).
        self.base_vadr = np.arange(6)

        self.groups = self._index_groups()
        if config.foot_geom == "box":
            self.foot_box_geom = self._foot_box_geoms()
            self.foot_corner_geoms = None
        else:
            self.foot_corner_geoms = self._foot_corner_geoms()
            self.foot_box_geom = None
        self.torque_limit = self._torque_limits()

    # ---- index helpers -------------------------------------------------
    def _joint_ids(self, names):
        qadr, vadr, actid = [], [], []
        for n in names:
            j = self.m.joint(n)
            d = j.dofadr[0]
            qadr.append(j.qposadr[0])
            vadr.append(d)
            actid.append(self._dof_to_actid.get(d, -1))
        return JointGroups(list(names),
                           np.array(qadr), np.array(vadr), np.array(actid))

    def _index_groups(self):
        cfg = self.config
        g = {}
        for side in ("left", "right"):
            g[f"leg_{side}"] = self._joint_ids(cfg.leg_joints[side])
            g[f"arm_{side}"] = self._joint_ids(cfg.arm_joints[side])
        g["waist"] = self._joint_ids(cfg.waist_joints)
        if cfg.extra_joints:
            g["extra"] = self._joint_ids(cfg.extra_joints)
        return g

    def _actuator_for_dof(self):
        mp = {}
        for i in range(self.nu):
            jid = self.m.actuator_trnid[i, 0]
            mp[self.m.jnt_dofadr[jid]] = i
        return mp

    def _foot_corner_geoms(self):
        """Sphere collision geoms attached to each ankle link (G1-style feet)."""
        out = {"left": [], "right": []}
        for side in ("left", "right"):
            bid = self.foot_body[side]
            for gi in range(self.m.ngeom):
                if (self.m.geom_bodyid[gi] == bid
                        and self.m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_SPHERE):
                    out[side].append(gi)
        return out

    def _foot_box_geoms(self):
        """The collision box geom on each foot link (Talos-style box feet)."""
        out = {}
        for side in ("left", "right"):
            bid = self.foot_body[side]
            found = None
            for gi in range(self.m.ngeom):
                if (self.m.geom_bodyid[gi] == bid
                        and self.m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX
                        and self.m.geom_contype[gi]):
                    found = gi
                    break
            if found is None:
                raise RuntimeError(f"no collision box geom on foot body {side}")
            out[side] = found
        return out

    def _torque_limits(self):
        lim = np.zeros(self.nu)
        for i in range(self.nu):
            jid = self.m.actuator_trnid[i, 0]
            hi = self.m.jnt_actfrcrange[jid, 1]
            lim[i] = hi if hi > 0 else 1e3
        return lim

    # ---- home / nominal pose ------------------------------------------
    def keyframe_qpos(self) -> np.ndarray:
        return self.m.key(self.config.keyframe).qpos.copy()

    # ---- dynamics ------------------------------------------------------
    def forward(self):
        mujoco.mj_forward(self.m, self.d)

    def mass_matrix(self) -> np.ndarray:
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M

    def bias(self) -> np.ndarray:
        """h(q, v): Coriolis + centrifugal + gravity (generalized forces)."""
        return self.d.qfrc_bias.copy()

    # ---- centre of mass ------------------------------------------------
    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.pelvis_id].copy()

    def com_jacobian(self) -> np.ndarray:
        J = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, J, self.pelvis_id)
        return J

    def com_vel(self, Jcom: Optional[np.ndarray] = None) -> np.ndarray:
        if Jcom is None:
            Jcom = self.com_jacobian()
        return Jcom @ self.d.qvel

    def body_angular_jacobian(self, body_id: int) -> np.ndarray:
        Jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.m, self.d, None, Jr, body_id)
        return Jr

    def pelvis_quat(self) -> np.ndarray:
        return self.d.xquat[self.pelvis_id].copy()

    @property
    def total_mass(self) -> float:
        return float(self.m.body_subtreemass[self.pelvis_id])

    # ---- feet ----------------------------------------------------------
    def foot_pos(self, side: str) -> np.ndarray:
        return self.d.site_xpos[self.foot_site[side]].copy()

    def foot_rot(self, side: str) -> np.ndarray:
        return self.d.site_xmat[self.foot_site[side]].reshape(3, 3).copy()

    def site_jacobian(self, side: str):
        Jp = np.zeros((3, self.nv))
        Jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.m, self.d, Jp, Jr, self.foot_site[side])
        return Jp, Jr

    def point_jacobian(self, point: np.ndarray, body_id: int) -> np.ndarray:
        Jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, Jp, None, point, body_id)
        return Jp

    def point_jacobian_dot(self, point: np.ndarray, body_id: int) -> np.ndarray:
        """Time-derivative of the translational point Jacobian (for Jdot*v)."""
        Jdp = np.zeros((3, self.nv))
        mujoco.mj_jacDot(self.m, self.d, Jdp, None, point, body_id)
        return Jdp

    def foot_corner_points(self, side: str) -> np.ndarray:
        """World positions of the 4 sole contact points for one foot (4x3).

        Sphere feet (G1): the sphere geom centres.  Box feet (Talos): the four
        bottom corners of the collision box, transformed to world.
        """
        if self.foot_box_geom is not None:
            gid = self.foot_box_geom[side]
            sx, sy, sz = self.m.geom_size[gid]
            pos = self.d.geom_xpos[gid]
            R = self.d.geom_xmat[gid].reshape(3, 3)
            local = np.array([[sx, sy, -sz], [sx, -sy, -sz],
                              [-sx, sy, -sz], [-sx, -sy, -sz]])
            return pos + local @ R.T
        return self.d.geom_xpos[self.foot_corner_geoms[side]].copy()

    # ---- contact sensing ----------------------------------------------
    def foot_contact_force(self, side: str) -> float:
        """Total normal contact force magnitude under one foot."""
        bid = self.foot_body[side]
        total = 0.0
        f6 = np.zeros(6)
        for c in range(self.d.ncon):
            con = self.d.contact[c]
            b1 = self.m.geom_bodyid[con.geom1]
            b2 = self.m.geom_bodyid[con.geom2]
            if bid in (b1, b2):
                mujoco.mj_contactForce(self.m, self.d, c, f6)
                total += abs(f6[0])  # normal is first component in contact frame
        return total
