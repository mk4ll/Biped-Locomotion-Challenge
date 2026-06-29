"""Stage 0 -- model inspection.

Discovers (does NOT assume) everything the controller depends on:
  - DOF layout (nq, nv, nu) and the floating base
  - actuator TYPE (must be 'motor'/torque, not 'position')
  - joint names + torque limits + actuator->dof mapping
  - foot / pelvis / torso frame (site/body) names + ids
  - foot contact geoms and corner sites
  - available keyframes

Run:  python scripts/00_inspect_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from src.utils.config import load_params, resolve

ACT_TRNTYPE = {
    mujoco.mjtTrn.mjTRN_JOINT: "joint",
}
# Map gaintype to a human label for the 'is it a motor or position?' check.
GAINTYPE = {
    mujoco.mjtGain.mjGAIN_FIXED: "fixed (motor/torque)",
    mujoco.mjtGain.mjGAIN_AFFINE: "affine",
    mujoco.mjtGain.mjGAIN_MUSCLE: "muscle",
}
BIASTYPE = {
    mujoco.mjtBias.mjBIAS_NONE: "none (=> motor/torque)",
    mujoco.mjtBias.mjBIAS_AFFINE: "affine (=> position/velocity servo)",
    mujoco.mjtBias.mjBIAS_MUSCLE: "muscle",
}


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    params = load_params()
    scene = resolve(params["model"]["scene_flat"])
    print(f"Loading scene: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    hr("DIMENSIONS")
    print(f"nq (position coords)     = {model.nq}")
    print(f"nv (DOF / velocity)      = {model.nv}")
    print(f"nu (actuators)           = {model.nu}")
    print(f"nbody                    = {model.nbody}")
    print(f"timestep                 = {model.opt.timestep}")
    print(f"gravity                  = {model.opt.gravity}")

    hr("JOINTS (first joint should be the floating base)")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = mujoco.mjtJoint(model.jnt_type[j]).name
        qadr = model.jnt_qposadr[j]
        vadr = model.jnt_dofadr[j]
        print(f"  [{j:2d}] {name:28s} type={jtype:14s} qpos@{qadr:2d} dof@{vadr:2d}")

    hr("ACTUATORS  (MUST be motor/torque, NOT position)")
    all_motor = True
    print(f"{'idx':>3} {'name':28s} {'trn':6s} {'gain':22s} {'bias':28s} ctrlrange")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        trn = ACT_TRNTYPE.get(model.actuator_trntype[a], str(model.actuator_trntype[a]))
        gain = GAINTYPE.get(model.actuator_gaintype[a], str(model.actuator_gaintype[a]))
        bias = BIASTYPE.get(model.actuator_biastype[a], str(model.actuator_biastype[a]))
        cr = model.actuator_ctrlrange[a]
        is_motor = (model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE)
        all_motor = all_motor and is_motor
        print(f"{a:>3} {name:28s} {trn:6s} {gain:22s} {bias:28s} [{cr[0]:.0f},{cr[1]:.0f}]")
    print(f"\n==> ALL actuators are torque/motor: {all_motor}")

    hr("SELECTION MATRIX S (actuator -> dof)")
    # Build S: maps each actuator to the dof it drives.
    nv, nu = model.nv, model.nu
    S = np.zeros((nu, nv))
    for a in range(nu):
        jnt = model.actuator_trnid[a, 0]
        dof = model.jnt_dofadr[jnt]
        S[a, dof] = 1.0
    underactuated = nv - nu
    print(f"S shape = {S.shape}  (nu x nv)")
    print(f"Underactuated DOFs (floating base) = {underactuated} (expect 6)")
    driven = np.where(S.any(axis=0))[0]
    print(f"Actuated dof indices: {driven.tolist()}")
    print(f"Unactuated dof indices: {sorted(set(range(nv)) - set(driven.tolist()))}")

    hr("KEY FRAMES (sites/bodies used by the controller)")
    for label, key in [("base_body", params["model"]["base_body"]),
                       ("torso_body", params["model"]["torso_body"])]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, key)
        print(f"  body  {label:12s} = '{key}'  id={bid}")
    for label, key in [("imu_site", params["model"]["imu_site"]),
                       ("left_foot", params["model"]["feet"]["left"]["site"]),
                       ("right_foot", params["model"]["feet"]["right"]["site"])]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, key)
        print(f"  site  {label:12s} = '{key}'  id={sid}")
    print("  foot corner sites:")
    for side in ("left", "right"):
        for c in params["model"]["feet"][side]["corners"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, c)
            print(f"    {side:5s} {c:18s} id={sid}")

    hr("KEYFRAMES")
    for k in range(model.nkey):
        kname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        print(f"  [{k}] '{kname}'")

    # Reset to keyframe and report base height + total mass.
    hr("KEYFRAME 'stand' SANITY")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, params["model"]["keyframe"])
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    total_mass = mujoco.mj_getTotalmass(model)
    print(f"total mass               = {total_mass:.3f} kg")
    print(f"weight (m*g)             = {total_mass * abs(model.opt.gravity[2]):.2f} N")
    base_z = data.qpos[2]
    print(f"base height (qpos[2])    = {base_z:.3f} m")
    for side in ("left", "right"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                params["model"]["feet"][side]["site"])
        print(f"{side} foot site pos        = {data.site_xpos[sid]}")
    com = data.subtree_com[0]
    print(f"CoM (world)              = {com}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
