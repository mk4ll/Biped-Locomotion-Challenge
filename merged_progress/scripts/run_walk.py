"""Unified terrain-aware walking (merged): flat / incline / stairs.

Same offline->online->WBC pipeline as the rest of the repo, now terrain-aware:
the offline footstep planner places feet ON the surface (terrain.height /
footstep_x), the online DCM layer is unchanged, and the WBC friction cones are
expressed in the surface frame (terrain.normal). This replaces the bolted-on
incline of Stage 6 with the terrain abstraction (credit: terrain design adapted
from Kanellos' work).

  python scripts/run_walk.py --terrain flat
  python scripts/run_walk.py --terrain incline --angle 10
  python scripts/run_walk.py --terrain stairs
  python scripts/run_walk.py --terrain incline --angle 8 --viewer
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain


def build_on_terrain(params, terrain_name, angle_deg=8.0, stairs_kw=None, robot="g1",
                     decorate=None):
    """Create env+controller on a terrain and put the robot in a valid start pose.
    robot: 'g1' or 'talos' (robot-agnostic stack). decorate: optional mjSpec hook."""
    if terrain_name == "incline":
        terrain = make_terrain("incline", angle=np.deg2rad(angle_deg))
    elif terrain_name == "stairs":
        terrain = make_terrain("stairs", **(stairs_kw or {}))
    else:
        terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain, decorate=decorate)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    m, d = env.model, env.data

    if terrain_name == "incline":
        alpha = terrain.angle
        Rs = terrain.surface_R(0.0, 0.0)
        ctrl.ori_task.R_des = np.eye(3)                 # torso vertical vs gravity
        act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                     for a in range(m.nu)]
        # Split incline angle between ankle and hip so ankles stay within
        # their ±50° limit.  Baseline ankle is ~-30° in the keyframe, so
        # safe additional pre-tilt is at most ~17°.  Anything beyond that
        # is absorbed as forward hip-pitch flexion (leaning into the slope).
        ANKLE_LIMIT = 0.30               # max additional ankle pre-tilt [rad] ≈ 17°
        ankle_delta = min(alpha, ANKLE_LIMIT)
        hip_delta   = alpha - ankle_delta  # remaining lean via hip pitch
        hip_joints  = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
        knee_joints = ["left_knee_joint",      "right_knee_joint"]
        for j in mcfg["ankle_pitch_joints"]:
            adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            d.qpos[adr] -= ankle_delta
            if j in act_names:
                ctrl.pos_task.q_nom[act_names.index(j)] -= ankle_delta
        # Lean hips forward to compensate for slope beyond ankle range,
        # and add a proportional knee-flex so the CoM height stays similar.
        for j in hip_joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                adr = m.jnt_qposadr[jid]
                d.qpos[adr] -= hip_delta
                if j in act_names:
                    ctrl.pos_task.q_nom[act_names.index(j)] -= hip_delta
        for j in knee_joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0:
                adr = m.jnt_qposadr[jid]
                d.qpos[adr] += 0.5 * hip_delta   # partial knee-flex to balance CoM
                if j in act_names:
                    ctrl.pos_task.q_nom[act_names.index(j)] += 0.5 * hip_delta
        mujoco.mj_forward(m, d)
        nrm = Rs @ np.array([0., 0., 1.])
        corners = [c for foot in ctrl.contacts.stance for c in foot["corners"]]
        d.qpos[2] -= (min(d.site_xpos[c] @ nrm for c in corners) - 0.005)
        mujoco.mj_forward(m, d)
    # flat / stairs: crouch keyframe pose is already valid on the flat start ground
    return env, ctrl, terrain


def settle(env, ctrl, terrain, seconds=0.8):
    base = ctrl.base_id
    com_ref = env.data.subtree_com[base].copy()
    # Use the terrain surface orientation so the WBC friction cone is aligned
    # with the actual contact normal (critical on steep inclines).
    if terrain is not None:
        ctrl.R_surface = terrain.surface_R(com_ref[0], com_ref[1])
    for _ in range(int(seconds / env.dt)):
        ctrl.com_task.a_ref = np.zeros(3)
        ctrl.com_task.p_ref = com_ref
        ctrl.com_task.v_ref = np.zeros(3)
        ctrl.contacts.set_support("double")
        tau_fb = ctrl.gc.compute(env.data,
                                 active_site_ids=ctrl.contacts.active_site_ids)[0]
        res = ctrl.wbc.solve(env.data, [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                             ctrl.contacts.stance, R_surface=ctrl.R_surface,
                             fallback_tau=tau_fb)
        env.step(res["tau"])


# Speed bundles (validated on G1 flat ground).
# Each bundle is (gait_params, requires_mpc).
# slow  : baseline gait, ~0.154 m/s
# normal: 2.3× baseline, ~0.357 m/s — stable without MPC
# fast  : 2.6× baseline, ~0.393 m/s — requires DCM preview-MPC for stability
SPEED_BUNDLES = {
    "slow":   (dict(step_length=0.10, t_ss=0.50, t_ds=0.15, swing_apex=0.05), False),
    "normal": (dict(step_length=0.20, t_ss=0.44, t_ds=0.12, swing_apex=0.07), False),
    "fast":   (dict(step_length=0.28, t_ss=0.48, t_ds=0.12, swing_apex=0.09), True),
}


def run(terrain_name="flat", angle_deg=8.0, viewer=False, robot="g1", step_len=None,
        mpc=False, arm_swing=False, step_timing=False, hard_stairs=False, speed=None):
    params = load_params()
    if speed is not None and speed in SPEED_BUNDLES:
        g = params["gait"]
        bundle, needs_mpc = SPEED_BUNDLES[speed]
        for k, v in bundle.items():
            g[k] = v
        if needs_mpc:
            params["dcm_mpc"]["enabled"] = True   # fast mode requires MPC
            params["capture"]["max_shift"] = 0.14  # slightly wider corrective budget
    if step_len is not None:                       # fine-grained override
        params["gait"]["step_length"] = step_len
    if mpc:
        params["dcm_mpc"]["enabled"] = True
    if arm_swing:
        params["arm_swing"]["enabled"] = True
    if step_timing:
        params["step_timing"]["enabled"] = True
    if terrain_name == "stairs":
        if hard_stairs:
            # Hard: 4 cm risers, 20 cm run (standard indoor stairs)
            stairs_kw = dict(rise=0.04, run=0.20, n_steps=6, x0=0.30)
            g = params["gait"]
            g["step_length"] = stairs_kw["run"]
            g["swing_apex"] = 0.10               # extra clearance for 4 cm riser
            g["t_ss"] = 0.55                     # slightly longer SS to clear tall riser
            g["n_steps"] = 18
        else:
            # Easy: 2.5 cm risers, 22 cm run (wider tread — foot lands at center)
            stairs_kw = dict(rise=0.025, run=0.22, n_steps=6, x0=0.30)
            g = params["gait"]
            g["step_length"] = stairs_kw["run"]
            g["swing_apex"] = 0.06
            g["n_steps"] = 18
    else:
        stairs_kw = dict(rise=0.025, run=0.16, n_steps=6, x0=0.30)
    env, ctrl, terrain = build_on_terrain(params, terrain_name, angle_deg, stairs_kw, robot)
    settle(env, ctrl, terrain, 0.8)

    base = ctrl.base_id
    init_left = env.data.site_xpos[ctrl.left_site].copy()
    init_right = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)
    x0 = com0[0]
    n = int(plan.duration / env.dt)
    log = {"x": [], "z": [], "bz": []}

    # On steep inclines the pelvis crouches as it climbs (CoM height trades
    # against slope height); use a lower threshold so a valid climbing gait
    # is not mis-classified as a fall.
    fall_thresh = 0.45

    def loop(i):
        ctrl.step(plan, i * env.dt)
        c = env.data.subtree_com[base]
        log["x"].append(c[0]); log["z"].append(c[2]); log["bz"].append(env.data.qpos[2])
        # "fell" = base dropped well below the local surface height
        return env.data.qpos[2] - terrain.height(c[0], 0.0) < fall_thresh

    fell = False
    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for i in range(n):
                if not v.is_running():
                    break
                if loop(i):
                    fell = True; break
                v.sync()
    else:
        for i in range(n):
            if loop(i):
                fell = True; break

    dist = env.data.subtree_com[base][0] - x0
    rise = env.data.subtree_com[base][2] - com0[2]
    label = terrain_name + (f" {angle_deg:.0f}deg" if terrain_name == "incline" else "")
    if terrain_name == "stairs" and hard_stairs:
        label += " HARD (4cm risers)"
    print(f"\n===== Merged terrain walk: {label} =====")
    print(f"fell             = {fell}")
    print(f"forward distance = {dist:.3f} m")
    print(f"height gain      = {rise*1000:.0f} mm")
    ok = (not fell) and dist > 0.3
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    ap.add_argument("--angle", type=float, default=8.0)
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    ap.add_argument("--step-len", type=float, default=None,
                    help="step length [m] = speed knob (flat safe <=0.20; ~0.23 m/s)")
    ap.add_argument("--mpc", action="store_true",
                    help="use DCM preview-MPC instead of one-step proportional law")
    ap.add_argument("--arm-swing", action="store_true",
                    help="enable contralateral arm swing (natural gait)")
    ap.add_argument("--step-timing", action="store_true",
                    help="use step timing QP (Khadiv et al.): joint footstep+timing optimisation")
    ap.add_argument("--hard-stairs", action="store_true",
                    help="steeper stair configuration: 4 cm risers, 20 cm run (standard indoor)")
    ap.add_argument("--speed", default=None, choices=["slow", "normal", "fast"],
                    help="gait speed preset (slow~0.15, normal~0.27, fast~0.42 m/s)")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(args.terrain, args.angle, args.viewer, args.robot, args.step_len,
        mpc=args.mpc, arm_swing=args.arm_swing,
        step_timing=args.step_timing, hard_stairs=args.hard_stairs, speed=args.speed)
