"""67 Shuffle Dance GIF for the G1 robot.

The '67 shuffle' is a viral dance: alternating arms raise up and down
(left then right) in a rhythmic wave while the body does a small lateral
bounce.

  python scripts/run_dance.py              # headless, saves GIF
  python scripts/run_dance.py --viewer     # live MuJoCo viewer

Output: logs/gifs/raw/dance_67.gif
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
from src.planning.terrain import make_terrain

# ── render settings ───────────────────────────────────────────────────────────
W, H = 480, 360
GIF_FPS = 20
FRAME_EVERY = 25  # sim runs 500 Hz, capture every 25 steps → 20 fps real-time playback

# ── dance timing ──────────────────────────────────────────────────────────────
PHASE1_DUR = 1.0   # both arms rise slowly
PHASE2_DUR = 6.0   # the 67 shuffle
PHASE3_DUR = 1.0   # both arms lower to rest
TOTAL_DUR = PHASE1_DUR + PHASE2_DUR + PHASE3_DUR   # 8 s

CYCLE_DUR = 0.35   # one left+right arm cycle  (~2.9 Hz — snappy rhythm)

# arm angles [rad] — big amplitudes so motion is clearly visible
ARM_REST_PITCH   =  0.0    # neutral shoulder pitch
ARM_HIGH_PITCH   =  1.20   # +69 deg  (dramatic raise)
ARM_RAISE1_PITCH =  0.70   # +40 deg  (phase 1 slow raise)
ARM_HIGH_ROLL    =  0.70   # +40 deg  (wide lateral spread)
ARM_MID_ROLL     =  0.20   # +11 deg  (resting side during other arm's raise)

# lateral CoM sway amplitude [m] and knee flex boost [rad]
COM_SWAY_AMP    = 0.07     # 7 cm — clearly visible hip sway
KNEE_FLEX_BOOST = 0.12     # +7 deg — bouncy knee dip on the support side

# WBC gains boosted for the dance: pos_task weight 1→40 so arms actually track
DANCE_POSTURE_WEIGHT = 40.0
DANCE_POSTURE_KP     = 200.0
DANCE_POSTURE_KD     = 25.0


def _find_act_idx(terms, model, joint_name):
    """Return index of joint_name in pos_task.q_nom (act_dof order), or None."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    dof = model.jnt_dofadr[jid]
    act_dof_list = list(terms.act_dof)
    return act_dof_list.index(dof) if dof in act_dof_list else None


def settle(env, ctrl, seconds=0.8):
    """Hold double-support stance and let the sim settle."""
    base = ctrl.base_id
    com_ref = env.data.subtree_com[base].copy()
    ctrl.contacts.set_support("double")
    for _ in range(int(seconds / env.dt)):
        ctrl.com_task.a_ref = np.zeros(3)
        ctrl.com_task.p_ref = com_ref.copy()
        ctrl.com_task.v_ref = np.zeros(3)
        ctrl.contacts.set_support("double")
        tau_fb = ctrl.gc.compute(env.data,
                                 active_site_ids=ctrl.contacts.active_site_ids)[0]
        res = ctrl.wbc.solve(env.data,
                             [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                             ctrl.contacts.stance,
                             R_surface=ctrl.R_surface,
                             fallback_tau=tau_fb)
        env.step(res["tau"])


def dance_step(env, ctrl, t, com_ref, q_nom_base, joint_idx, knee_idx):
    """Apply one control step with choreography for time t."""
    m = env.model
    base = ctrl.base_id

    # ── choreography ──────────────────────────────────────────────────────────
    sho_L_pitch = joint_idx["left_shoulder_pitch"]
    sho_R_pitch = joint_idx["right_shoulder_pitch"]
    sho_L_roll  = joint_idx["left_shoulder_roll"]
    sho_R_roll  = joint_idx["right_shoulder_roll"]
    knee_L      = knee_idx["left_knee"]
    knee_R      = knee_idx["right_knee"]

    q = ctrl.pos_task.q_nom

    if t < PHASE1_DUR:
        # Phase 1: both arms slowly rise to +30 deg pitch
        alpha = t / PHASE1_DUR
        pitch = alpha * ARM_RAISE1_PITCH
        if sho_L_pitch is not None:
            q[sho_L_pitch] = q_nom_base[sho_L_pitch] + pitch
        if sho_R_pitch is not None:
            q[sho_R_pitch] = q_nom_base[sho_R_pitch] + pitch
        com_sway = 0.0
        knee_delta_L = 0.0
        knee_delta_R = 0.0

    elif t < PHASE1_DUR + PHASE2_DUR:
        # Phase 2: the 67 shuffle
        phase_t = t - PHASE1_DUR
        # Fraction through current cycle [0, 1)
        cycle_frac = (phase_t % CYCLE_DUR) / CYCLE_DUR
        # Left arm up = first half of cycle; right arm up = second half
        if cycle_frac < 0.5:
            # Left arm HIGH, right arm low — sway left
            blend = np.sin(np.pi * cycle_frac / 0.5)   # 0→1→0 smooth peak
            lp = ARM_HIGH_PITCH                         # left: full raise
            rp = ARM_RAISE1_PITCH * (1.0 - blend * 0.6)  # right: drops down
            lr = ARM_HIGH_ROLL                          # left: wide spread out
            rr = -ARM_MID_ROLL * blend                  # right: slight inward
            com_sway = -COM_SWAY_AMP * blend            # hips sway left (−y)
            knee_delta_L = 0.0
            knee_delta_R = KNEE_FLEX_BOOST * blend      # right knee dips (support)
        else:
            # Right arm HIGH, left arm low — sway right
            blend = np.sin(np.pi * (cycle_frac - 0.5) / 0.5)
            rp = ARM_HIGH_PITCH                         # right: full raise
            lp = ARM_RAISE1_PITCH * (1.0 - blend * 0.6)  # left: drops down
            rr = -ARM_HIGH_ROLL                         # right: wide spread out
            lr = ARM_MID_ROLL * blend                   # left: slight inward
            com_sway = COM_SWAY_AMP * blend             # hips sway right (+y)
            knee_delta_R = 0.0
            knee_delta_L = KNEE_FLEX_BOOST * blend      # left knee dips (support)

        if sho_L_pitch is not None:
            q[sho_L_pitch] = q_nom_base[sho_L_pitch] + lp
        if sho_R_pitch is not None:
            q[sho_R_pitch] = q_nom_base[sho_R_pitch] + rp
        if sho_L_roll is not None:
            q[sho_L_roll] = q_nom_base[sho_L_roll] + lr
        if sho_R_roll is not None:
            q[sho_R_roll] = q_nom_base[sho_R_roll] + rr
        if knee_L is not None:
            q[knee_L] = q_nom_base[knee_L] + knee_delta_L
        if knee_R is not None:
            q[knee_R] = q_nom_base[knee_R] + knee_delta_R

    else:
        # Phase 3: both arms lower back to rest
        alpha = (t - PHASE1_DUR - PHASE2_DUR) / PHASE3_DUR
        pitch = (1.0 - alpha) * ARM_RAISE1_PITCH
        if sho_L_pitch is not None:
            q[sho_L_pitch] = q_nom_base[sho_L_pitch] + pitch
        if sho_R_pitch is not None:
            q[sho_R_pitch] = q_nom_base[sho_R_pitch] + pitch
        if sho_L_roll is not None:
            q[sho_L_roll] = q_nom_base[sho_L_roll]
        if sho_R_roll is not None:
            q[sho_R_roll] = q_nom_base[sho_R_roll]
        com_sway = 0.0
        knee_delta_L = 0.0
        knee_delta_R = 0.0

    # ── WBC ───────────────────────────────────────────────────────────────────
    com_now = env.data.subtree_com[base].copy()
    p_ref = com_ref.copy()
    p_ref[1] += com_sway   # lateral sway

    ctrl.com_task.a_ref = np.zeros(3)
    ctrl.com_task.p_ref = p_ref
    ctrl.com_task.v_ref = np.zeros(3)
    ctrl.contacts.set_support("double")

    tau_fb = ctrl.gc.compute(env.data,
                             active_site_ids=ctrl.contacts.active_site_ids)[0]
    res = ctrl.wbc.solve(env.data,
                         [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                         ctrl.contacts.stance,
                         R_surface=ctrl.R_surface,
                         fallback_tau=tau_fb)
    env.step(res["tau"])


def run(viewer=False, robot="g1"):
    params = load_params()
    terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)

    m = env.model
    terms = ctrl.terms

    # ── joint indices in pos_task.q_nom ───────────────────────────────────────
    joint_names = {
        "left_shoulder_pitch":  "left_shoulder_pitch_joint",
        "right_shoulder_pitch": "right_shoulder_pitch_joint",
        "left_shoulder_roll":   "left_shoulder_roll_joint",
        "right_shoulder_roll":  "right_shoulder_roll_joint",
    }
    knee_names = {
        "left_knee":  "left_knee_joint",
        "right_knee": "right_knee_joint",
    }
    joint_idx = {k: _find_act_idx(terms, m, v) for k, v in joint_names.items()}
    knee_idx  = {k: _find_act_idx(terms, m, v) for k, v in knee_names.items()}

    found = {k: v for k, v in {**joint_idx, **knee_idx}.items() if v is not None}
    print(f"Joint indices found: {found}")
    missing = [k for k, v in {**joint_idx, **knee_idx}.items() if v is None]
    if missing:
        print(f"  (not found in act_dof, will skip): {missing}")

    # ── settle ────────────────────────────────────────────────────────────────
    settle(env, ctrl, seconds=0.8)

    base = ctrl.base_id
    com_ref = env.data.subtree_com[base].copy()
    q_nom_base = ctrl.pos_task.q_nom.copy()

    # Boost posture-task gains so arms actually track targets.
    # Default weight=1 vs com weight=100 means arms are nearly ignored.
    # In double-support dance, COM is stable so we can afford high posture weight.
    _saved_w  = ctrl.pos_task.weight
    _saved_kp = ctrl.pos_task.kp
    _saved_kd = ctrl.pos_task.kd
    ctrl.pos_task.weight = DANCE_POSTURE_WEIGHT
    ctrl.pos_task.kp     = DANCE_POSTURE_KP
    ctrl.pos_task.kd     = DANCE_POSTURE_KD

    # ── render setup ──────────────────────────────────────────────────────────
    n_steps = int(TOTAL_DUR / env.dt)

    out_dir = Path(__file__).resolve().parents[1] / "logs" / "gifs" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dance_67.gif"

    if viewer:
        import mujoco.viewer as _mj_viewer
        with _mj_viewer.launch_passive(env.model, env.data) as v:
            for i in range(n_steps):
                if not v.is_running():
                    break
                t = i * env.dt
                dance_step(env, ctrl, t, com_ref, q_nom_base, joint_idx, knee_idx)
                v.sync()
        ctrl.pos_task.weight = _saved_w
        ctrl.pos_task.kp     = _saved_kp
        ctrl.pos_task.kd     = _saved_kd
        print("Viewer closed.")
        return

    # ── headless capture ──────────────────────────────────────────────────────
    import imageio.v3 as iio

    renderer = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base
    cam.distance = 2.4
    cam.azimuth = 145.0
    cam.elevation = -18.0

    frames = []
    print(f"Rendering {n_steps} steps ({TOTAL_DUR:.1f} s)...")
    for i in range(n_steps):
        t = i * env.dt
        dance_step(env, ctrl, t, com_ref, q_nom_base, joint_idx, knee_idx)
        if i % FRAME_EVERY == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        # bail if robot falls
        if env.data.qpos[2] < 0.40:
            print(f"  Robot fell at t={t:.2f}s, stopping.")
            break

    ctrl.pos_task.weight = _saved_w
    ctrl.pos_task.kp     = _saved_kp
    ctrl.pos_task.kd     = _saved_kd

    iio.imwrite(str(out_path), frames, extension=".gif", fps=GIF_FPS, loop=0)
    kb = out_path.stat().st_size // 1024
    print(f"Saved {out_path} ({len(frames)} frames, {kb} KB)")
    return str(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="67 Shuffle Dance GIF for G1 robot")
    ap.add_argument("--viewer", action="store_true",
                    help="open live MuJoCo viewer instead of saving GIF")
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    args = ap.parse_args()
    run(viewer=args.viewer, robot=args.robot)
