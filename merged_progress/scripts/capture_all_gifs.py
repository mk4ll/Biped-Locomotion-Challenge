"""Headless GIF capture for all 19 main.py modes.

Renders each scenario offscreen using MuJoCo's renderer, saves full-res GIFs to
logs/gifs/raw/ and a manifest to logs/gifs/manifest_all.txt. Run compress_all_gifs.py
afterwards to produce small preview GIFs.

  python scripts/capture_all_gifs.py
  python scripts/capture_all_gifs.py --mode 5 d h   # only selected modes
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco
import imageio.v3 as iio

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain
from run_walk import build_on_terrain, settle

W, H = 480, 360
GIF_FPS = 20
FRAME_EVERY = 3         # ~20 fps from ~60 fps sim (dt=0.002 → every 3rd frame = 167 sim steps/s captured)
RAW = Path(__file__).resolve().parents[1] / "logs" / "gifs" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def cam(base_id, dist=2.8, az=140.0, el=-18.0):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    c.trackbodyid = base_id
    c.distance = dist
    c.azimuth = az
    c.elevation = el
    return c


def frame(renderer, env, camera):
    renderer.update_scene(env.data, camera=camera)
    return renderer.render().copy()


def save(frames, name, fps=GIF_FPS):
    p = RAW / name
    iio.imwrite(str(p), frames, extension=".gif", fps=fps, loop=0)
    print(f"  → {name} ({len(frames)} frames, {p.stat().st_size//1024} KB)")
    return name


def run_sim(env, ctrl, plan, cam_obj, renderer, max_steps=None, fall_thresh=0.45):
    """Step simulation, collect frames. Returns (frames, fell)."""
    base = ctrl.base_id
    terrain = ctrl.terrain if hasattr(ctrl, 'terrain') else None
    n = max_steps or int(plan.duration / env.dt)
    frames, fell = [], False
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, cam_obj))
        h = terrain.height(env.data.subtree_com[base][0], 0.0) if terrain is not None else 0.0
        if env.data.qpos[2] - h < fall_thresh:
            fell = True
            break
    return frames, fell


def make_plan(params, env, ctrl, terrain, path=None, velocity=None):
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    return WalkPlan(params, il, ir, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"],
                    terrain=terrain, path=path, velocity=velocity), com0


# ─── mode 1: inspect model ─────────────────────────────────────────────────────

def capture_1_inspect():
    print("[1] inspect model — static robot pose...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.5)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    # slow 360° pan around the standing robot
    frames = []
    for az in np.linspace(0, 360, 120, endpoint=False):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        c.trackbodyid = base
        c.distance = 2.2
        c.azimuth = float(az)
        c.elevation = -12.0
        frames.append(frame(renderer, env, c))
    return save(frames, "01_inspect.gif", fps=24)


# ─── mode 2: gravity compensation ──────────────────────────────────────────────

def capture_2_grav_comp():
    print("[2] gravity compensation...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.0, az=110, el=-14)
    frames = []
    gc = ctrl.gc
    d = env.data
    n = int(3.0 / env.dt)
    for i in range(n):
        tau, _ = gc.compute(d, active_site_ids=ctrl.contacts.active_site_ids)
        env.step(tau)
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, camera))
    return save(frames, "02_grav_comp.gif")


# ─── mode 3: standing balance ───────────────────────────────────────────────────

def capture_3_stand_balance():
    print("[3] standing balance / weight shift...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.5)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=1.8, az=100, el=-14)
    # recreate the sway sequence from 02_stand_balance.py
    from src.control.walking_controller import WalkingController
    m, d = env.model, env.data
    frames = []
    com0 = d.subtree_com[base].copy()
    # sway cycle: left → centre → right → centre → single support
    for target_y, dur in [(0.07, 1.2), (0.0, 0.6), (-0.07, 1.2),
                          (0.0, 0.6), (0.0, 1.5)]:
        n = int(dur / env.dt)
        for i in range(n):
            ctrl.com_task.p_ref = np.array([com0[0], target_y, com0[2]])
            ctrl.com_task.a_ref = np.zeros(3)
            ctrl.com_task.v_ref = np.zeros(3)
            ctrl.contacts.set_support("double")
            tau_fb = ctrl.gc.compute(d, active_site_ids=ctrl.contacts.active_site_ids)[0]
            res = ctrl.wbc.solve(d, [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                                 ctrl.contacts.stance, R_surface=ctrl.R_surface,
                                 fallback_tau=tau_fb)
            env.step(res["tau"])
            if i % FRAME_EVERY == 0:
                frames.append(frame(renderer, env, camera))
    return save(frames, "03_stand_balance.gif")


# ─── mode 4: offline planner plots ──────────────────────────────────────────────

def capture_4_plan_walk():
    print("[4] offline planner — running 03_plan_walk.py then converting PNG...")
    import subprocess, shutil
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "03_plan_walk.py")],
                   cwd=str(root), capture_output=True)
    png = root / "logs" / "stage3_plan.png"
    if not png.exists():
        print("  WARNING: stage3_plan.png not found, using blank frame")
        img = Image.new("RGB", (W, H), (30, 32, 36))
        frames = [np.array(img)] * 40
    else:
        img = Image.open(png).convert("RGB").resize((W, H))
        arr = np.array(img)
        # hold the plot for 3s then fade out
        frames = [arr] * 60 + [np.clip(arr * max(0, 1 - k/20), 0, 255).astype(np.uint8)
                                for k in range(20)]
    return save(frames, "04_plan_walk.gif", fps=20)


# ─── mode 5: flat walk ──────────────────────────────────────────────────────────

def capture_5_flat():
    print("[5] flat walk...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.5, az=145)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "05_flat_walk.gif")


# ─── mode 6: incline 16 deg ─────────────────────────────────────────────────────

def capture_6_incline():
    print("[6] incline 16°...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "incline", angle_deg=16.0, robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.8, az=150, el=-14)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "06_incline_16deg.gif")


# ─── mode 7: stairs 2.5 cm ──────────────────────────────────────────────────────

def capture_7_stairs():
    print("[7] stairs 2.5 cm risers, 22 cm run...")
    params = load_params()
    stairs_kw = dict(rise=0.025, run=0.22, n_steps=6, x0=0.30)
    g = params["gait"]
    g["step_length"] = stairs_kw["run"]
    g["swing_apex"] = 0.06
    g["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, "stairs", stairs_kw=stairs_kw, robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.8, az=155, el=-12)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "07_stairs_easy.gif")


# ─── mode 8: omni curve ─────────────────────────────────────────────────────────

def capture_8_omni():
    print("[8] omni curve...")
    params = load_params()
    env, mcfg = make_robot_env("g1")   # omni uses flat ground, no terrain
    ctrl = WalkingController(env, params, mcfg=mcfg)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=3.5, az=100, el=-22)
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], velocity=(0.10, 0.0, 0.12))
    n = int(plan.duration / env.dt)
    frames, fell = [], False
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, camera))
        if env.data.qpos[2] < 0.45:
            fell = True; break
    return save(frames, "08_omni_curve.gif")


# ─── mode 9: push recovery ──────────────────────────────────────────────────────

def capture_9_push():
    print("[9] push recovery...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.5, az=130, el=-16)
    plan, _ = make_plan(params, env, ctrl, terrain)
    m, d = env.model, env.data
    n = int(plan.duration / env.dt)
    pushes = [(int(n * 0.35), np.array([0.0, 60.0, 0.0]), 80),
              (int(n * 0.65), np.array([0.0, -60.0, 0.0]), 80)]
    push_idx = 0
    frames, fell = [], False
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        # apply impulses
        if push_idx < len(pushes):
            pi, fvec, dur = pushes[push_idx]
            if pi <= i < pi + dur:
                d.xfrc_applied[pelvis_id, :3] = fvec
            else:
                d.xfrc_applied[pelvis_id, :3] = 0.0
                if i >= pi + dur:
                    push_idx += 1
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, camera))
        if d.qpos[2] < 0.45:
            fell = True; break
    d.xfrc_applied[pelvis_id, :3] = 0.0
    return save(frames, "09_push_recovery.gif")


# ─── mode 0: slip limit sweep ───────────────────────────────────────────────────

def capture_0_slip():
    print("[0] slip limit sweep — convert plot PNG...")
    import subprocess
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "06_walk_incline.py"), "--sweep"],
                   cwd=str(root), capture_output=True)
    # look for any generated plot
    candidates = list((root / "logs").glob("stage6_walk_incline*.png")) + \
                 list((root / "logs").glob("slip_limit*.png")) + \
                 list((root / "logs").glob("06_*.png"))
    if not candidates:
        candidates = sorted((root / "logs").glob("*.png"))
    if candidates:
        png = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        img = Image.open(png).convert("RGB").resize((W, H))
        arr = np.array(img)
        frames = [arr] * 60
    else:
        frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 40
    return save(frames, "0_slip_sweep.gif", fps=20)


# ─── mode a: lecture plots ───────────────────────────────────────────────────────

def capture_a_plots():
    print("[a] lecture plots — convert PNG...")
    import subprocess
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "plot_walk.py"),
                    "--terrain", "stairs"], cwd=str(root), capture_output=True)
    candidates = list((root / "logs").glob("plot_walk*stairs*.png")) + \
                 list((root / "logs").glob("plot_walk_g1_stairs*.png"))
    if candidates:
        png = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        img = Image.open(png).convert("RGB").resize((W, H))
        arr = np.array(img)
        frames = [arr] * 60
    else:
        frames = [np.zeros((H, W, 3), dtype=np.uint8)] * 40
    return save(frames, "a_lecture_plots.gif", fps=20)


# ─── mode b: full eval battery ───────────────────────────────────────────────────

def capture_b_eval():
    print("[b] eval battery — static robot + scrolling results...")
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / "evaluate.py")],
                            cwd=str(root), capture_output=True, text=True)
    lines = [l for l in (result.stdout + result.stderr).splitlines()
             if l.strip() and not l.startswith("Warning")][:24]
    # render as text on dark background
    bg = (24, 27, 30)
    fg = (200, 200, 180)
    frames = []
    for start in list(range(0, max(1, len(lines) - 16))) + [max(0, len(lines) - 16)] * 30:
        img = Image.new("RGB", (W, H), bg)
        draw = ImageDraw.Draw(img)
        draw.text((12, 10), "EVALUATION RESULTS", fill=(212, 146, 46))
        for j, line in enumerate(lines[start:start + 16]):
            color = (100, 220, 120) if "PASS" in line or "False" in line else \
                    (220, 80, 80) if "FAIL" in line or "True" in line else fg
            draw.text((12, 36 + j * 18), line[:68], fill=color)
        frames.append(np.array(img))
    return save(frames, "b_evaluate.gif", fps=8)


# ─── mode c: waiter navigate ────────────────────────────────────────────────────

def capture_c_navigate():
    print("[c] waiter navigate...")
    from run_navigate import tray_decorator, _make_walkable_layout
    from src.planning import navigation
    params = load_params()
    params["gait"]["step_length"] = 0.10
    tables, goal, path, seed = _make_walkable_layout(4, (0.0, 0.0), 1)
    terrain = make_terrain("flat", obstacles=tuple(tables), markers=(goal,))
    torso = "torso_link"
    env, mcfg = make_robot_env("g1", terrain=terrain, decorate=tray_decorator(torso))
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    settle(env, ctrl, terrain, 1.0)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=4.0, az=120, el=-25)
    plan, _ = make_plan(params, env, ctrl, terrain, path=path)
    n = int(plan.duration / env.dt)
    frames, fell = [], False
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, camera))
        if env.data.qpos[2] < 0.45:
            fell = True; break
    return save(frames, "c_navigate.gif")


# ─── mode d: sisyphus ───────────────────────────────────────────────────────────

def capture_d_sisyphus():
    print("[d] sisyphus...")
    from run_sisyphus import boulder_decorator, extend_arms, PHYSICS_RADIUS, VISUAL_RADIUS
    params = load_params()
    params["gait"]["step_length"] = 0.14
    params["gait"]["n_steps"] = 8
    alpha = np.deg2rad(5.0)
    x0 = PHYSICS_RADIUS + 0.42
    deco = boulder_decorator(alpha, x0, PHYSICS_RADIUS, 1.3, visual_radius=VISUAL_RADIUS)
    env, ctrl, terrain = build_on_terrain(params, "incline", 5.0, None, "g1", deco)
    settle(env, ctrl, terrain, 0.6)
    extend_arms(env, ctrl, "g1")
    ctrl.slope_accel_ff = np.array([1.2, 0.0])
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=3.0, az=160, el=-14)
    plan, com0 = make_plan(params, env, ctrl, terrain)
    n = int(plan.duration / env.dt)
    frames, fell = [], False
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(frame(renderer, env, camera))
        c = env.data.subtree_com[base]
        if env.data.qpos[2] - terrain.height(c[0], 0.0) < 0.45:
            fell = True; break
    return save(frames, "d_sisyphus.gif")


# ─── mode e: DCM MPC flat walk ──────────────────────────────────────────────────

def capture_e_mpc():
    print("[e] DCM MPC flat walk (fast speed)...")
    params = load_params()
    params["dcm_mpc"]["enabled"] = True
    params["capture"]["max_shift"] = 0.14
    # Fast speed bundle: 0.28 m step → 0.467 m/s
    g = params["gait"]
    g["step_length"] = 0.28
    g["t_ss"] = 0.48
    g["t_ds"] = 0.12
    g["swing_apex"] = 0.09
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.5, az=145)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "e_mpc_walk.gif")


# ─── mode f: arm swing ──────────────────────────────────────────────────────────

def capture_f_arm_swing():
    print("[f] arm swing flat walk...")
    params = load_params()
    params["arm_swing"]["enabled"] = True
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.2, az=90, el=-14)  # side view to show arm swing
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "f_arm_swing.gif")


# ─── mode g: step timing QP ─────────────────────────────────────────────────────

def capture_g_step_timing():
    print("[g] step timing QP...")
    params = load_params()
    params["step_timing"]["enabled"] = True
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.5, az=145)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "g_step_timing.gif")


# ─── mode h: online velocity following ──────────────────────────────────────────

def capture_h_vel_change():
    print("[h] online velocity following...")
    from run_velocity_change import run as run_vel
    # We run the script via subprocess and capture output, but for GIF we
    # recreate the multi-segment plan inline.
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot="g1")
    settle(env, ctrl, terrain, 0.5)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=3.5, az=100, el=-22)
    frames = []

    SEGMENTS = [
        (0.15, 0.0,  0.0),
        (0.10, 0.0,  0.18),
        (0.15, 0.0,  0.0),
        (0.08, 0.06, 0.10),
    ]
    for seg_i, (vx, vy, vyaw) in enumerate(SEGMENTS):
        il = env.data.site_xpos[ctrl.left_site].copy()
        ir = env.data.site_xpos[ctrl.right_site].copy()
        com0 = env.data.subtree_com[base].copy()
        plan = WalkPlan(params, il, ir, com0,
                        com_height=params["gait"]["com_height"],
                        gravity=params["env"]["gravity"],
                        terrain=terrain, velocity=(vx, vy, vyaw))
        n = int(plan.duration / env.dt)
        for i in range(n):
            ctrl.step(plan, i * env.dt)
            if i % FRAME_EVERY == 0:
                frames.append(frame(renderer, env, camera))
            if env.data.qpos[2] < 0.45:
                break
    return save(frames, "h_vel_change.gif")


# ─── mode i: hard stairs ────────────────────────────────────────────────────────

def capture_i_hard_stairs():
    print("[i] hard stairs 4 cm risers...")
    params = load_params()
    stairs_kw = dict(rise=0.04, run=0.20, n_steps=6, x0=0.30)
    g = params["gait"]
    g["step_length"] = stairs_kw["run"]
    g["swing_apex"] = 0.10
    g["t_ss"] = 0.55
    g["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, "stairs", stairs_kw=stairs_kw, robot="g1")
    settle(env, ctrl, terrain, 0.8)
    renderer = mujoco.Renderer(env.model, H, W)
    base = ctrl.base_id
    camera = cam(base, dist=2.8, az=155, el=-12)
    plan, _ = make_plan(params, env, ctrl, terrain)
    frames, _ = run_sim(env, ctrl, plan, camera, renderer)
    return save(frames, "i_hard_stairs.gif")


# ─── dispatcher ──────────────────────────────────────────────────────────────────

CAPTURES = {
    "1": capture_1_inspect,
    "2": capture_2_grav_comp,
    "3": capture_3_stand_balance,
    "4": capture_4_plan_walk,
    "5": capture_5_flat,
    "6": capture_6_incline,
    "7": capture_7_stairs,
    "8": capture_8_omni,
    "9": capture_9_push,
    "0": capture_0_slip,
    "a": capture_a_plots,
    "b": capture_b_eval,
    "c": capture_c_navigate,
    "d": capture_d_sisyphus,
    "e": capture_e_mpc,
    "f": capture_f_arm_swing,
    "g": capture_g_step_timing,
    "h": capture_h_vel_change,
    "i": capture_i_hard_stairs,
}

ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
         "a", "b", "c", "d", "e", "f", "g", "h", "i"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="*", default=None,
                    help="Which modes to capture (default: all). E.g. --mode 5 d h")
    args = ap.parse_args()
    keys = args.mode if args.mode else ORDER
    # validate
    bad = [k for k in keys if k not in CAPTURES]
    if bad:
        print(f"Unknown mode(s): {bad}. Valid: {ORDER}")
        sys.exit(1)

    manifest = []
    for k in keys:
        try:
            name = CAPTURES[k]()
            manifest.append(f"{k}\t{name}\tOK")
        except Exception as exc:
            print(f"  ERROR in mode {k}: {exc}")
            manifest.append(f"{k}\t(none)\tERROR: {exc}")

    mpath = Path(__file__).resolve().parents[1] / "logs" / "gifs" / "manifest_all.txt"
    mpath.write_text("\n".join(manifest) + "\n")
    print(f"\nManifest written: {mpath}")
    print(f"Raw GIFs in: {RAW}")
