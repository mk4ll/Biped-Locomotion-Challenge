"""Headless GIF capture for key locomotion scenarios.

Renders each scenario offscreen using MuJoCo's renderer, saves GIFs at 20 fps,
and writes a manifest to logs/gifs/manifest.txt.

  python scripts/capture_gifs.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import imageio.v3 as iio

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain
from scripts.run_walk import build_on_terrain, settle

W, H = 480, 360          # render resolution
GIF_FPS = 20             # output GIF framerate
FRAME_EVERY = 3          # capture 1 frame every N sim frames (sim ~1kHz → ~333 fps raw → /16 ~ 20fps)
LOGS = Path(__file__).resolve().parents[1] / "logs" / "gifs"
LOGS.mkdir(parents=True, exist_ok=True)


def make_cam(base_id, distance=2.8, azimuth=140.0, elevation=-18.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_frame(renderer, env, cam):
    renderer.update_scene(env.data, camera=cam)
    return renderer.render().copy()


def save_gif(frames, path, fps=GIF_FPS):
    iio.imwrite(str(path), frames, extension=".gif", fps=fps, loop=0)
    kb = path.stat().st_size // 1024
    print(f"  saved {path.name} ({len(frames)} frames, {kb} KB)")


# ─── scenarios ───────────────────────────────────────────────────────────────

def capture_flat(robot="g1"):
    print("\n[1/6] flat walk...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "flat", robot=robot)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(base, distance=2.5, azimuth=145)
    frames = []
    n = int(plan.duration / env.dt)
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(render_frame(renderer, env, cam))
        if env.data.qpos[2] < 0.45:
            break
    save_gif(frames, LOGS / "flat_walk.gif")
    return "flat_walk.gif"


def capture_incline(robot="g1"):
    print("\n[2/6] incline 12deg...")
    params = load_params()
    env, ctrl, terrain = build_on_terrain(params, "incline", angle_deg=12.0, robot=robot)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(base, distance=2.8, azimuth=130, elevation=-12)
    frames = []
    n = int(plan.duration / env.dt)
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(render_frame(renderer, env, cam))
        if env.data.qpos[2] - terrain.height(env.data.subtree_com[base][0], 0.0) < 0.45:
            break
    save_gif(frames, LOGS / "incline_12deg.gif")
    return "incline_12deg.gif"


def capture_stairs(robot="g1"):
    print("\n[3/6] stairs 2.5cm risers...")
    params = load_params()
    stairs_kw = dict(rise=0.025, run=0.16, n_steps=6, x0=0.30)
    params["gait"]["step_length"] = stairs_kw["run"]
    params["gait"]["swing_apex"] = 0.06
    params["gait"]["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, "stairs", stairs_kw=stairs_kw, robot=robot)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(base, distance=3.0, azimuth=125, elevation=-14)
    frames = []
    n = int(plan.duration / env.dt)
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(render_frame(renderer, env, cam))
        if env.data.qpos[2] - terrain.height(env.data.subtree_com[base][0], 0.0) < 0.45:
            break
    save_gif(frames, LOGS / "stairs_25mm.gif")
    return "stairs_25mm.gif"


def capture_hard_stairs(robot="g1"):
    print("\n[4/6] hard stairs 4cm risers...")
    params = load_params()
    stairs_kw = dict(rise=0.04, run=0.20, n_steps=6, x0=0.30)
    params["gait"]["step_length"] = stairs_kw["run"]
    params["gait"]["swing_apex"] = 0.10
    params["gait"]["t_ss"] = 0.55
    params["gait"]["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, "stairs", stairs_kw=stairs_kw, robot=robot)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(base, distance=3.2, azimuth=125, elevation=-12)
    frames = []
    n = int(plan.duration / env.dt)
    for i in range(n):
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(render_frame(renderer, env, cam))
        if env.data.qpos[2] - terrain.height(env.data.subtree_com[base][0], 0.0) < 0.45:
            break
    save_gif(frames, LOGS / "stairs_40mm_hard.gif")
    return "stairs_40mm_hard.gif"


def capture_push(robot="g1"):
    print("\n[5/6] push recovery (80N lateral)...")
    params = load_params()
    params["gait"]["step_length"] = 0.10
    params["gait"]["n_steps"] = 14
    params["capture"]["enabled"] = True
    terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(base, distance=2.4, azimuth=150, elevation=-16)
    frames = []
    n = int(plan.duration / env.dt)
    push_start = int(2.0 / env.dt)
    push_end = push_start + int(0.10 / env.dt)
    for i in range(n):
        if push_start <= i < push_end:
            env.data.xfrc_applied[base, :] = [0.0, 80.0, 0.0, 0.0, 0.0, 0.0]
        else:
            env.data.xfrc_applied[base, :] = 0.0
        ctrl.step(plan, i * env.dt)
        if i % FRAME_EVERY == 0:
            frames.append(render_frame(renderer, env, cam))
        if env.data.qpos[2] < 0.45:
            break
    save_gif(frames, LOGS / "push_recovery_80N.gif")
    return "push_recovery_80N.gif"


def capture_velocity_change(robot="g1"):
    print("\n[6/6] online velocity change...")
    params = load_params()

    SEGMENTS = [
        (0.10, 0.0,   0.0,   6),
        (0.09, 0.0,   0.18,  6),
        (0.10, 0.0,   0.0,   6),
        (0.09, 0.0,  -0.15,  4),
        (0.10, 0.0,   0.0,   4),
    ]

    terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    settle(env, ctrl, terrain, 0.8)

    renderer = mujoco.Renderer(env.model, H, W)
    cam = make_cam(ctrl.base_id, distance=3.5, azimuth=0, elevation=-30)
    frames = []
    base = ctrl.base_id
    t_global = 0.0

    for (vx, vy, vyaw, n_steps) in SEGMENTS:
        plocal = params.copy()
        plocal["gait"] = dict(params["gait"])
        plocal["gait"]["step_length"] = vx
        plocal["gait"]["n_steps"] = n_steps
        il = env.data.site_xpos[ctrl.left_site].copy()
        ir = env.data.site_xpos[ctrl.right_site].copy()
        com0 = env.data.subtree_com[base].copy()
        plan = WalkPlan(plocal, il, ir, com0, com_height=params["gait"]["com_height"],
                        gravity=params["env"]["gravity"], terrain=terrain,
                        velocity=(vx, vy, vyaw))
        n = int(plan.duration / env.dt)
        for i in range(n):
            ctrl.step(plan, i * env.dt)
            t_global += env.dt
            if int(t_global / env.dt) % FRAME_EVERY == 0:
                # keep camera overhead, follow robot top-down
                frames.append(render_frame(renderer, env, cam))
            if env.data.qpos[2] < 0.40:
                break

    save_gif(frames, LOGS / "velocity_change.gif")
    return "velocity_change.gif"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    args = ap.parse_args()

    t0 = time.time()
    gifs = []
    gifs.append(capture_flat(args.robot))
    gifs.append(capture_incline(args.robot))
    gifs.append(capture_stairs(args.robot))
    gifs.append(capture_hard_stairs(args.robot))
    gifs.append(capture_push(args.robot))
    gifs.append(capture_velocity_change(args.robot))

    manifest = LOGS / "manifest.txt"
    manifest.write_text("\n".join(gifs))
    total = time.time() - t0
    print(f"\nAll GIFs saved to {LOGS}/")
    print(f"Total time: {total:.1f} s")
