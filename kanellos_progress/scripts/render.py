"""Record a controller scenario to a GIF so you can watch the G1 in the scene.

Examples
--------
  python scripts/render.py                       # standing, flat -> logs/stand.gif
  python scripts/render.py --shift --seconds 8   # lateral weight shift
  python scripts/render.py --terrain incline --angle 10
  python scripts/render.py --push 150
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import StandingController


def make_camera():
    cam = mujoco.MjvCamera()
    cam.azimuth = 130
    cam.elevation = -15
    cam.distance = 2.6
    cam.lookat[:] = [0.1, 0.0, 0.55]
    return cam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=8.0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--shift", action="store_true")
    p.add_argument("--shift-amp", type=float, default=0.06)
    p.add_argument("--shift-period", type=float, default=3.0)
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--push-t", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle)) \
        if args.terrain == "incline" else terrain_mod.make(args.terrain)
    model, data = load_g1(terrain=terr)
    robot = Robot(model, data)
    q_nom = reset_to_crouch(robot, terr)
    ctrl = StandingController(robot, terr, q_nom=q_nom)
    com0 = robot.com().copy()

    dt = model.opt.timestep
    n = int(args.seconds / dt)
    every = max(1, int(round(1.0 / (args.fps * dt))))
    push_step = int(args.push_t / dt)
    push_dur = int(0.1 / dt)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = make_camera()
    frames = []

    for i in range(n):
        t = i * dt
        if args.shift:
            y = com0[1] + args.shift_amp * np.sin(2 * np.pi * t / args.shift_period)
            ctrl.set_com_target([com0[0], y])
        tau, sol = ctrl()
        data.ctrl[:] = tau
        data.xfrc_applied[:] = 0.0
        if args.push and push_step <= i < push_step + push_dur:
            data.xfrc_applied[robot.pelvis_id, 1] = args.push
        mujoco.mj_step(model, data)

        if i % every == 0:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.55]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        if data.qpos[2] < 0.4:
            print(f"fell at t={t:.2f}s")
            break

    out = args.out or os.path.join(
        "logs", f"stand_{terr.name}{'_shift' if args.shift else ''}"
        f"{'_push' if args.push else ''}.gif")
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {out}  ({len(frames)} frames, {args.seconds:.1f}s)")


if __name__ == "__main__":
    main()
