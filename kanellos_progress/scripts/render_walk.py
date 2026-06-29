"""Record a walking (or marching) run to a GIF -- works for G1 and Talos.

Examples
--------
  python scripts/render_walk.py                              # G1 forward walk
  python scripts/render_walk.py --robot talos                # Talos forward walk
  python scripts/render_walk.py --mpc                        # DCM preview-MPC
  python scripts/render_walk.py --terrain incline --angle 8
  python scripts/render_walk.py --march                      # step in place
  python scripts/render_walk.py --robot talos --out logs/talos_walk.gif

A tracking camera follows the base so the robot stays centred as it advances.
Camera framing auto-adjusts to the robot (Talos is ~30% taller than the G1).
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1, load_talos, TALOS_CONFIG, G1_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import WalkingController
from biped.gait import terrain_gait


def make_terrain(args):
    """Build the terrain.  Stairs use a fixed grid run (= nominal stride) so each
    step lands on a tread centre; rise/count scale with the robot."""
    if args.terrain == "incline":
        return terrain_mod.make("incline", angle=np.deg2rad(args.angle))
    if args.terrain == "stairs":
        rise = 0.05 if args.robot == "talos" else 0.03
        n = 5 if args.robot == "talos" else 4
        return terrain_mod.Stairs(rise=rise, run=0.15, n_steps=n,
                                  x0=0.525, width=1.8)
    return terrain_mod.make(args.terrain)


def build(args):
    terr = make_terrain(args)
    if args.robot == "talos":
        model, data = load_talos(terrain=terr)
        robot = Robot(model, data, TALOS_CONFIG)
    else:
        model, data = load_g1(terrain=terr)
        robot = Robot(model, data, G1_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    # omnidirectional velocity command (forward/back/strafe/turn) overrides the
    # straight per-terrain gait
    if any(v is not None for v in (args.vx, args.vy, args.vyaw)):
        from biped.controllers import WalkGains
        vel = (args.vx or 0.0, args.vy or 0.0, args.vyaw or 0.0)
        ctrl = WalkingController(robot, terr, q_nom, gains=WalkGains(w_ori=10.0),
                                velocity=vel, n_steps=args.n_steps)
        return model, data, robot, ctrl, terr
    gp = terrain_gait(terr, args.robot)
    step_len = 0.0 if args.march else (args.step_len if args.step_len is not None
                                       else gp["step_len"])
    ctrl = WalkingController(
        robot, terr, q_nom, mpc=args.mpc, n_steps=args.n_steps,
        step_len=step_len,
        t_ss=args.t_ss if args.t_ss is not None else gp["t_ss"],
        t_ds=args.t_ds if args.t_ds is not None else gp["t_ds"],
        t_ds0=gp["t_ds0"],
        swing_clearance=(args.clearance if args.clearance is not None
                         else gp["swing_clearance"]))
    return model, data, robot, ctrl, terr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="g1", choices=["g1", "talos"])
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=8.0)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--step-len", type=float, default=None,
                   help="override; default is the per-terrain gait profile")
    p.add_argument("--t-ss", type=float, default=None)
    p.add_argument("--t-ds", type=float, default=None)
    p.add_argument("--march", action="store_true")
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--vx", type=float, default=None, help="forward velocity (m/s)")
    p.add_argument("--vy", type=float, default=None, help="left velocity (m/s)")
    p.add_argument("--vyaw", type=float, default=None, help="turn rate (rad/s)")
    p.add_argument("--clearance", type=float, default=None,
                   help="swing clearance (m); auto for stairs")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, data, robot, ctrl, terr = build(args)
    dt = model.opt.timestep
    n = int((ctrl.plan.total_time + 1.0) / dt)
    every = max(1, int(round(1.0 / (args.fps * dt))))

    # camera framing scales with the robot height
    tall = args.robot == "talos"
    look_z = 0.9 if tall else 0.55
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 130, -12
    cam.distance = 3.6 if tall else 2.8

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    fell = False
    z0_base = data.qpos[2]
    for i in range(n):
        t = i * dt
        tau, _ = ctrl(t)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if i % every == 0:
            # follow the climb on stairs so the robot stays framed
            climbed = max(0.0, data.qpos[2] - z0_base)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], look_z + climbed]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        z = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        if data.qpos[2] < (0.6 if tall else 0.4) or tilt > 45:
            print(f"fell at t={t:.2f}s")
            fell = True
            break

    mode = "march" if args.march else "walk"
    tag = f"{args.robot}_{mode}_{terr.name}{'_mpc' if args.mpc else ''}"
    out = args.out or os.path.join("logs", f"{tag}.gif")
    os.makedirs("logs", exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {out}  ({len(frames)} frames, dist={data.qpos[0]:.2f}m, fell={fell})")


if __name__ == "__main__":
    main()
