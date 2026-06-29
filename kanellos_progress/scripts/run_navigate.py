"""Navigate a humanoid to a goal while avoiding obstacles (omnidirectional gait).

A potential-field navigator (:mod:`biped.navigation`) turns the robot's pose and
the obstacle field into a body-frame velocity command each replanning cycle; the
omnidirectional :class:`WalkingController` walks it (turn-to-face + forward, with
sideways/curving as needed).  Obstacles are orange cylinders; the goal is a green
disk.

    python scripts/run_navigate.py                 # G1, default course
    python scripts/run_navigate.py --robot talos
    python scripts/run_navigate.py --video         # save logs/navigate_<robot>.gif
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1, load_talos, TALOS_CONFIG, G1_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import OnlineWalkingController
from biped.navigation import Navigator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="g1", choices=["g1", "talos"])
    p.add_argument("--goal", type=float, nargs=2, default=[2.6, 0.0])
    p.add_argument("--max-time", type=float, default=80.0)
    p.add_argument("--video", nargs="?", const=True, default=False)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()

    goal = tuple(args.goal)
    # one obstacle on the path: the robot arcs around it (swirl field) and
    # continues to the goal
    obstacles = [(1.4, 0.05, 0.25)]
    terr = terrain_mod.Flat(obstacles=tuple(obstacles), markers=(goal,))

    if args.robot == "talos":
        model, data = load_talos(terrain=terr)
        robot = Robot(model, data, TALOS_CONFIG)
    else:
        model, data = load_g1(terrain=terr)
        robot = Robot(model, data, G1_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    # CONTINUOUS online walker: the navigator updates its velocity command every
    # tick (no replanning / no stop-and-restart), so the robot flows around
    # obstacles.  Translation stays in the stable speed band (>= ~0.2 m/s).
    ctrl = OnlineWalkingController(robot, terr, q_nom, velocity=(0.22, 0.0, 0.0),
                                  max_dy=0.08)
    # avoidance kicks in only near the obstacle; a strong tangential swirl arcs
    # the robot around it (gentle, sustained curve the gait handles well)
    nav = Navigator(goal=goal, obstacles=obstacles, v_max=0.24, vyaw_max=0.0,
                    k_yaw=0.0, goal_tol=0.25, slow_radius=0.4,
                    influence=0.7, k_repel=0.4, k_swirl=1.1)

    dt = model.opt.timestep
    renderer = cam = frames = None
    if args.video:
        renderer = mujoco.Renderer(model, height=420, width=560)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90, -65, 4.2  # near top-down
        frames = []
        every = max(1, int(round(1.0 / (args.fps * dt))))

    fell = done = False
    i = 0
    nmax = int(args.max_time / dt)
    vcmd = np.array([0.2, 0.0, 0.0])     # low-passed command (smooth transitions)
    beta = 1.0 - np.exp(-dt / 0.25)      # ~250 ms smoothing
    while i < nmax:
        pos = np.array([data.qpos[0], data.qpos[1]])
        cmd, done = nav.command(pos, ctrl.theta)
        if done:
            break
        vcmd += beta * (np.array(cmd) - vcmd)
        # keep forward speed in the stable band so it never crawls into a fall
        vx = vcmd[0] if vcmd[0] > 0.16 else max(vcmd[0], 0.0)
        ctrl.set_velocity(vx, vcmd[1], vcmd[2])
        tau, _ = ctrl(data.time)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        z = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        if data.qpos[2] < (0.6 if args.robot == "talos" else 0.4) or tilt > 45:
            fell = True
            break
        if renderer is not None and i % every == 0:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        i += 1

    pos = np.array([data.qpos[0], data.qpos[1]])
    print("\n=== Navigation (%s) ===" % args.robot)
    print("goal=%s  reached=%s  fell=%s" % (goal, done, fell))
    print("final pos=(%.2f, %.2f)  dist to goal=%.2f m  time=%.1fs"
          % (pos[0], pos[1], np.linalg.norm(pos - np.array(goal)), data.time))

    if frames:
        import imageio.v2 as imageio
        out = args.video if isinstance(args.video, str) else \
            os.path.join("logs", f"navigate_{args.robot}.gif")
        imageio.mimsave(out, frames, fps=args.fps, loop=0)
        print("saved %s (%d frames)" % (out, len(frames)))
    return done and not fell


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
