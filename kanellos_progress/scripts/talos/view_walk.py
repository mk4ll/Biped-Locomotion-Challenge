"""Watch the PAL Talos walk live under whole-body QP torque control.

macOS requires the viewer on the main thread, so launch with mjpython:

    mjpython scripts/talos/view_walk.py
    mjpython scripts/talos/view_walk.py --mpc
    mjpython scripts/talos/view_walk.py --terrain incline --angle 6 --step-len 0.12
    mjpython scripts/talos/view_walk.py --march

Ctrl+drag a body in the window to shove the robot and watch it balance.
(Linux/Windows: plain `python scripts/talos/view_walk.py`.)
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biped.robot import Robot, load_talos, TALOS_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import WalkingController
from biped.gait import terrain_gait


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=6.0)
    p.add_argument("--step-len", type=float, default=None,
                   help="override; default is the per-terrain gait profile")
    p.add_argument("--march", action="store_true", help="step in place")
    p.add_argument("--n-steps", type=int, default=12)
    p.add_argument("--t-ss", type=float, default=None)
    p.add_argument("--t-ds", type=float, default=None)
    p.add_argument("--mpc", action="store_true", help="DCM preview-MPC for the CoP")
    p.add_argument("--realtime", type=float, default=1.0, help="playback speed factor")
    args = p.parse_args()

    if args.terrain == "incline":
        terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle))
    elif args.terrain == "stairs":
        terr = terrain_mod.Stairs(rise=0.05, run=0.15, n_steps=5,
                                  x0=0.525, width=1.8)
    else:
        terr = terrain_mod.make(args.terrain)
    model, data = load_talos(terrain=terr)
    robot = Robot(model, data, TALOS_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    gp = terrain_gait(terr, "talos")
    step_len = 0.0 if args.march else (args.step_len if args.step_len is not None
                                       else gp["step_len"])
    ctrl = WalkingController(
        robot, terr, q_nom, n_steps=args.n_steps, step_len=step_len,
        t_ss=args.t_ss if args.t_ss is not None else gp["t_ss"],
        t_ds=args.t_ds if args.t_ds is not None else gp["t_ds"],
        t_ds0=gp["t_ds0"], mpc=args.mpc,
        swing_clearance=gp["swing_clearance"])

    dt = model.opt.timestep
    print(f"Talos whole-body QP @ {1.0/dt:.0f} Hz | 94 kg | torque motors | "
          f"{'march' if step_len == 0 else f'forward {step_len} m/step'} "
          f"on {terr.name}{' | DCM preview-MPC' if args.mpc else ''}")
    print("Ctrl+drag a body to push the robot.  Close the window to quit.")

    t0 = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tic = time.time()
            walk_t = data.time - t0
            if walk_t >= ctrl.plan.total_time:
                ctrl.restart()
                t0 = data.time
                walk_t = 0.0
            tau, _ = ctrl(walk_t)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = dt / max(args.realtime, 1e-3) - (time.time() - tic)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
