"""Open the MuJoCo viewer and watch the G1 walk under live dynamic control.

macOS requires the viewer to own the main thread, so launch with mjpython:

    mjpython scripts/view_walk.py
    mjpython scripts/view_walk.py --terrain incline --angle 6 --step-len 0.12
    mjpython scripts/view_walk.py --march          # step in place

The control is fully dynamic: every simulation step the whole-body QP solves
inverse dynamics from the *measured* state (mass matrix, bias forces, contact
Jacobians) and commands joint *torques* -- no scripted joint playback.  Drag the
robot with the mouse (double-click a body, then Ctrl+drag) to apply a push and
watch it balance and keep walking.

(Linux/Windows can use plain `python scripts/view_walk.py`.)
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1
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
    p.add_argument("--march", action="store_true", help="step in place (step-len 0)")
    p.add_argument("--n-steps", type=int, default=12)
    p.add_argument("--t-ss", type=float, default=None)
    p.add_argument("--t-ds", type=float, default=None)
    p.add_argument("--mpc", action="store_true",
                   help="DCM preview-MPC for the CoP (vs one-step feedback)")
    p.add_argument("--realtime", type=float, default=1.0, help="playback speed factor")
    args = p.parse_args()

    if args.terrain == "incline":
        terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle))
    elif args.terrain == "stairs":
        terr = terrain_mod.Stairs(rise=0.03, run=0.15, n_steps=4,
                                  x0=0.525, width=1.6)
    else:
        terr = terrain_mod.make(args.terrain)
    model, data = load_g1(terrain=terr)
    robot = Robot(model, data)
    q_nom = reset_to_crouch(robot, terr)
    # per-terrain finetuned gait; explicit flags override
    gp = terrain_gait(terr, "g1")
    step_len = 0.0 if args.march else (args.step_len if args.step_len is not None
                                       else gp["step_len"])
    ctrl = WalkingController(
        robot, terr, q_nom, n_steps=args.n_steps, step_len=step_len,
        t_ss=args.t_ss if args.t_ss is not None else gp["t_ss"],
        t_ds=args.t_ds if args.t_ds is not None else gp["t_ds"],
        t_ds0=gp["t_ds0"], mpc=args.mpc,
        swing_clearance=gp["swing_clearance"])

    dt = model.opt.timestep
    hz = 1.0 / dt
    print(f"Dynamic whole-body QP control @ {hz:.0f} Hz | torque actuators | "
          f"{'march in place' if step_len == 0 else f'forward {step_len} m/step'} "
          f"on {terr.name}{' | DCM preview-MPC' if args.mpc else ''}")
    print("Ctrl+drag a body in the window to push the robot.  Close window to quit.")

    t0 = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tic = time.time()
            walk_t = data.time - t0
            if walk_t >= ctrl.plan.total_time:        # finished -> keep walking
                ctrl.restart()
                t0 = data.time
                walk_t = 0.0
            tau, _ = ctrl(walk_t)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()
            # pace to wall-clock (real time)
            lag = dt / max(args.realtime, 1e-3) - (time.time() - tic)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
