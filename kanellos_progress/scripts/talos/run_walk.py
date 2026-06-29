"""Talos DCM walking demo + test (in-place march or forward).

Mirror of scripts/run_walk.py but for the PAL Talos (94 kg, box feet, 32 torque
motors).  Kept in its own folder so Talos gait params/gains can be tuned without
touching the G1 setup.

Examples
--------
  python scripts/talos/run_walk.py --step-len 0.15 --n-steps 10
  python scripts/talos/run_walk.py --step-len 0.0  --n-steps 8     # march
  python scripts/talos/run_walk.py --step-len 0.15 --mpc           # preview-MPC
  python scripts/talos/run_walk.py --terrain incline --angle 6 --step-len 0.12
  python scripts/talos/run_walk.py --step-len 0.15 --push 80       # mid-walk shove
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biped.robot import Robot, load_talos, TALOS_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import WalkingController


def build(args):
    clr = args.clearance
    if args.terrain == "incline":
        terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle))
    elif args.terrain == "stairs":
        terr = terrain_mod.Stairs(rise=0.05, run=args.step_len, n_steps=5,
                                  x0=0.525, width=1.8)
        clr = max(clr, terr.rise + 0.06)
    else:
        terr = terrain_mod.make(args.terrain)
    model, data = load_talos(terrain=terr)
    robot = Robot(model, data, TALOS_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    ctrl = WalkingController(robot, terr, q_nom,
                            reactive=args.reactive, mpc=args.mpc,
                            n_steps=args.n_steps, step_len=args.step_len,
                            t_ss=args.t_ss, t_ds=args.t_ds,
                            first_swing=args.first_swing,
                            swing_clearance=clr)
    return model, data, robot, ctrl, terr


def run(args):
    model, data, robot, ctrl, terr = build(args)
    dt = model.opt.timestep
    base_x0 = data.qpos[0]
    n = int((ctrl.plan.total_time + 1.0) / dt)
    push_step = int(args.push_t / dt)
    push_dur = int(0.1 / dt)

    tilt_max = 0.0
    taus, comz = [], []
    fell = False
    for i in range(n):
        t = i * dt
        tau, sol = ctrl(t)
        data.ctrl[:] = tau
        data.xfrc_applied[:] = 0.0
        if args.push and push_step <= i < push_step + push_dur:
            data.xfrc_applied[robot.pelvis_id, 1] = args.push  # lateral shove
        mujoco.mj_step(model, data)

        zaxis = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        taus.append(float(np.max(np.abs(tau))))
        comz.append(robot.com()[2])
        if data.qpos[2] < 0.6 or tilt > 40:
            fell = True
            print(f"[t={t:.2f}] FELL (base z={data.qpos[2]:.2f}, tilt={tilt:.1f})")
            break

    dist = data.qpos[0] - base_x0
    print("\n=== Talos walking summary ===")
    print(f"terrain={terr.name}  mode={'march' if args.step_len==0 else 'forward'}  "
          f"steps={args.n_steps}  mpc={args.mpc}  fell={fell}")
    print(f"plan {ctrl.plan.total_time:.2f}s   forward distance: {dist:+.3f} m")
    print(f"CoM height mean={np.mean(comz):.3f} (target {ctrl.z_des:.3f})  "
          f"max tilt={tilt_max:.1f} deg")
    print(f"foot force L/R mean: {robot.foot_contact_force('left'):.0f}/"
          f"{robot.foot_contact_force('right'):.0f} N   peak |tau|: {max(taus):.1f} Nm")
    return not fell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=6.0)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--step-len", type=float, default=0.15)
    p.add_argument("--t-ss", type=float, default=0.7)
    p.add_argument("--t-ds", type=float, default=0.2)
    p.add_argument("--clearance", type=float, default=0.06)
    p.add_argument("--first-swing", default="right", choices=["left", "right"])
    p.add_argument("--push", type=float, default=0.0, help="lateral pelvis push (N)")
    p.add_argument("--push-t", type=float, default=3.0, help="time of push (s)")
    p.add_argument("--reactive", action="store_true")
    p.add_argument("--mpc", action="store_true", help="DCM preview-MPC for the CoP")
    args = p.parse_args()
    sys.exit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
