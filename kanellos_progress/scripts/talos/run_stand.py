"""Talos standing balance (whole-body QP) + optional weight shift / push / viewer.

    python scripts/talos/run_stand.py
    python scripts/talos/run_stand.py --shift --seconds 8
    python scripts/talos/run_stand.py --push 200
    python scripts/talos/run_stand.py --terrain incline --angle 8
    mjpython scripts/talos/run_stand.py --view
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biped.robot import Robot, load_talos, TALOS_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers.standing import StandingController


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=8.0)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--shift", action="store_true", help="lateral CoM weight shift")
    p.add_argument("--push", type=float, default=0.0, help="lateral shove (N) at t=2s")
    p.add_argument("--view", action="store_true", help="interactive viewer (mjpython)")
    args = p.parse_args()

    terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle)) \
        if args.terrain == "incline" else terrain_mod.make(args.terrain)
    model, data = load_talos(terrain=terr)
    robot = Robot(model, data, TALOS_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    ctrl = StandingController(robot, terr, q_nom=q_nom)
    com0 = robot.com()[:2].copy()

    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)

    dt = model.opt.timestep
    n = int(args.seconds / dt)
    push_step, push_dur = int(2.0 / dt), int(0.1 / dt)
    tilt_max, taus = 0.0, []
    for i in range(n):
        t = i * dt
        if args.shift:
            ctrl.set_com_target([com0[0], com0[1] + 0.05 * np.sin(2 * np.pi * t / 4)])
        tau, _ = ctrl()
        data.ctrl[:] = tau
        data.xfrc_applied[:] = 0.0
        if args.push and push_step <= i < push_step + push_dur:
            data.xfrc_applied[robot.pelvis_id, 1] = args.push
        mujoco.mj_step(model, data)
        z = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        taus.append(float(np.abs(tau).max()))
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, dt - 1e-4))
        if data.qpos[2] < 0.6 or tilt > 40:
            print(f"[t={t:.2f}] FELL"); break
    if viewer is not None:
        viewer.close()

    print(f"\n=== Talos standing ({terr.name}) ===")
    print(f"max tilt={tilt_max:.2f} deg  peak |tau|={max(taus):.1f} Nm")
    print(f"foot force L/R = {robot.foot_contact_force('left'):.0f}/"
          f"{robot.foot_contact_force('right'):.0f} N  (weight {robot.total_mass*9.81:.0f} N)")


if __name__ == "__main__":
    main()
