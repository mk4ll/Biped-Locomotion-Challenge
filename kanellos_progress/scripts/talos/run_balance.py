"""Talos one-leg balance: shift the CoM over one foot, then lift the other.

    python scripts/talos/run_balance.py               # stand on right foot
    python scripts/talos/run_balance.py --foot left
    python scripts/talos/run_balance.py --seconds 8
    mjpython scripts/talos/run_balance.py --view

Same whole-body inverse-dynamics QP as walking; only the contact set (one foot)
and a raised-foot tracking task differ.
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
from biped.wbc import Task, Contact, WholeBodyQP

COM_KP, COM_KD = 150.0, 26.0
ORI_KP, ORI_KD = 220.0, 30.0
POST_KP, POST_KD = 60.0, 12.0
FOOT_KP, FOOT_KD = 350.0, 40.0
FOOT_ORI_KP, FOOT_ORI_KD = 200.0, 28.0
W_COM, W_ORI, W_POST, W_FOOT, W_FOOT_ORI = 250.0, 5.0, 0.2, 50.0, 6.0
MU = 0.5
T_SHIFT = 2.0           # s to shift CoM over the stance foot
LIFT = 0.15             # raised-foot clearance (m)


def run(args):
    terr = terrain_mod.Flat()
    model, data = load_talos(terrain=terr)
    robot = Robot(model, data, TALOS_CONFIG)
    q_nom = reset_to_crouch(robot, terr)
    stance = args.foot
    swing = "left" if stance == "right" else "right"

    qp = WholeBodyQP(robot)
    S = np.zeros((robot.nu, robot.nv))
    for k, va in enumerate(robot.act_vadr):
        S[k, va] = 1.0
    quat_des = q_nom[3:7].copy()

    stance_xy = robot.foot_pos(stance)[:2].copy()
    com_init = robot.com().copy()
    z_des = float(com_init[2])
    nudge = -np.sign(stance_xy[1]) * 0.02     # small inward margin toward centre
    com_target = np.array([stance_xy[0], stance_xy[1] + nudge, z_des])

    swing_des = robot.foot_pos(swing).copy()
    swing_des[2] = terr.height(swing_des[0], swing_des[1]) + LIFT

    dt = model.opt.timestep
    n = int(args.seconds / dt)

    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)

    tilt_max, taus, fell = 0.0, [], False
    for i in range(n):
        t = i * dt
        alpha = min(t / T_SHIFT, 1.0)
        com_des = (1 - alpha) * com_init + alpha * com_target

        Jcom = robot.com_jacobian()
        com = robot.com()
        vcom = robot.com_vel(Jcom)
        tasks = [Task(Jcom, COM_KP * (com_des - com) - COM_KD * vcom, W_COM)]

        Jr = robot.body_angular_jacobian(robot.pelvis_id)
        err = np.zeros(3)
        mujoco.mju_subQuat(err, quat_des, robot.pelvis_quat())
        tasks.append(Task(Jr, ORI_KP * err - ORI_KD * (Jr @ data.qvel), W_ORI))

        acc = np.zeros(robot.nv)
        for grp in robot.groups.values():
            for qa, va in zip(grp.qadr, grp.vadr):
                acc[va] = POST_KP * (q_nom[qa] - data.qpos[qa]) - POST_KD * data.qvel[va]
        tasks.append(Task(S, acc[robot.act_vadr], W_POST))

        if alpha >= 1.0:
            Jp, Jrf = robot.site_jacobian(swing)
            fp = robot.foot_pos(swing)
            Jdv = robot.point_jacobian_dot(fp, robot.foot_body[swing]) @ data.qvel
            tasks.append(Task(Jp, FOOT_KP * (swing_des - fp)
                              - FOOT_KD * (Jp @ data.qvel) - Jdv, W_FOOT))
            qf = np.zeros(4)
            mujoco.mju_mat2Quat(qf, robot.foot_rot(swing).flatten())
            ef = np.zeros(3)
            mujoco.mju_subQuat(ef, quat_des, qf)
            tasks.append(Task(Jrf, FOOT_ORI_KP * ef - FOOT_ORI_KD * (Jrf @ data.qvel),
                              W_FOOT_ORI))
            sides = (stance,)
        else:
            sides = ("left", "right")

        contacts = []
        for side in sides:
            for pt in robot.foot_corner_points(side):
                contacts.append(Contact(pt, robot.foot_body[side],
                                        terr.normal(pt[0], pt[1]), MU))

        sol = qp.solve(tasks, contacts)
        data.ctrl[:] = sol["tau"] if sol else 0.0
        mujoco.mj_step(model, data)

        z = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        taus.append(float(np.abs(data.ctrl).max()))
        if data.qpos[2] < 0.6 or tilt > 40:
            fell = True
            print(f"[t={t:.2f}] FELL")
            break
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
    if viewer is not None:
        viewer.close()

    print(f"\n=== Talos one-leg balance ({stance} foot) ===")
    print(f"duration {i*dt:.2f}s  fell={fell}  max tilt={tilt_max:.1f} deg  "
          f"peak |tau|={max(taus):.1f} Nm")
    print(f"raised-foot height {robot.foot_pos(swing)[2]:.3f} m (target {swing_des[2]:.3f})")
    return not fell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foot", default="right", choices=["left", "right"])
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--view", action="store_true")
    args = p.parse_args()
    sys.exit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
