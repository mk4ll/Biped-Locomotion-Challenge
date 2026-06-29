"""One-leg balance demo: G1 stands stably on one foot.

Phases
------
  0–1.5 s  : double support, shift CoM laterally over the stance foot
  1.5 s+   : single-support — swing foot held up at hip height, DCM balanced
              over the stance foot

The whole-body QP is the same inverse-dynamics solver used for walking; the
only change vs. the walking controller is (a) only the stance foot's corner
points are in the contact set, and (b) we add a task to track the raised
foot at a desired height instead of a swing trajectory.

Usage
-----
  python scripts/run_balance.py               # stand on right foot, headless
  python scripts/run_balance.py --foot left   # stand on left foot
  python scripts/run_balance.py --view        # interactive MuJoCo viewer
  python scripts/run_balance.py --seconds 8   # longer run
  mjpython scripts/run_balance.py --view      # macOS (mjpython required)
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.wbc import Task, Contact, WholeBodyQP


# --------------------------------------------------------------------------
# Gains
# --------------------------------------------------------------------------
COM_KP       = 150.0
COM_KD       = 26.0
ORI_KP       = 220.0
ORI_KD       = 30.0
POST_KP      = 60.0
POST_KD      = 12.0
FOOT_KP      = 350.0   # raised-foot position task
FOOT_KD      = 40.0
FOOT_ORI_KP  = 200.0
FOOT_ORI_KD  = 28.0
W_COM        = 250.0
W_ORI        = 5.0
W_POST       = 0.2
W_FOOT       = 50.0
W_FOOT_ORI   = 6.0
MU           = 0.5

# Phase durations
T_SHIFT = 1.8   # s to shift CoM over stance foot before lifting


def build_selection_matrix(robot):
    S = np.zeros((robot.nu, robot.nv))
    for k, vadr in enumerate(robot.act_vadr):
        S[k, vadr] = 1.0
    return S


def run(args):
    terr = terrain_mod.Flat()
    model, data = load_g1(terrain=terr)
    robot = Robot(model, data)
    q_nom = reset_to_crouch(robot, terr)

    stance = args.foot
    swing  = "left" if stance == "right" else "right"

    qp = WholeBodyQP(robot)
    S  = build_selection_matrix(robot)
    quat_des = q_nom[3:7].copy()

    # --- desired CoM: above the stance foot with slight inward offset ------
    stance_foot_xy = robot.foot_pos(stance)[:2].copy()
    com_init = robot.com().copy()
    z_des    = float(com_init[2])

    # target: CoM directly above stance foot (+ tiny inward nudge for margin)
    lateral_sign = 1.0 if stance == "right" else -1.0   # right foot is at +y
    com_target = np.array([stance_foot_xy[0],
                           stance_foot_xy[1] + lateral_sign * 0.02,
                           z_des])

    # raised foot target: same x/y but clearance height above ground
    swing_foot_init = robot.foot_pos(swing).copy()
    swing_foot_des  = swing_foot_init.copy()
    swing_foot_des[2] = terr.height(swing_foot_des[0], swing_foot_des[1]) + 0.18

    dt = model.opt.timestep
    n  = int(args.seconds / dt)

    # ---- viewer (optional) -----------------------------------------------
    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)

    tilt_max = 0.0
    taus     = []
    fell     = False

    for i in range(n):
        t = i * dt

        # Linearly ramp CoM target from initial to above-stance over T_SHIFT
        alpha  = min(t / T_SHIFT, 1.0)
        com_des = (1.0 - alpha) * com_init + alpha * com_target

        # ---- CoM task -------------------------------------------------------
        Jcom = robot.com_jacobian()
        com  = robot.com()
        vcom = robot.com_vel(Jcom)
        acc_com = COM_KP * (com_des - com) - COM_KD * vcom
        task_com = Task(Jcom, acc_com, W_COM)

        # ---- pelvis orientation task ----------------------------------------
        Jr    = robot.body_angular_jacobian(robot.pelvis_id)
        omega = Jr @ data.qvel
        err_q = np.zeros(3)
        mujoco.mju_subQuat(err_q, quat_des, robot.pelvis_quat())
        task_ori = Task(Jr, ORI_KP * err_q - ORI_KD * omega, W_ORI)

        # ---- posture task ---------------------------------------------------
        q = data.qpos
        acc_p = np.zeros(robot.nv)
        for grp in robot.groups.values():
            for qa, va in zip(grp.qadr, grp.vadr):
                acc_p[va] = POST_KP * (q_nom[qa] - q[qa]) - POST_KD * data.qvel[va]
        task_post = Task(S, acc_p[robot.act_vadr], W_POST)

        tasks = [task_com, task_ori, task_post]

        # ---- raised-foot task (active once CoM shift is complete) -----------
        if alpha >= 1.0:
            Jp, Jrf = robot.site_jacobian(swing)
            foot_pos = robot.foot_pos(swing)
            bid      = robot.foot_body[swing]
            v        = data.qvel
            Jdv      = robot.point_jacobian_dot(foot_pos, bid) @ v
            foot_vel = Jp @ v
            acc_foot = (FOOT_KP * (swing_foot_des - foot_pos)
                        - FOOT_KD * foot_vel)
            tasks.append(Task(Jp, acc_foot - Jdv, W_FOOT))

            # keep foot level (z-axis up)
            quat_foot = np.zeros(4)
            mujoco.mju_mat2Quat(quat_foot, robot.foot_rot(swing).flatten())
            err_f = np.zeros(3)
            mujoco.mju_subQuat(err_f, quat_des, quat_foot)   # upright
            omega_f = Jrf @ v
            tasks.append(Task(Jrf,
                              FOOT_ORI_KP * err_f - FOOT_ORI_KD * omega_f,
                              W_FOOT_ORI))

        # ---- contacts: stance foot only after weight shift ------------------
        if alpha < 1.0:
            contact_sides = ("left", "right")
        else:
            contact_sides = (stance,)

        contacts = []
        for side in contact_sides:
            bid = robot.foot_body[side]
            for p in robot.foot_corner_points(side):
                nrm = terr.normal(p[0], p[1])
                contacts.append(Contact(p, bid, nrm, MU))

        sol = qp.solve(tasks, contacts)
        tau = sol["tau"] if sol is not None else np.zeros(robot.nu)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        zaxis = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt  = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        taus.append(float(np.max(np.abs(tau))))

        if data.qpos[2] < 0.35 or tilt > 45:
            fell = True
            print(f"[t={t:.2f}s] FELL (pelvis z={data.qpos[2]:.2f}, tilt={tilt:.1f}°)")
            break

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()

    if viewer is not None:
        viewer.close()

    print(f"\n=== One-leg balance ({stance} foot) ===")
    print(f"duration: {i * dt:.2f} s  |  fell: {fell}")
    print(f"max tilt: {tilt_max:.1f}°  |  peak |tau|: {max(taus):.1f} Nm")
    swing_z = robot.foot_pos(swing)[2]
    print(f"raised foot height: {swing_z:.3f} m  (target {swing_foot_des[2]:.3f} m)")
    return not fell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foot", default="right", choices=["left", "right"],
                   help="which foot to stand on")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--view", action="store_true",
                   help="open interactive MuJoCo viewer (use mjpython on macOS)")
    args = p.parse_args()
    ok = run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
