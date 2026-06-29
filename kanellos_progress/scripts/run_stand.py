"""Phase 1 demo / test: G1 standing balance with optional weight shift + push.

Examples
--------
  python scripts/run_stand.py                       # 5 s flat, headless, report
  python scripts/run_stand.py --shift --seconds 8   # lateral CoM weight shift
  python scripts/run_stand.py --terrain incline --angle 8
  python scripts/run_stand.py --push 120            # 120 N lateral shove at t=2 s
  python scripts/run_stand.py --view                # interactive viewer
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
from biped.controllers import StandingController


def build(args):
    terr = terrain_mod.make(args.terrain, angle=np.deg2rad(args.angle)) \
        if args.terrain == "incline" else terrain_mod.make(args.terrain)
    model, data = load_g1(terrain=terr)
    robot = Robot(model, data)
    q_nom = reset_to_crouch(robot, terr)
    ctrl = StandingController(robot, terr, q_nom=q_nom)
    return model, data, robot, ctrl, terr


def com_target_schedule(t, args, com0):
    """Lateral weight-shift demo: sinusoidal sway between the feet."""
    if not args.shift:
        return com0[:2]
    amp = args.shift_amp
    y = com0[1] + amp * np.sin(2 * np.pi * t / args.shift_period)
    return np.array([com0[0], y])


def run(args):
    model, data, robot, ctrl, terr = build(args)
    dt = model.opt.timestep
    n = int(args.seconds / dt)
    com0 = robot.com().copy()

    log = {k: [] for k in ("t", "comx", "comy", "comz", "pelz", "tilt",
                           "fl", "fr", "tau_max")}
    push_step = int(args.push_t / dt)
    push_dur = int(0.1 / dt)  # 100 ms impulse

    viewer = None
    if args.view:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)

    fell = False
    for i in range(n):
        t = i * dt
        ctrl.set_com_target(com_target_schedule(t, args, com0))
        tau, sol = ctrl()
        if sol is None:
            print(f"[t={t:.3f}] QP infeasible")
        data.ctrl[:] = tau

        # external push: lateral force on the pelvis for a short window
        data.xfrc_applied[:] = 0.0
        if args.push and push_step <= i < push_step + push_dur:
            data.xfrc_applied[robot.pelvis_id, 1] = args.push

        mujoco.mj_step(model, data)

        com = robot.com()
        up = robot.foot_rot  # noqa
        # tilt = angle of pelvis z-axis from world up
        zaxis = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
        log["t"].append(t)
        log["comx"].append(com[0]); log["comy"].append(com[1]); log["comz"].append(com[2])
        log["pelz"].append(data.qpos[2]); log["tilt"].append(tilt)
        log["fl"].append(robot.foot_contact_force("left"))
        log["fr"].append(robot.foot_contact_force("right"))
        log["tau_max"].append(float(np.max(np.abs(tau))))

        if data.qpos[2] < 0.4 or tilt > 45:
            fell = True
            print(f"[t={t:.3f}] FELL (pelvis z={data.qpos[2]:.2f}, tilt={tilt:.1f} deg)")
            break

        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    L = {k: np.array(v) for k, v in log.items()}
    print("\n=== Phase 1 standing summary ===")
    print(f"terrain={terr.name}  duration simulated={L['t'][-1]:.2f}s  fell={fell}")
    print(f"pelvis height: mean={L['pelz'].mean():.3f}  min={L['pelz'].min():.3f}")
    print(f"max tilt: {L['tilt'].max():.2f} deg")
    print(f"CoM x drift: {L['comx'][-1]-com0[0]:+.3f} m   "
          f"CoM y range: [{L['comy'].min():.3f}, {L['comy'].max():.3f}]")
    print(f"foot force L/R mean: {L['fl'].mean():.0f}/{L['fr'].mean():.0f} N  "
          f"(total weight ~{robot.total_mass*9.81:.0f} N)")
    print(f"peak |tau|: {L['tau_max'].max():.1f} Nm")

    if args.plot:
        save_plot(L, com0, args, terr)
    return not fell


def save_plot(L, com0, args, terr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(L["t"], L["comy"], label="CoM y")
    ax[0, 0].set_title("CoM lateral"); ax[0, 0].set_xlabel("s"); ax[0, 0].legend()
    ax[0, 1].plot(L["t"], L["pelz"]); ax[0, 1].set_title("pelvis height"); ax[0, 1].set_xlabel("s")
    ax[1, 0].plot(L["t"], L["tilt"]); ax[1, 0].set_title("pelvis tilt (deg)"); ax[1, 0].set_xlabel("s")
    ax[1, 1].plot(L["t"], L["fl"], label="L"); ax[1, 1].plot(L["t"], L["fr"], label="R")
    ax[1, 1].set_title("foot normal force"); ax[1, 1].set_xlabel("s"); ax[1, 1].legend()
    fig.suptitle(f"G1 standing balance - {terr.name}")
    fig.tight_layout()
    out = os.path.join("logs", f"stand_{terr.name}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    p.add_argument("--angle", type=float, default=8.0, help="incline degrees")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--shift", action="store_true", help="lateral weight-shift demo")
    p.add_argument("--shift-amp", type=float, default=0.06)
    p.add_argument("--shift-period", type=float, default=3.0)
    p.add_argument("--push", type=float, default=0.0, help="lateral push force (N)")
    p.add_argument("--push-t", type=float, default=2.0)
    p.add_argument("--view", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    ok = run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
