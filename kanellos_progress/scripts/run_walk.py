"""Phase 2/3 demo + test: DCM walking (in-place march or forward) for the G1.

Examples
--------
  python scripts/run_walk.py --step-len 0.0 --n-steps 8        # march in place
  python scripts/run_walk.py --step-len 0.15 --n-steps 12      # walk forward
  python scripts/run_walk.py --step-len 0.15 --terrain incline --angle 6
  python scripts/run_walk.py --step-len 0.15 --video            # save logs/walk.gif
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
from biped.controllers import WalkingController


def build(args):
    clr = args.clearance
    if args.terrain == "incline":
        terr = terrain_mod.make("incline", angle=np.deg2rad(args.angle))
    elif args.terrain == "stairs":
        # grid-aligned stairs: tread depth = step length, modest rise
        terr = terrain_mod.Stairs(rise=0.03, run=args.step_len, n_steps=4,
                                  x0=0.525, width=1.6)
        clr = max(clr, terr.rise + 0.05)
    else:
        terr = terrain_mod.make(args.terrain)
    model, data = load_g1(terrain=terr)
    robot = Robot(model, data)
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
    total = ctrl.plan.total_time + 1.0  # tail to settle
    n = int(total / dt)

    renderer = cam = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 130, -12, 2.8
        every = max(1, int(round(1.0 / (30 * dt))))

    L = {k: [] for k in ("t", "comx", "comy", "comz", "xix", "xiy",
                          "zmpx", "zmpy", "pcmdx", "pcmdy", "fl", "fr", "tilt",
                          "tau_max", "phase")}
    push_step = int(args.push_t / dt)
    push_dur = int(0.1 / dt)  # 100 ms impulse

    fell = False
    for i in range(n):
        t = i * dt
        tau, sol = ctrl(t)
        data.ctrl[:] = tau
        data.xfrc_applied[:] = 0.0
        if args.push and push_step <= i < push_step + push_dur:
            data.xfrc_applied[robot.pelvis_id, 1] = args.push  # lateral shove
        mujoco.mj_step(model, data)

        com = robot.com()
        zaxis = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
        sp = sol["plan"]
        for k, val in (("t", t), ("comx", com[0]), ("comy", com[1]), ("comz", com[2]),
                       ("xix", sol["xi"][0]), ("xiy", sol["xi"][1]),
                       ("zmpx", sp["zmp"][0]), ("zmpy", sp["zmp"][1]),
                       ("pcmdx", sol["p_cmd"][0]), ("pcmdy", sol["p_cmd"][1]),
                       ("fl", robot.foot_contact_force("left")),
                       ("fr", robot.foot_contact_force("right")),
                       ("tilt", tilt), ("tau_max", float(np.max(np.abs(tau)))),
                       ("phase", sp["phase"])):
            L[k].append(val)

        if renderer is not None and i % every == 0:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

        if data.qpos[2] < 0.4 or tilt > 50:
            fell = True
            print(f"[t={t:.2f}] FELL (pelvis z={data.qpos[2]:.2f}, tilt={tilt:.1f})")
            break

    A = {k: np.array(v) for k, v in L.items()}
    dist = data.qpos[0] - base_x0
    dcm_err = np.hypot(A["xix"] - A["pcmdx"], A["xiy"] - A["pcmdy"])  # rough
    print("\n=== Walking summary ===")
    print(f"terrain={terr.name}  mode={'march' if args.step_len==0 else 'forward'}  "
          f"steps={args.n_steps}  fell={fell}")
    print(f"sim time={A['t'][-1]:.2f}s / plan {ctrl.plan.total_time:.2f}s")
    print(f"forward distance (base x): {dist:+.3f} m")
    print(f"CoM height mean={A['comz'].mean():.3f} (target {ctrl.z_des:.3f})  "
          f"max tilt={A['tilt'].max():.1f} deg")
    print(f"foot force L/R mean: {A['fl'].mean():.0f}/{A['fr'].mean():.0f} N")
    print(f"peak |tau|: {A['tau_max'].max():.1f} Nm")

    if args.plot:
        save_plot(A, terr, args)
    if frames:
        import imageio.v2 as imageio
        out = args.video if isinstance(args.video, str) else \
            os.path.join("logs", f"walk_{terr.name}_{'march' if args.step_len==0 else 'fwd'}.gif")
        imageio.mimsave(out, frames, fps=30, loop=0)
        print(f"saved {out} ({len(frames)} frames)")
    return not fell


def save_plot(A, terr, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(A["comy"], A["comx"], label="CoM")
    ax[0, 0].plot(A["xiy"], A["xix"], "--", label="DCM")
    ax[0, 0].plot(A["zmpy"], A["zmpx"], ":", label="ZMP ref")
    ax[0, 0].set_title("top-down (x vs y)"); ax[0, 0].set_xlabel("y"); ax[0, 0].set_ylabel("x")
    ax[0, 0].legend(); ax[0, 0].axis("equal")
    ax[0, 1].plot(A["t"], A["comy"], label="CoM y")
    ax[0, 1].plot(A["t"], A["xiy"], "--", label="DCM y")
    ax[0, 1].plot(A["t"], A["zmpy"], ":", label="ZMP y")
    ax[0, 1].set_title("lateral vs time"); ax[0, 1].legend()
    ax[1, 0].plot(A["t"], A["fl"], label="L"); ax[1, 0].plot(A["t"], A["fr"], label="R")
    ax[1, 0].set_title("foot normal force"); ax[1, 0].legend()
    ax[1, 1].plot(A["t"], A["comz"]); ax[1, 1].plot(A["t"], A["tilt"])
    ax[1, 1].set_title("CoM height & tilt(deg)")
    fig.suptitle(f"G1 DCM walking - {terr.name}")
    fig.tight_layout()
    out = os.path.join("logs", f"walk_{terr.name}.png")
    fig.savefig(out, dpi=110); print(f"saved {out}")


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
    p.add_argument("--reactive", action="store_true", help="capture-point stepping")
    p.add_argument("--mpc", action="store_true",
                   help="DCM preview-MPC for the CoP (vs one-step feedback)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--video", nargs="?", const=True, default=False)
    args = p.parse_args()
    ok = run(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
