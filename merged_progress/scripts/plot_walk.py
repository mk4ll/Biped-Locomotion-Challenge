"""Lecture-style locomotion plots (merged).

Runs a walk and produces the figures used in the course lectures:
  (A) Path planning / footstep placement (top-down x-y): planned footsteps as
      foot rectangles + actual foot landings + CoM / DCM / ZMP paths.
  (B) CoM height vs forward position, with the terrain surface (shows climbing).
  (C) Sagittal (x) and lateral (y) CoM / DCM / ZMP vs time.
  (D) Swing-foot height vs time.

  python scripts/plot_walk.py --terrain flat
  python scripts/plot_walk.py --terrain incline --angle 12
  python scripts/plot_walk.py --terrain stairs
  python scripts/plot_walk.py --omni curve        # vx=0.1, vyaw=0.12
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.utils.config import load_params
from src.sim.mujoco_env import make_env_from_params
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_walk import build_on_terrain, settle

# foot half-extents from g1.xml corner geoms (relative to the foot site)
FOOT_X = (-0.05, 0.12)
FOOT_Y = (-0.03, 0.03)


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def run_and_log(terrain_name, angle_deg, omni, robot="g1"):
    params = load_params()
    velocity = None
    if omni:
        velocity = {"forward": (0.12, 0, 0), "strafe": (0, 0.08, 0),
                    "curve": (0.10, 0, 0.12), "back": (-0.10, 0, 0)}[omni]
    if terrain_name == "stairs":
        g = params["gait"]; g["step_length"] = 0.16; g["swing_apex"] = 0.06; g["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, terrain_name, angle_deg,
                                          dict(rise=0.025, run=0.16, n_steps=6, x0=0.30), robot)
    settle(env, ctrl, terrain, 0.8)
    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain, velocity=velocity)
    n = int(plan.duration / env.dt)

    log = {k: [] for k in ["t", "cx", "cy", "cz", "cxr", "cyr", "czr",
                           "dx", "dy", "zx", "zy", "lx", "ly", "lz", "rx", "ry", "rz",
                           "swing"]}
    for i in range(n):
        t = i * env.dt
        ref, _ = ctrl.step(plan, t)
        c = env.data.subtree_com[base]
        lf = env.data.site_xpos[ctrl.left_site]
        rf = env.data.site_xpos[ctrl.right_site]
        log["t"].append(t)
        log["cx"].append(c[0]); log["cy"].append(c[1]); log["cz"].append(c[2])
        log["cxr"].append(ref["com"][0]); log["cyr"].append(ref["com"][1]); log["czr"].append(ref["com"][2])
        log["dx"].append(ref["dcm"][0]); log["dy"].append(ref["dcm"][1])
        log["zx"].append(ref["zmp"][0]); log["zy"].append(ref["zmp"][1])
        log["lx"].append(lf[0]); log["ly"].append(lf[1]); log["lz"].append(lf[2])
        log["rx"].append(rf[0]); log["ry"].append(rf[1]); log["rz"].append(rf[2])
        log["swing"].append(ref["swing"])
        if env.data.qpos[2] - terrain.height(c[0], 0.0) < 0.45:
            break
    return params, terrain, plan, {k: np.array(v) if k != "swing" else v
                                   for k, v in log.items()}


def make_plots(terrain_name, angle_deg, omni, plan, terrain, log, robot="g1"):
    label = robot.upper() + " — " + (omni or terrain_name +
            (f" {angle_deg:.0f}deg" if terrain_name == "incline" else ""))
    fig = plt.figure(figsize=(13, 9))

    # (A) top-down path planning + footsteps
    axA = fig.add_subplot(2, 2, 1)
    for fs in plan.footsteps:
        col = "tab:blue" if fs["foot"] == "left" else "tab:red"
        th = fs.get("heading", 0.0)
        corners = np.array([[FOOT_X[0], FOOT_Y[0]], [FOOT_X[1], FOOT_Y[0]],
                            [FOOT_X[1], FOOT_Y[1]], [FOOT_X[0], FOOT_Y[1]],
                            [FOOT_X[0], FOOT_Y[0]]])
        corners = (_rot(th) @ corners.T).T + fs["pos"][:2]
        axA.plot(corners[:, 0], corners[:, 1], color=col, lw=1.3)
        axA.plot(fs["pos"][0], fs["pos"][1], "o", color=col, ms=3)
    axA.plot(log["cx"], log["cy"], "k-", lw=1.6, label="CoM (actual)")
    axA.plot(log["dx"], log["dy"], "g--", lw=1, label="DCM ref")
    axA.plot(log["zx"], log["zy"], "m:", lw=1.2, label="ZMP ref")
    axA.set_aspect("equal"); axA.grid(True, alpha=0.4); axA.legend(fontsize=8)
    axA.set_xlabel("x [m]"); axA.set_ylabel("y [m]")
    axA.set_title(f"(A) Path planning & footstep placement — {label}\n"
                  "blue=left foot, red=right foot")

    # (B) CoM height vs forward position + terrain surface
    axB = fig.add_subplot(2, 2, 2)
    xs = np.linspace(log["cx"].min() - 0.1, log["cx"].max() + 0.1, 400)
    surf = np.array([terrain.height(x, 0.0) for x in xs])
    axB.fill_between(xs, surf - 0.02, surf, color="0.7", label="terrain surface")
    axB.plot(log["cx"], log["cz"], "k-", lw=1.6, label="CoM z (actual)")
    axB.plot(log["cxr"], log["czr"], "r--", lw=1.1, label="CoM z (ref)")
    axB.grid(True, alpha=0.4); axB.legend(fontsize=8)
    axB.set_xlabel("CoM x [m]"); axB.set_ylabel("z [m]")
    axB.set_title("(B) CoM height (climbs the terrain)")

    # (C) sagittal + lateral vs time
    axC = fig.add_subplot(2, 2, 3)
    t = log["t"]
    axC.plot(t, log["cx"], "k", label="CoM x")
    axC.plot(t, log["dx"], "g--", lw=0.8, label="DCM x")
    axC.plot(t, log["cy"], "tab:purple", label="CoM y")
    axC.plot(t, log["dy"], "c--", lw=0.8, label="DCM y")
    axC.plot(t, log["zy"], "m:", lw=1, label="ZMP y")
    axC.grid(True, alpha=0.4); axC.legend(fontsize=7, ncol=2)
    axC.set_xlabel("t [s]"); axC.set_ylabel("position [m]")
    axC.set_title("(C) CoM / DCM / ZMP vs time")

    # (D) swing-foot height vs time
    axD = fig.add_subplot(2, 2, 4)
    axD.plot(t, log["lz"], "tab:blue", label="left foot z")
    axD.plot(t, log["rz"], "tab:red", label="right foot z")
    axD.grid(True, alpha=0.4); axD.legend(fontsize=8)
    axD.set_xlabel("t [s]"); axD.set_ylabel("foot z [m]")
    axD.set_title("(D) Foot height (swing clearance)")

    fig.tight_layout()
    name = "plot_walk_" + robot + "_" + (omni or terrain_name +
                           (f"_{angle_deg:.0f}deg" if terrain_name == "incline" else ""))
    out = Path(__file__).resolve().parents[1] / "logs" / f"{name}.png"
    fig.savefig(out, dpi=115)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="flat", choices=["flat", "incline", "stairs"])
    ap.add_argument("--angle", type=float, default=12.0)
    ap.add_argument("--omni", default=None,
                    choices=["forward", "back", "strafe", "curve"])
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    args = ap.parse_args()
    params, terrain, plan, log = run_and_log(args.terrain, args.angle, args.omni, args.robot)
    make_plots(args.terrain, args.angle, args.omni, plan, terrain, log, args.robot)


if __name__ == "__main__":
    main()
