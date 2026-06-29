"""FUN task A — waiter robot: navigate around random tables holding a tray + frappe.

Path planning (artificial potential field) to a goal around randomly placed
circular "tables", converted to a footstep path the DCM+WBC stack walks. The
robot carries a tray with a frappe (a cup geom on a plate, attached to the torso
so it stays level while the torso stays upright -- the frappe doesn't spill).

  python scripts/run_navigate.py                 # random tables, headless + plot
  python scripts/run_navigate.py --seed 3 --viewer
  python scripts/run_navigate.py --robot talos --tables 4
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain
from src.planning import navigation
from run_walk import settle


def tray_decorator(torso_body):
    """Return a decorate(spec, mcfg) that bolts a tray + frappe onto the torso."""
    def deco(spec, mcfg):
        b = spec.body(torso_body)
        # tray plate
        tray = b.add_geom()
        tray.name = "tray"
        tray.type = mujoco.mjtGeom.mjGEOM_BOX
        tray.size = [0.12, 0.11, 0.006]
        tray.pos = [0.26, 0.0, 0.02]
        tray.rgba = [0.75, 0.75, 0.78, 1.0]
        tray.contype = 0; tray.conaffinity = 0          # visual only
        # frappe cup
        cup = b.add_geom()
        cup.name = "frappe_cup"
        cup.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        cup.size = [0.035, 0.055, 0.0]
        cup.pos = [0.26, 0.0, 0.08]
        cup.rgba = [0.55, 0.36, 0.18, 1.0]              # coffee brown
        cup.contype = 0; cup.conaffinity = 0
        # frappe foam
        foam = b.add_geom()
        foam.name = "frappe_foam"
        foam.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        foam.size = [0.036, 0.012, 0.0]
        foam.pos = [0.26, 0.0, 0.145]
        foam.rgba = [0.93, 0.88, 0.75, 1.0]
        foam.contype = 0; foam.conaffinity = 0
        # straw (pink bendy straw sticking out of the foam)
        straw = b.add_geom()
        straw.name = "frappe_straw"
        straw.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        straw.size = [0.003, 0.045, 0.0]   # 3 mm radius, 9 cm total height
        straw.pos = [0.26, 0.018, 0.202]    # offset slightly to the side, above foam
        straw.rgba = [0.95, 0.35, 0.65, 1.0]  # classic pink frappe straw
        straw.contype = 0; straw.conaffinity = 0
    return deco


def _make_walkable_layout(n_tables, start, seed0):
    """Resample slalom courses until the planned weave both clears every table and
    stays within the gait's turn rate (so every random run is actually walkable)."""
    s = seed0
    for _ in range(40):
        tables, goal, path = navigation.make_slalom(n_tables, seed=s)
        # planned path must clear every table footprint (+ robot radius)
        clears = all(min(np.hypot(path[:, 0] - tx, path[:, 1] - ty)) > tr + 0.27
                     for (tx, ty, tr) in tables)
        if clears and navigation.path_curviness(path) < 0.25:
            return tables, goal, path, s
        s += 1
    return tables, goal, path, s          # best effort


def run(robot="g1", n_tables=4, seed=None, viewer=False):
    params = load_params()
    params["gait"]["step_length"] = 0.10        # gentle pace for stable turning
    start = (0.0, 0.0)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 100000))   # random each run
    tables, goal, path, seed = _make_walkable_layout(n_tables, start, seed)
    print(f"(slalom seed = {seed}, {len(tables)} tables)")

    # terrain with the tables as cylindrical obstacles + a goal marker
    terrain = make_terrain("flat", obstacles=tuple(tables), markers=(goal,))
    torso = {"g1": "torso_link", "talos": "torso_2_link"}[robot]
    env, mcfg = make_robot_env(robot, terrain=terrain, decorate=tray_decorator(torso))
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    # Settle into a stable standing pose before walking (critical for stability)
    settle(env, ctrl, terrain, 1.0)

    base = ctrl.base_id
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain, path=path)
    n = int(plan.duration / env.dt)

    log = {"x": [], "y": [], "tilt": []}
    fell = False
    min_clear = np.inf

    def loop(i):
        nonlocal min_clear
        ctrl.step(plan, i * env.dt)
        c = env.data.subtree_com[base]
        log["x"].append(c[0]); log["y"].append(c[1])
        R = env.data.xmat[base].reshape(3, 3)
        log["tilt"].append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        for (tx, ty, tr) in tables:
            min_clear = min(min_clear, np.hypot(c[0] - tx, c[1] - ty) - tr)
        return env.data.qpos[2] < 0.45

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for i in range(n):
                if not v.is_running():
                    break
                if loop(i):
                    fell = True; break
                v.sync()
    else:
        for i in range(n):
            if loop(i):
                fell = True; break

    com = env.data.subtree_com[base]
    reached = np.hypot(com[0] - goal[0], com[1] - goal[1])
    max_tilt = max(log["tilt"])
    print(f"\n===== Waiter robot ({robot.upper()}): navigate around {n_tables} tables =====")
    print(f"tables           = {[(round(x,2),round(y,2),round(r,2)) for x,y,r in tables]}")
    print(f"goal             = ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"fell             = {fell}")
    print(f"dist to goal     = {reached:.2f} m")
    print(f"min obstacle clearance = {min_clear*100:.0f} cm (>0 = no collision)")
    print(f"max torso tilt   = {max_tilt:.1f} deg  -> frappe {'SPILLED!' if max_tilt>12 else 'stayed put :)'}")
    ok = (not fell) and reached < 0.45 and min_clear > 0.0 and max_tilt < 12
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    _plot(start, goal, tables, path, log, robot, seed)
    return ok


def _plot(start, goal, tables, path, log, robot, seed):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    # obstacles (tables)
    for i, (x, y, r) in enumerate(tables):
        ax.add_patch(Circle((x, y), r, color="tab:brown", alpha=0.6))
        ax.add_patch(Circle((x, y), r, fill=False, color="0.3", lw=1.2))
        ax.text(x, y, "table", ha="center", va="center", fontsize=8, color="0.15")
    # planned path
    ax.plot(path[:, 0], path[:, 1], "g--", lw=2.0, label="planned trajectory (slalom weave)")
    # walked CoM (faint, shows it tracked the plan)
    ax.plot(log["x"], log["y"], "-", color="0.55", lw=1.2, alpha=0.8, label="walked CoM")
    # initial & final position of the waiter
    xf, yf = log["x"][-1], log["y"][-1]
    ax.plot(*start, "o", color="tab:blue", ms=14, label="waiter INITIAL position")
    ax.text(start[0], start[1] + 0.09, "start", ha="center", fontsize=9, color="tab:blue")
    ax.plot(xf, yf, "s", color="tab:red", ms=13, label="waiter FINAL position")
    ax.text(xf, yf + 0.09, "end", ha="center", fontsize=9, color="tab:red")
    ax.plot(*goal, "g*", ms=20, label="goal (frappe delivered)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.4); ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Waiter {robot.upper()}: slalom weave around tables — "
                 f"start → end (seed {seed})")
    out = Path(__file__).resolve().parents[1] / "logs" / f"navigate_{robot}_seed{seed}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    ap.add_argument("--tables", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a new random layout each run")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(args.robot, args.tables, args.seed, args.viewer)
