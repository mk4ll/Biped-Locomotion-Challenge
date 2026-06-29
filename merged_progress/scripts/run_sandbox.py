"""Sandbox 'walk forever' mode — flat walk in a continuous viewer loop.

The robot walks flat ground indefinitely. When one fixed-length plan finishes,
a new plan is created from the current foot positions and the robot continues
immediately. Falls (qpos[2] < 0.45) trigger a warning and a brief re-settle
instead of exiting.

Usage:
    python scripts/run_sandbox.py [--robot g1|talos] [--speed slow|normal|fast]
    python scripts/run_sandbox.py --robot g1 --speed fast

Press Ctrl-C or close the MuJoCo viewer window to stop.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco
import mujoco.viewer

from src.utils.config import load_params
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain

# Import helpers from run_walk so we don't duplicate terrain / settle logic.
from scripts.run_walk import build_on_terrain, settle, SPEED_BUNDLES

FALL_THRESH = 0.45   # qpos[2] below this => fall detected


def make_plan(params, env, ctrl, terrain):
    """Build a fresh WalkPlan from the robot's current foot positions."""
    base   = ctrl.base_id
    left   = env.data.site_xpos[ctrl.left_site].copy()
    right  = env.data.site_xpos[ctrl.right_site].copy()
    com0   = env.data.subtree_com[base].copy()
    return WalkPlan(
        params, left, right, com0,
        com_height=params["gait"]["com_height"],
        gravity=params["env"]["gravity"],
        terrain=terrain,
    )


def sandbox(robot="g1", speed=None):
    print("=" * 72)
    print("  SANDBOX MODE -- walk forever (Ctrl-C or close viewer to exit)")
    print("=" * 72)

    params = load_params()
    if speed is not None and speed in SPEED_BUNDLES:
        bundle, needs_mpc = SPEED_BUNDLES[speed]
        for k, v in bundle.items():
            params["gait"][k] = v
        if needs_mpc:
            params["dcm_mpc"]["enabled"] = True
            params["capture"]["max_shift"] = 0.14

    env, ctrl, terrain = build_on_terrain(params, "flat", robot=robot)
    settle(env, ctrl, terrain, 0.8)
    print(f"  robot={robot.upper()}  speed={speed or 'normal (default)'}")
    print("  Starting first plan cycle...\n")

    cycle      = 0
    fall_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            cycle += 1
            plan = make_plan(params, env, ctrl, terrain)
            n    = int(plan.duration / env.dt)
            print(f"[cycle {cycle:4d}]  plan duration={plan.duration:.2f}s  "
                  f"steps={n}  falls_so_far={fall_count}")

            fell = False
            for i in range(n):
                if not v.is_running():
                    break

                ctrl.step(plan, i * env.dt)
                v.sync()

                if env.data.qpos[2] < FALL_THRESH:
                    fell = True
                    break

            if not v.is_running():
                break

            if fell:
                fall_count += 1
                print(f"  WARNING: fall detected (qpos[2]={env.data.qpos[2]:.3f} "
                      f"< {FALL_THRESH}). Re-settling... (fall #{fall_count})")
                settle(env, ctrl, terrain, 1.2)
                print("  Re-settle done. Resuming walk.")

    print(f"\nSandbox ended after {cycle} plan cycles, {fall_count} fall(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Walk forever on flat ground (sandbox mode)."
    )
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"],
                    help="robot model to use")
    ap.add_argument("--speed", default=None, choices=["slow", "normal", "fast"],
                    help="gait speed preset (slow~0.15, normal~0.27, fast~0.47 m/s)")
    args = ap.parse_args()
    sandbox(robot=args.robot, speed=args.speed)
