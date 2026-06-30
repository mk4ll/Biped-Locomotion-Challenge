"""Sandbox: extended flat walk — compare DCM control modes.

Walks flat ground for ~40 steps in the viewer (single shot, no infinite loop).
Supports all three DCM control modes so you can directly compare stability:

  python scripts/run_sandbox.py                         # proportional DCM (default)
  python scripts/run_sandbox.py --mpc                   # DCM preview-MPC
  python scripts/run_sandbox.py --step-timing           # Khadiv step-timing QP
  python scripts/run_sandbox.py --arm-swing             # add contralateral arm swing
  python scripts/run_sandbox.py --speed fast --mpc      # fast speed + MPC

Press Ctrl-C or close the viewer to stop early.
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

from scripts.run_walk import build_on_terrain, settle, SPEED_BUNDLES

FALL_THRESH = 0.45
N_STEPS_SANDBOX = 40   # enough steps to see stability over a good stretch


def sandbox(robot="g1", speed=None, mpc=False, step_timing=False,
            arm_swing=False, viewer=False):
    print("=" * 72)
    print("  SANDBOX MODE — extended flat walk")
    print("=" * 72)

    params = load_params()

    if speed is not None and speed in SPEED_BUNDLES:
        bundle, needs_mpc = SPEED_BUNDLES[speed]
        for k, v in bundle.items():
            params["gait"][k] = v
        if needs_mpc:
            mpc = True   # fast preset requires MPC

    if mpc:
        params["dcm_mpc"]["enabled"] = True
        params["capture"]["max_shift"] = 0.14
    if step_timing:
        params["step_timing"]["enabled"] = True
    if arm_swing:
        params["arm_swing"]["enabled"] = True

    params["gait"]["n_steps"] = N_STEPS_SANDBOX

    mode_str = ("DCM preview-MPC" if mpc
                else "Step-timing QP" if step_timing
                else "Proportional DCM")
    print(f"  robot={robot.upper()}  speed={speed or 'normal'}  control={mode_str}")
    print()

    env, ctrl, terrain = build_on_terrain(params, "flat", robot=robot)
    settle(env, ctrl, terrain, 0.8)

    base = ctrl.base_id
    init_left  = env.data.site_xpos[ctrl.left_site].copy()
    init_right = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    x0   = com0[0]

    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"],
                    terrain=terrain)
    n = int(plan.duration / env.dt)

    fell = False

    if viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for i in range(n):
                if not v.is_running():
                    break
                ctrl.step(plan, i * env.dt)
                v.sync()
                if env.data.qpos[2] < FALL_THRESH:
                    fell = True
                    break
    else:
        for i in range(n):
            ctrl.step(plan, i * env.dt)
            if env.data.qpos[2] < FALL_THRESH:
                fell = True
                break

    dist = env.data.subtree_com[base][0] - x0
    print(f"\n===== Sandbox: {robot.upper()} / {mode_str} =====")
    print(f"fell             = {fell}")
    print(f"forward distance = {dist:.3f} m  ({N_STEPS_SANDBOX} steps planned)")
    print(f"\nRESULT: {'PASS' if not fell and dist > 0.5 else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extended flat walk — compare DCM control modes."
    )
    ap.add_argument("--robot",  default="g1", choices=["g1", "talos"])
    ap.add_argument("--speed",  default=None, choices=["slow", "normal", "fast"])
    ap.add_argument("--mpc",    action="store_true",
                    help="use DCM preview-MPC instead of proportional law")
    ap.add_argument("--step-timing", action="store_true",
                    help="use step-timing QP (Khadiv et al.) for footstep + timing")
    ap.add_argument("--arm-swing", action="store_true",
                    help="enable contralateral arm swing")
    ap.add_argument("--viewer", action="store_true",
                    help="open the MuJoCo viewer")
    args = ap.parse_args()
    sandbox(robot=args.robot, speed=args.speed, mpc=args.mpc,
            step_timing=args.step_timing, arm_swing=args.arm_swing,
            viewer=args.viewer)
