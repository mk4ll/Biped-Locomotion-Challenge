"""Turn-in-place test: pure yaw rotation, no forward translation.

Uses the omnidirectional footstep planner with vx=vy=0 and vyaw=0.2 rad/s.
12 steps → ~87° headless. Modelled on run_omni.py (no pre-settle, which
actually improves rotational stability). 87° is the certified stable limit:
adding more steps or a faster rate causes falls.

  python scripts/run_turn.py          # ~87 deg turn, headless
  python scripts/run_turn.py --viewer # same, with live viewer
  python scripts/run_turn.py --robot talos
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for sibling script imports

import numpy as np

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain

VYAW    = 0.20   # yaw rate [rad/s]
N_STEPS = 12     # gives ~87° at these timings (without pre-settle)
T_SS    = 0.52
T_DS    = 0.15


def run(viewer=False, robot="g1"):
    params = load_params()
    params["gait"]["t_ss"]    = T_SS
    params["gait"]["t_ds"]    = T_DS
    params["gait"]["n_steps"] = N_STEPS

    terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    # No pre-settle: matches run_omni.py behaviour; pre-settling actually
    # reduces rotational stability by shifting the pelvis position.

    base = ctrl.base_id
    il   = env.data.site_xpos[ctrl.left_site].copy()
    ir   = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()

    R0   = env.data.xmat[base].reshape(3, 3)
    yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
    p0   = com0[:2].copy()

    T_step   = T_SS + T_DS
    expected = np.degrees(VYAW * T_step * N_STEPS)

    plan = WalkPlan(params, il, ir, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"],
                    velocity=(0.0, 0.0, VYAW))
    n = int(plan.duration / env.dt)

    fell = False

    def loop(i):
        ctrl.step(plan, i * env.dt)
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

    R = env.data.xmat[base].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    turned = np.degrees(np.arctan2(np.sin(yaw - yaw0), np.cos(yaw - yaw0)))
    drift  = np.linalg.norm(env.data.subtree_com[base][:2] - p0)

    print(f"\n===== Turn in place (robot={robot.upper()}, vyaw={VYAW} rad/s, {N_STEPS} steps) =====")
    print(f"fell             = {fell}")
    print(f"net yaw turned   = {turned:+.1f} deg  (expected ~{expected:.0f} deg)")
    print(f"lateral drift    = {drift*100:.1f} cm  (ideal = 0)")

    ok = (not fell) and abs(turned) > 45.0
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(args.viewer, args.robot)
