"""Omnidirectional walking (merged): forward / back / strafe / turn / curve.

Same DCM + WBC stack, driven by a body-frame velocity command (vx, vy, vyaw).
The footstep planner places steps in the commanded direction; the pelvis yaw
tracks the heading.

  python scripts/run_omni.py --vx 0.12              # forward
  python scripts/run_omni.py --vx -0.10             # backward
  python scripts/run_omni.py --vy 0.08              # strafe left
  python scripts/run_omni.py --vx 0.10 --vyaw 0.15  # walk a curve
  python scripts/run_omni.py --vyaw 0.2 --viewer    # turn in place
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan


def run(vx=0.0, vy=0.0, vyaw=0.0, viewer=False, robot="g1"):
    params = load_params()
    env, mcfg = make_robot_env(robot)
    ctrl = WalkingController(env, params, mcfg=mcfg)
    base = ctrl.base_id
    init_left = env.data.site_xpos[ctrl.left_site].copy()
    init_right = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], velocity=(vx, vy, vyaw))
    n = int(plan.duration / env.dt)
    p0 = com0[:2].copy()
    yaw0 = np.arctan2(env.data.xmat[base].reshape(3, 3)[1, 0],
                      env.data.xmat[base].reshape(3, 3)[0, 0])

    def loop(i):
        ctrl.step(plan, i * env.dt)
        return env.data.qpos[2] < 0.45

    fell = False
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
    R = env.data.xmat[base].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    disp = com[:2] - p0
    print(f"\n===== Omnidirectional walk (vx={vx}, vy={vy}, vyaw={vyaw}) =====")
    print(f"fell           = {fell}")
    print(f"displacement   = dx {disp[0]:+.3f} m, dy {disp[1]:+.3f} m")
    print(f"net yaw turn   = {np.degrees(yaw - yaw0):+.1f} deg")
    ok = (not fell) and (np.linalg.norm(disp) > 0.15 or abs(yaw - yaw0) > 0.15)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vyaw", type=float, default=0.0)
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(args.vx, args.vy, args.vyaw, args.viewer, args.robot)
