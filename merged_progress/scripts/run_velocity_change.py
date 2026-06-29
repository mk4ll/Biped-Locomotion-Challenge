"""Online velocity following: change walking direction mid-gait.

Demonstrates adaptive navigation by executing sequential plan segments with
different velocity commands.  Each segment replans from the current foot
positions, enabling the robot to:
  1. Walk straight forward
  2. Turn right while walking
  3. Walk straight again

This captures the spirit of the OnlineWalkingController from Kanellos' stack
(event-driven velocity updates) within merged_progress's offline-plan architecture
by replanning at gait-cycle boundaries.

  python scripts/run_velocity_change.py
  python scripts/run_velocity_change.py --viewer
  python scripts/run_velocity_change.py --robot talos
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain
from scripts.run_walk import build_on_terrain, settle


SEGMENTS = [
    # (vx [m/step], vy [m/step], vyaw [rad/step], n_steps, label)
    (0.10, 0.0,  0.0,   6, "straight forward"),
    (0.09, 0.0,  0.18,  6, "turn right while walking"),
    (0.10, 0.0,  0.0,   6, "straight forward again"),
    (0.09, 0.0, -0.15,  4, "veer left"),
    (0.10, 0.0,  0.0,   4, "final straight"),
]


def run(robot="g1", viewer=False, step_timing=False):
    params = load_params()
    params["gait"]["step_length"] = 0.10
    if step_timing:
        params["step_timing"]["enabled"] = True

    terrain = make_terrain("flat")
    env, mcfg = make_robot_env(robot, terrain=terrain)
    ctrl = WalkingController(env, params, terrain=terrain, mcfg=mcfg)
    settle(env, ctrl, terrain, 0.8)

    base = ctrl.base_id
    log = {"x": [], "y": [], "heading": []}
    fell = False
    t_global = 0.0

    # initial foot positions (updated between segments)
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()

    def run_segment(seg_vx, seg_vy, seg_vyaw, seg_steps, label, t_off):
        nonlocal fell, il, ir, com0, t_global
        p = dict(params)
        for k, v in params.items():
            p[k] = v
        p["gait"] = dict(params["gait"])
        p["gait"]["n_steps"] = seg_steps
        p["gait"]["step_length"] = seg_vx

        com_now = env.data.subtree_com[base].copy()
        plan = WalkPlan(p, il.copy(), ir.copy(), com_now,
                        com_height=p["gait"]["com_height"],
                        gravity=p["env"]["gravity"],
                        terrain=terrain,
                        velocity=(seg_vx, seg_vy, seg_vyaw))
        n = int(plan.duration / env.dt)
        print(f"  segment: {label}  ({seg_steps} steps, vx={seg_vx:.2f} vyaw={seg_vyaw:.2f})")

        def loop(i):
            ctrl.step(plan, i * env.dt)
            c = env.data.subtree_com[base]
            log["x"].append(c[0]); log["y"].append(c[1])
            R = env.data.xmat[base].reshape(3, 3)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            log["heading"].append(yaw)
            return env.data.qpos[2] < 0.45

        for i in range(n):
            if fell:
                break
            if loop(i):
                fell = True; break
            t_global += env.dt

        # update foot positions for next segment
        il[:] = env.data.site_xpos[ctrl.left_site].copy()
        ir[:] = env.data.site_xpos[ctrl.right_site].copy()

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for (vx, vy, vyaw, ns, lbl) in SEGMENTS:
                if fell or not v.is_running():
                    break

                # patch run_segment to sync viewer
                p = dict(params)
                for k, val in params.items():
                    p[k] = val
                p["gait"] = dict(params["gait"])
                p["gait"]["n_steps"] = ns
                p["gait"]["step_length"] = vx

                com_now = env.data.subtree_com[base].copy()
                plan = WalkPlan(p, il.copy(), ir.copy(), com_now,
                                com_height=p["gait"]["com_height"],
                                gravity=p["env"]["gravity"], terrain=terrain,
                                velocity=(vx, vy, vyaw))
                n_seg = int(plan.duration / env.dt)
                print(f"  segment: {lbl}  ({ns} steps, vx={vx:.2f} vyaw={vyaw:.2f})")
                for i in range(n_seg):
                    if not v.is_running():
                        break
                    ctrl.step(plan, i * env.dt)
                    c = env.data.subtree_com[base]
                    log["x"].append(c[0]); log["y"].append(c[1])
                    if env.data.qpos[2] < 0.45:
                        fell = True; break
                    v.sync()
                il[:] = env.data.site_xpos[ctrl.left_site].copy()
                ir[:] = env.data.site_xpos[ctrl.right_site].copy()
    else:
        for (vx, vy, vyaw, ns, lbl) in SEGMENTS:
            if fell:
                break
            run_segment(vx, vy, vyaw, ns, lbl, t_global)

    com_final = env.data.subtree_com[base]
    dist = np.hypot(com_final[0] - com0[0], com_final[1] - com0[1])
    total_angle = log["heading"][-1] - log["heading"][0] if log["heading"] else 0.0
    print(f"\n===== Online velocity following ({robot.upper()}) =====")
    print(f"fell             = {fell}")
    print(f"total distance   = {dist:.2f} m")
    print(f"net heading change = {np.degrees(total_angle):.1f} deg")
    ok = (not fell) and dist > 0.3
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1", choices=["g1", "talos"])
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--step-timing", action="store_true",
                    help="enable step timing QP during velocity following")
    args = ap.parse_args()
    run(args.robot, args.viewer, args.step_timing)
