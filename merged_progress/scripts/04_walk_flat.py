"""Stage 4 -- dynamic walking on flat ground.

Drives the WBC with the DCM walking plan and reports whether the robot walks
a continuous, stable gait (>= 8 steps) without falling.

Run headless (metrics):  python scripts/04_walk_flat.py
With viewer:             python scripts/04_walk_flat.py --viewer
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_env_from_params
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan


def build():
    params = load_params()
    env = make_env_from_params("scene_flat")
    ctrl = WalkingController(env, params)
    init_left = env.data.site_xpos[ctrl.left_site].copy()
    init_right = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[ctrl.base_id].copy()
    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"])
    return params, env, ctrl, plan


def run(viewer=False):
    params, env, ctrl, plan = build()
    dt = env.dt
    n_steps = int(plan.duration / dt)

    log = {k: [] for k in ["t", "com_x", "com_y", "com_z", "com_x_ref", "com_y_ref",
                           "zmp_x", "zmp_y", "base_z", "fz", "tau_max", "ok", "support"]}

    def record(t, ref, res):
        com = env.data.subtree_com[ctrl.base_id]
        log["t"].append(t)
        log["com_x"].append(com[0]); log["com_y"].append(com[1]); log["com_z"].append(com[2])
        log["com_x_ref"].append(ref["com"][0]); log["com_y_ref"].append(ref["com"][1])
        log["zmp_x"].append(ref["zmp"][0]); log["zmp_y"].append(ref["zmp"][1])
        log["base_z"].append(env.data.qpos[2])
        f = res["f"].reshape(-1, 3) if res["f"].size else np.zeros((1, 3))
        log["fz"].append(f[:, 2].sum())
        log["tau_max"].append(np.max(np.abs(res["tau"])))
        log["ok"].append(res["ok"])
        log["support"].append(ref["support"])

    fell = False
    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for i in range(n_steps):
                if not v.is_running():
                    break
                t = i * dt
                ref, res = ctrl.step(plan, t)
                record(t, ref, res)
                v.sync()
                if env.data.qpos[2] < 0.45:
                    fell = True; break
    else:
        for i in range(n_steps):
            t = i * dt
            ref, res = ctrl.step(plan, t)
            record(t, ref, res)
            if env.data.qpos[2] < 0.45:
                fell = True; break

    _report(log, plan, fell)


def _report(log, plan, fell):
    t = np.array(log["t"])
    com_x = np.array(log["com_x"]); com_y = np.array(log["com_y"])
    com_xr = np.array(log["com_x_ref"]); com_yr = np.array(log["com_y_ref"])
    base_z = np.array(log["base_z"]); ok = np.array(log["ok"])
    dist = com_x[-1] - com_x[0]
    track = np.sqrt((com_x - com_xr) ** 2 + (com_y - com_yr) ** 2)

    print("\n===== Stage 4: dynamic walking (flat) =====")
    print(f"planned duration    = {plan.duration:.2f} s, simulated = {t[-1]:.2f} s")
    print(f"fell                = {fell}  (min base z = {base_z.min():.3f} m)")
    print(f"forward distance    = {dist:.3f} m  (CoM x)")
    print(f"CoM track err       = mean {track.mean()*1000:.1f} mm, max {track.max()*1000:.1f} mm")
    print(f"QP feasible         = {ok.mean()*100:.1f}%")
    print(f"max |tau|           = {np.max(log['tau_max']):.1f} N*m")

    completed = (not fell) and (t[-1] > 0.95 * plan.duration)
    walked = dist > 0.5
    ok_all = completed and walked and ok.mean() > 0.95
    print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'} "
          f"(completed={completed}, distance>{0.5}m={walked})")
    _plot(log)
    return ok_all


def _plot(log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    t = np.array(log["t"])
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(t, log["com_x"], "k", label="CoM x")
    ax[0].plot(t, log["com_x_ref"], "r--", label="ref")
    ax[0].set_ylabel("x [m]"); ax[0].legend(); ax[0].grid(True)
    ax[0].set_title("Stage 4: dynamic walking (flat)")
    ax[1].plot(t, log["com_y"], "k", label="CoM y")
    ax[1].plot(t, log["com_y_ref"], "r--", label="ref")
    ax[1].plot(t, log["zmp_y"], "m:", label="ZMP y")
    ax[1].set_ylabel("y [m]"); ax[1].legend(); ax[1].grid(True)
    ax[2].plot(t, log["base_z"]); ax[2].set_ylabel("base z [m]"); ax[2].grid(True)
    ax[3].plot(t, log["fz"]); ax[3].set_ylabel("Σ fz [N]"); ax[3].set_xlabel("t [s]")
    ax[3].grid(True)
    out = Path(__file__).resolve().parents[1] / "logs" / "stage4_walk_flat.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(viewer=args.viewer)
