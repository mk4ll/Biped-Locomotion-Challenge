"""Stage 1 -- gravity compensation test.

Holds the G1 upright & still using ONLY the computed feedforward torques
(no position servo). Verifies that M, h, J_c and the model couple correctly:
if the dynamics terms are right, the robot neither sags under gravity nor is
"glued" to the sim -- it balances by torque.

Run headless (verification):  python scripts/01_gravity_comp.py
With viewer:                   python scripts/01_gravity_comp.py --viewer
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_env_from_params
from src.dynamics.model_terms import ModelTerms
from src.control.gravity_comp import GravityCompensator


def build():
    params = load_params()
    env = make_env_from_params("scene_flat")
    sites = (params["model"]["feet"]["left"]["corners"]
             + params["model"]["feet"]["right"]["corners"])
    terms = ModelTerms(env.model, sites)
    total_mass = mujoco.mj_getTotalmass(env.model)
    gp = params["gravity_comp"]
    q_nom = env.data.qpos[[env.model.jnt_qposadr[env.model.dof_jntid[d]]
                           for d in terms.act_dof]].copy()
    gc = GravityCompensator(terms, total_mass,
                            gravity=params["env"]["gravity"],
                            reg=params["wbc"]["reg"]["force"],
                            q_nom=q_nom if gp.get("hold_posture") else None,
                            hold_kp=gp.get("hold_kp", 0.0),
                            hold_kd=gp.get("hold_kd", 0.0))
    return params, env, terms, gc


def run(viewer=False):
    params, env, terms, gc = build()
    dt = env.dt
    duration = params["gravity_comp"]["duration_s"]
    n_steps = int(duration / dt)

    base_z0 = env.data.qpos[2]
    com0 = env.data.subtree_com[0].copy()

    log = {"t": [], "base_z": [], "base_vel": [], "com_xy": [], "tau_max": [], "resid": []}

    def control_and_step():
        tau, f = gc.compute(env.data)
        log["resid"].append(gc.residual(env.data, tau, f))
        env.step(tau)
        t = len(log["t"]) * dt
        log["t"].append(t)
        log["base_z"].append(env.data.qpos[2])
        log["base_vel"].append(np.linalg.norm(env.data.qvel[:6]))
        log["com_xy"].append(env.data.subtree_com[0][:2].copy())
        log["tau_max"].append(np.max(np.abs(tau)))

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for _ in range(n_steps):
                if not v.is_running():
                    break
                control_and_step()
                v.sync()
    else:
        for _ in range(n_steps):
            control_and_step()

    # -- metrics --
    base_z = np.array(log["base_z"])
    base_vel = np.array(log["base_vel"])
    com_xy = np.array(log["com_xy"])
    resid = np.array(log["resid"])
    z_drift = abs(base_z[-1] - base_z0)
    com_drift = np.linalg.norm(com_xy[-1] - com0[:2])

    print("\n===== Stage 1: gravity compensation =====")
    print(f"duration              = {duration:.1f} s ({n_steps} steps @ {1/dt:.0f} Hz)")
    print(f"initial base height   = {base_z0:.4f} m")
    print(f"final   base height   = {base_z[-1]:.4f} m")
    print(f"base height drift     = {z_drift*1000:.2f} mm")
    print(f"CoM horizontal drift  = {com_drift*1000:.2f} mm")
    print(f"final base |vel|      = {base_vel[-1]:.4e}  (max {base_vel.max():.4e})")
    print(f"max |tau|             = {np.max(log['tau_max']):.1f} N*m")
    print(f"dynamics residual     = mean {resid.mean():.2e}, max {resid.max():.2e}")

    # Pass criteria: stays upright (small drift) and still (small velocity).
    ok = (z_drift < 0.02) and (com_drift < 0.02) and (base_vel.max() < 0.5)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(criteria: z_drift<20mm, com_drift<20mm, base_vel<0.5)")

    _save_plot(log, params)
    return ok


def _save_plot(log, params):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    t = np.array(log["t"])
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, np.array(log["base_z"]) * 1000)
    ax[0].set_ylabel("base height [mm... base_z*1000]"); ax[0].grid(True)
    ax[0].set_title("Stage 1 gravity compensation")
    ax[1].plot(t, log["base_vel"]); ax[1].set_ylabel("base |vel|"); ax[1].grid(True)
    ax[2].plot(t, log["resid"]); ax[2].set_ylabel("dyn residual"); ax[2].set_xlabel("t [s]")
    ax[2].grid(True)
    out = Path(__file__).resolve().parents[1] / "logs" / "stage1_gravity_comp.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    ok = run(viewer=args.viewer)
    sys.exit(0 if ok else 1)
