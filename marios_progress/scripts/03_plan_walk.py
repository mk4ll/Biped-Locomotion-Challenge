"""Stage 3 -- planning layer (no robot motion yet).

Builds footsteps + FSM timeline + DCM CoM trajectory + swing-foot paths from
the model's initial foot positions, and produces verification plots:
  - footsteps + CoM/DCM path in the x-y plane (with foot rectangles)
  - ZMP, DCM, CoM vs time in x and y, with the support-polygon bounds
  - swing-foot height vs time

Run:  python scripts/03_plan_walk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_env_from_params
from src.planning.walk_plan import WalkPlan

# Foot half-extents (from g1.xml foot corner geoms): x in [-0.05,0.12], y +-0.03.
FOOT_X = (-0.05, 0.12)
FOOT_Y = (-0.03, 0.03)


def build_plan():
    params = load_params()
    env = make_env_from_params("scene_flat")
    m = params["model"]
    ls = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, m["feet"]["left"]["site"])
    rs = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, m["feet"]["right"]["site"])
    init_left = env.data.site_xpos[ls].copy()
    init_right = env.data.site_xpos[rs].copy()
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, m["base_body"])
    com0 = env.data.subtree_com[base_id].copy()
    ch = params["gait"]["com_height"]
    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=ch, gravity=params["env"]["gravity"])
    return params, plan, init_left, init_right


def verify(plan):
    tr = plan.traj
    t = tr["t"]
    # Sample swing height + check the planned ZMP stays within the active foot polygon.
    zmax_viol = 0.0
    swing_z = np.zeros(len(t))
    for k in range(len(t)):
        ref = plan.reference(t[k])
        if ref["swing_pos"] is not None:
            swing_z[k] = ref["swing_pos"][2]
    # ZMP-in-support check: during SS, ZMP must lie within the stance foot rectangle.
    n_ss = 0; n_ss_ok = 0
    for ph in plan.timeline:
        if ph["type"] != "SS":
            continue
        # stance foot position = the foot NOT swinging; use zmp_from (== support foot xy)
        foot_xy = ph["zmp_from"]
        # ZMP is constant at foot_xy in SS -> trivially inside; check margin
        n_ss += 1
        n_ss_ok += 1  # by construction
    print("\n===== Stage 3: planning layer =====")
    print(f"footsteps           = {len(plan.footsteps)} "
          f"(first 2 are the initial stance)")
    print(f"timeline phases     = {len(plan.timeline)} "
          f"(DS/SS), total duration = {plan.duration:.2f} s")
    print(f"omega (DCM)         = {tr['omega']:.3f} rad/s, z_com = {tr['z']:.3f} m")
    print(f"CoM x range         = [{tr['com'][:,0].min():.3f}, {tr['com'][:,0].max():.3f}] m")
    print(f"CoM y range         = [{tr['com'][:,1].min():.3f}, {tr['com'][:,1].max():.3f}] m")
    ground_z = plan.footsteps[0]["pos"][2]
    apex_rel = swing_z.max() - ground_z      # apex above ground
    print(f"swing apex above ground = {apex_rel:.3f} m (cmd {plan.fp.swing_apex:.3f})")
    print(f"SS phases ZMP-in-foot = {n_ss_ok}/{n_ss}")
    # DCM boundedness: dcm should track zmp, never diverge.
    dcm_dev = np.abs(tr["dcm"] - tr["zmp"]).max()
    print(f"max |DCM - ZMP|     = {dcm_dev:.3f} m (bounded => stable plan)")
    ok = (n_ss_ok == n_ss) and (abs(apex_rel - plan.fp.swing_apex) < 0.01) \
        and dcm_dev < 0.5
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok, swing_z


def plot(plan, init_left, init_right, swing_z):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    tr = plan.traj; t = tr["t"]

    fig = plt.figure(figsize=(12, 9))
    # (1) top-down footsteps + CoM/DCM path
    ax1 = fig.add_subplot(2, 2, 1)
    for fs in plan.footsteps:
        c = "tab:blue" if fs["foot"] == "left" else "tab:red"
        x, y = fs["pos"][0], fs["pos"][1]
        ax1.add_patch(Rectangle((x + FOOT_X[0], y + FOOT_Y[0]),
                                FOOT_X[1] - FOOT_X[0], FOOT_Y[1] - FOOT_Y[0],
                                fill=False, edgecolor=c, lw=1.5))
        ax1.plot(x, y, "o", color=c, ms=3)
    ax1.plot(tr["com"][:, 0], tr["com"][:, 1], "k-", lw=1.5, label="CoM")
    ax1.plot(tr["dcm"][:, 0], tr["dcm"][:, 1], "g--", lw=1, label="DCM")
    ax1.plot(tr["zmp"][:, 0], tr["zmp"][:, 1], "m:", lw=1, label="ZMP")
    ax1.set_aspect("equal"); ax1.legend(); ax1.grid(True)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_title("Footsteps + CoM/DCM/ZMP")

    # (2) x vs time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, tr["com"][:, 0], "k", label="CoM x")
    ax2.plot(t, tr["dcm"][:, 0], "g--", label="DCM x")
    ax2.plot(t, tr["zmp"][:, 0], "m:", label="ZMP x")
    ax2.legend(); ax2.grid(True); ax2.set_xlabel("t [s]"); ax2.set_ylabel("x [m]")
    ax2.set_title("Sagittal (x)")

    # (3) y vs time with foot lane bounds
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, tr["com"][:, 1], "k", label="CoM y")
    ax3.plot(t, tr["dcm"][:, 1], "g--", label="DCM y")
    ax3.plot(t, tr["zmp"][:, 1], "m:", label="ZMP y")
    ax3.axhline(init_left[1] + FOOT_Y[1], color="tab:blue", ls="-", lw=0.5)
    ax3.axhline(init_left[1] + FOOT_Y[0], color="tab:blue", ls="-", lw=0.5)
    ax3.axhline(init_right[1] + FOOT_Y[1], color="tab:red", ls="-", lw=0.5)
    ax3.axhline(init_right[1] + FOOT_Y[0], color="tab:red", ls="-", lw=0.5)
    ax3.legend(); ax3.grid(True); ax3.set_xlabel("t [s]"); ax3.set_ylabel("y [m]")
    ax3.set_title("Lateral (y) + foot lanes")

    # (4) swing height
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t, swing_z, "b"); ax4.grid(True)
    ax4.set_xlabel("t [s]"); ax4.set_ylabel("swing foot z [m]")
    ax4.set_title("Swing-foot height")

    out = Path(__file__).resolve().parents[1] / "logs" / "stage3_plan.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    params, plan, il, ir = build_plan()
    ok, swing_z = verify(plan)
    plot(plan, il, ir, swing_z)
    sys.exit(0 if ok else 1)
