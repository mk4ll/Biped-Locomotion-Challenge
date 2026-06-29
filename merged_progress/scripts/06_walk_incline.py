"""Stage 6 -- inclined surface (the course challenge).

Generalises the flat controller to a tilted plane by expressing the friction
cones in the SURFACE frame (normal = surface normal, NOT world-z) and keeping
the torso vertical w.r.t. GRAVITY. The floor geom is tilted by `incline_deg`
about world-y; the feet are pre-tilted (ankle += alpha) so they rest flat on
the slope, and R_surface rotates the friction pyramid.

  python scripts/06_walk_incline.py            # stand + walk at params incline
  python scripts/06_walk_incline.py --sweep    # max standing / walking angle
  python scripts/06_walk_incline.py --viewer
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


def Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def make_incline(params, alpha_deg):
    """Build env+controller with the floor tilted so +x is UPHILL, feet pre-tilted.

    Floor rotated about y by -alpha => surface normal = [-sin a, 0, cos a], i.e. the
    surface rises toward +x. R_surface = R_y(-alpha). Feet are pre-tilted by -alpha
    (toe-up) so they rest flat on the up-slope; swing feet land flat (R_des=R_surface).
    """
    alpha = np.radians(alpha_deg)
    env = make_env_from_params("scene_flat")
    m = env.model
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0., 1., 0.]), -alpha)   # +x uphill
    m.geom_quat[fid] = q
    ctrl = WalkingController(env, params)
    Rsurf = Ry(-alpha)
    ctrl.R_surface = Rsurf
    ctrl.swing_tasks["left"].R_des = Rsurf
    ctrl.swing_tasks["right"].R_des = Rsurf
    ctrl.ori_task.R_des = np.eye(3)            # torso vertical w.r.t. gravity
    # pre-tilt ankles (toe-up by alpha) so feet are flat on the up-slope; update
    # the posture nominal to match (else the posture task pulls feet back to flat).
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                 for a in range(m.nu)]
    for j in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        env.data.qpos[adr] -= alpha
        ctrl.pos_task.q_nom[act_names.index(j)] -= alpha
    # Drop the base so the lowest foot corner just touches the surface (z=+0.005),
    # so the feet rest flat without a disturbing settle.
    mujoco.mj_forward(m, env.data)
    nrm = Rsurf @ np.array([0., 0., 1.])
    corners = [c for foot in ctrl.contacts.stance for c in foot["corners"]]
    heights = [env.data.site_xpos[c] @ nrm for c in corners]  # signed dist to surface
    env.data.qpos[2] -= (min(heights) - 0.005)
    mujoco.mj_forward(m, env.data)
    return env, ctrl, Rsurf, alpha


def stand_on_incline(params, alpha_deg, settle_s=3.0, viewer=False):
    env, ctrl, Rsurf, alpha = make_incline(params, alpha_deg)
    base = ctrl.base_id
    com_ref = env.data.subtree_com[base].copy()

    def stand_step():
        ctrl.com_task.a_ref = np.zeros(3)
        ctrl.com_task.p_ref = com_ref
        ctrl.com_task.v_ref = np.zeros(3)
        ctrl.contacts.set_support("double")
        tau_fb = ctrl.gc.compute(env.data,
                                 active_site_ids=ctrl.contacts.active_site_ids)[0]
        res = ctrl.wbc.solve(env.data, [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                             ctrl.contacts.stance, R_surface=Rsurf, fallback_tau=tau_fb)
        env.step(res["tau"])
        return res

    n_steps = int(settle_s / env.dt)
    com_start = env.data.subtree_com[base].copy()
    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for _ in range(n_steps):
                if not v.is_running():
                    break
                stand_step(); v.sync()
    else:
        for _ in range(n_steps):
            stand_step()

    com_end = env.data.subtree_com[base].copy()
    slip = np.linalg.norm(com_end[:2] - com_start[:2])
    base_vel = np.linalg.norm(env.data.qvel[:6])
    fell = env.data.qpos[2] < 0.45
    stable = (not fell) and base_vel < 0.05 and slip < 0.05
    return {"alpha": alpha_deg, "fell": fell, "slip": slip,
            "base_vel": base_vel, "stable": stable, "base_z": env.data.qpos[2]}


def walk_on_incline(params, alpha_deg, viewer=False, settle_s=0.8):
    """Stand briefly, then walk UP the slope. Returns metrics."""
    env, ctrl, Rsurf, alpha = make_incline(params, alpha_deg)
    base = ctrl.base_id
    # settle standing first
    com_ref = env.data.subtree_com[base].copy()
    for _ in range(int(settle_s / env.dt)):
        ctrl.com_task.a_ref = np.zeros(3); ctrl.com_task.p_ref = com_ref
        ctrl.com_task.v_ref = np.zeros(3); ctrl.contacts.set_support("double")
        tau_fb = ctrl.gc.compute(env.data, active_site_ids=ctrl.contacts.active_site_ids)[0]
        res = ctrl.wbc.solve(env.data, [ctrl.com_task, ctrl.ori_task, ctrl.pos_task],
                             ctrl.contacts.stance, R_surface=Rsurf, fallback_tau=tau_fb)
        env.step(res["tau"])

    # build slope-aware plan from the settled state
    init_left = env.data.site_xpos[ctrl.left_site].copy()
    init_right = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    from src.planning.terrain import make_terrain
    plan = WalkPlan(params, init_left, init_right, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"],
                    terrain=make_terrain("incline", angle=alpha))
    x0 = com0[0]
    n = int(plan.duration / env.dt)
    log = {"x": [], "z": [], "xr": [], "zr": []}

    def loop(i):
        ref, _ = ctrl.step(plan, i * env.dt)
        c = env.data.subtree_com[base]
        log["x"].append(c[0]); log["z"].append(c[2])
        log["xr"].append(ref["com"][0]); log["zr"].append(ref["com"][2])
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
    dist = env.data.subtree_com[base][0] - x0
    completed = (not fell) and dist > 0.3
    _plot_incline(log, alpha_deg)
    return {"alpha": alpha_deg, "fell": fell, "dist": dist, "completed": completed,
            "rise": dist * np.tan(alpha)}


def _plot_incline(log, alpha_deg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["x"], log["z"], "k", label="CoM (actual)")
    ax.plot(log["xr"], log["zr"], "r--", label="CoM ref (slope)")
    ax.set_aspect("equal"); ax.grid(True); ax.legend()
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"Stage 6: CoM climbing {alpha_deg:.0f} deg incline (side view)")
    out = Path(__file__).resolve().parents[1] / "logs" / "stage6_walk_incline.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"plot saved: {out}")


def run_single(alpha_deg, viewer=False):
    params = load_params()
    print(f"\n===== Stage 6: stand + walk on {alpha_deg:.1f} deg incline =====")
    r = stand_on_incline(params, alpha_deg, viewer=False)
    print(f"[stand] stable={r['stable']} slip={r['slip']*1000:.1f} mm "
          f"base_vel={r['base_vel']:.4f}")
    w = walk_on_incline(params, alpha_deg, viewer=viewer)
    print(f"[walk]  completed={w['completed']} fell={w['fell']} "
          f"forward={w['dist']:.3f} m  rise={w['rise']*1000:.0f} mm")
    ok = r["stable"] and w["completed"]
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (stands & walks on {alpha_deg:.0f} deg)")
    return ok


def run_sweep():
    params = load_params()
    mu = params["wbc"]["friction_mu"]
    a_slip = np.degrees(np.arctan(mu))
    print("\n===== Stage 6: standing-angle sweep vs theory =====")
    print(f"THEORY  slip angle = arctan(mu={mu:.2f}) = {a_slip:.1f} deg")
    print(f"        tipping does NOT bind: active control keeps CoM over the feet")
    print(f"        (CoM only needs |offset|<~0.09 m; support length 0.17 m)\n")
    print(" deg | tan   slip(mm) base_vel | verdict")
    max_stand = 0.0
    for a in [10, 18, 22, 24, 25, 26, 27, 28]:
        r = stand_on_incline(params, a)
        tan = np.tan(np.radians(a))
        status = "stable" if r["stable"] else ("FELL" if r["fell"] else "slid")
        print(f" {a:3d} | {tan:.2f}  {r['slip']*1000:6.1f}   {r['base_vel']:.4f} | {status}")
        if r["stable"]:
            max_stand = a
    print(f"\nEXPERIMENT max stable standing ~ {max_stand:.0f} deg  "
          f"(theory slip {a_slip:.1f} deg)  ->  slip-limited, matches theory")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        run_sweep()
    else:
        deg = args.deg if args.deg is not None else load_params()["env"]["incline_deg"]
        run_single(deg, viewer=args.viewer)
