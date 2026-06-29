"""Stage 2 -- WBC QP standing balance.

Three phases, all driven by the SAME WBC QP (torques only):
  (A) settle / stand        - hold nominal CoM + upright torso
  (B) weight shift          - CoM target oscillates laterally (left/right)
  (C) single support        - shift CoM over left foot, lift the right foot

Run headless (metrics):  python scripts/02_stand_balance.py
With viewer:             python scripts/02_stand_balance.py --viewer
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
from src.dynamics.contacts import ContactSet
from src.control.wbc_qp import WBCQP
from src.control.tasks import CoMTask, OrientationTask, PostureTask, FootTask
from src.control.gravity_comp import GravityCompensator


def build():
    params = load_params()
    env = make_env_from_params("scene_flat")
    m = params["model"]
    lc = m["feet"]["left"]["corners"]
    rc = m["feet"]["right"]["corners"]
    terms = ModelTerms(env.model, lc + rc)
    contacts = ContactSet(terms, lc, rc,
                          m["feet"]["left"]["site"], m["feet"]["right"]["site"])

    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, m["base_body"])
    left_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, m["feet"]["left"]["site"])
    right_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, m["feet"]["right"]["site"])

    tp = params["wbc"]["tasks"]
    com_task = CoMTask(terms, base_id, tp["com"]["kp"], tp["com"]["kd"], tp["com"]["weight"])
    ori_task = OrientationTask(terms, base_id, tp["orientation"]["kp"],
                               tp["orientation"]["kd"], tp["orientation"]["weight"])
    q_nom = env.data.qpos[[terms.model.jnt_qposadr[terms.model.dof_jntid[d]]
                           for d in terms.act_dof]].copy()
    pos_task = PostureTask(terms, q_nom, tp["posture"]["kp"], tp["posture"]["kd"],
                           tp["posture"]["weight"])
    swing_task = FootTask(terms, right_site, tp["swing_foot"]["kp"],
                          tp["swing_foot"]["kd"], tp["swing_foot"]["weight"])

    wbc = WBCQP(terms, params)
    gc = GravityCompensator(terms, mujoco.mj_getTotalmass(env.model),
                            params["env"]["gravity"], params["wbc"]["reg"]["force"])

    # Nominal references (captured at the keyframe).
    R_nom = env.data.xmat[base_id].reshape(3, 3).copy()
    ori_task.R_des = R_nom
    com0 = env.data.subtree_com[base_id].copy()
    com_task.p_ref = com0.copy()

    ctx = dict(params=params, env=env, terms=terms, contacts=contacts,
               com_task=com_task, ori_task=ori_task, pos_task=pos_task,
               swing_task=swing_task, wbc=wbc, gc=gc, com0=com0,
               left_site=left_site, right_site=right_site, base_id=base_id)
    return ctx


def run(viewer=False):
    ctx = build()
    env = ctx["env"]; params = ctx["params"]; dt = env.dt
    sb = params["stand_balance"]
    com0 = ctx["com0"]

    t_settle = sb["settle_s"]
    t_ws = sb["weight_shift_period_s"] * 2          # two periods
    t_ss = sb["ss_shift_s"] + sb["ss_lift_s"] + sb["hold_s"]
    T_total = t_settle + t_ws + t_ss
    n_steps = int(T_total / dt)
    left_xy0 = [None]

    log = {"t": [], "base_z": [], "com_y": [], "com_y_ref": [], "right_foot_z": [],
           "ok": [], "fz_total": []}

    right_lifted = [None]

    def references(t):
        tasks = [ctx["com_task"], ctx["ori_task"], ctx["pos_task"]]
        if t < t_settle:
            ctx["com_task"].p_ref = com0.copy()
            ctx["contacts"].set_support("double")
        elif t < t_settle + t_ws:
            # (B) lateral weight shift
            tau_t = t - t_settle
            y = sb["weight_shift_amp"] * np.sin(2 * np.pi * tau_t / sb["weight_shift_period_s"])
            ref = com0.copy(); ref[1] = com0[1] + y
            ctx["com_task"].p_ref = ref
            ctx["contacts"].set_support("double")
        else:
            # (C) single support on LEFT foot, lift RIGHT foot.
            if left_xy0[0] is None:
                left_xy0[0] = env.data.site_xpos[ctx["left_site"]][:2].copy()
            left_xy = left_xy0[0]
            ss_t = t - t_settle - t_ws
            if ss_t < sb["ss_shift_s"]:
                # ramp CoM from nominal to over the stance foot (still double support)
                a = ss_t / sb["ss_shift_s"]
                ref = com0.copy()
                ref[0] = com0[0] + a * (left_xy[0] - com0[0])
                ref[1] = com0[1] + a * (left_xy[1] - com0[1])
                ctx["com_task"].p_ref = ref
                ctx["contacts"].set_support("double")
            else:
                # over the stance foot: single support + raise swing foot
                ctx["com_task"].p_ref = np.array([left_xy[0], left_xy[1], com0[2]])
                ctx["contacts"].set_support("left")
                if right_lifted[0] is None:
                    right_lifted[0] = env.data.site_xpos[ctx["right_site"]].copy()
                lift_t = min(1.0, (ss_t - sb["ss_shift_s"]) / sb["ss_lift_s"])
                ctx["swing_task"].p_ref = (right_lifted[0]
                                           + np.array([0, 0, lift_t * sb["single_support_lift"]]))
                tasks = tasks + [ctx["swing_task"]]
        return tasks

    def control_and_step():
        t = len(log["t"]) * dt
        tasks = references(t)
        stance = ctx["contacts"].stance
        tau_fallback = ctx["gc"].compute(env.data, active_site_ids=ctx["contacts"].active_site_ids)[0]
        res = ctx["wbc"].solve(env.data, tasks, stance, fallback_tau=tau_fallback)
        env.step(res["tau"])
        log["t"].append(t)
        log["base_z"].append(env.data.qpos[2])
        log["com_y"].append(env.data.subtree_com[ctx["base_id"]][1])
        log["com_y_ref"].append(ctx["com_task"].p_ref[1])
        log["right_foot_z"].append(env.data.site_xpos[ctx["right_site"]][2])
        log["ok"].append(res["ok"])
        f = res["f"].reshape(-1, 3) if res["f"].size else np.zeros((1, 3))
        log["fz_total"].append(f[:, 2].sum())

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

    _report(log, ctx, sb, t_settle, t_ws)


def _report(log, ctx, sb, t_settle, t_ws):
    t = np.array(log["t"])
    base_z = np.array(log["base_z"])
    com_y = np.array(log["com_y"])
    com_y_ref = np.array(log["com_y_ref"])
    rf_z = np.array(log["right_foot_z"])
    ok = np.array(log["ok"])

    # Weight-shift tracking error (phase B).
    mB = (t >= t_settle) & (t < t_settle + t_ws)
    ws_err = np.abs(com_y[mB] - com_y_ref[mB]).mean() if mB.any() else float("nan")
    ws_range = com_y[mB].max() - com_y[mB].min() if mB.any() else 0.0
    # Single support (phase C end).
    mC = t >= t_settle + t_ws
    rf_lift = rf_z[mC].max() - rf_z[0]
    min_base_z = base_z.min()

    print("\n===== Stage 2: WBC standing balance =====")
    print(f"steps                 = {len(t)}  ({1/ctx['env'].dt:.0f} Hz, {t[-1]:.1f} s)")
    print(f"QP feasible fraction  = {ok.mean()*100:.1f}%  (infeasible {ctx['wbc'].n_infeasible})")
    print(f"min base height       = {min_base_z:.3f} m  (start {base_z[0]:.3f})")
    print(f"[B] weight-shift: mean track err = {ws_err*1000:.1f} mm, "
          f"CoM_y range = {ws_range*1000:.1f} mm (cmd ±{sb['weight_shift_amp']*1000:.0f})")
    print(f"[C] single support: right-foot lift = {rf_lift*1000:.1f} mm "
          f"(cmd {sb['single_support_lift']*1000:.0f})")

    upright = min_base_z > 0.70
    ws_ok = ws_err < 0.02 and ws_range > 0.5 * (2 * sb["weight_shift_amp"])
    ss_ok = rf_lift > 0.5 * sb["single_support_lift"]
    ok_all = upright and ws_ok and ss_ok and ok.mean() > 0.95
    print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}  "
          f"(upright={upright}, weight_shift={ws_ok}, single_support={ss_ok})")

    _plot(log, ctx)
    return ok_all


def _plot(log, ctx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    t = np.array(log["t"])
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(t, np.array(log["base_z"])); ax[0].set_ylabel("base z [m]"); ax[0].grid(True)
    ax[0].set_title("Stage 2: WBC standing balance")
    ax[1].plot(t, log["com_y"], label="CoM y")
    ax[1].plot(t, log["com_y_ref"], "--", label="ref"); ax[1].legend(); ax[1].grid(True)
    ax[1].set_ylabel("CoM y [m]")
    ax[2].plot(t, log["right_foot_z"]); ax[2].set_ylabel("right foot z [m]"); ax[2].grid(True)
    ax[3].plot(t, log["fz_total"]); ax[3].set_ylabel("Σ fz [N]"); ax[3].set_xlabel("t [s]")
    ax[3].grid(True)
    out = Path(__file__).resolve().parents[1] / "logs" / "stage2_stand_balance.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    run(viewer=args.viewer)
