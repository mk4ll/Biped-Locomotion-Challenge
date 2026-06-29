"""Automated evaluation battery (merged).

Runs every capability headless, collects rubric-mapped metrics, and writes
logs/eval_report.md. One table = the whole project's verified status.

  python scripts/evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mujoco

from src.utils.config import load_params
from src.sim.mujoco_env import make_env_from_params
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_walk import build_on_terrain, settle


def _metrics(env, ctrl, plan, push=None, push_t=1.8, push_dur=0.1):
    """Run a plan, return metrics dict. push = (fx,fy,fz) world force on pelvis."""
    base = ctrl.base_id
    com0 = env.data.subtree_com[base].copy()
    n = int(plan.duration / env.dt)
    peak_tau = 0.0
    max_tilt = 0.0
    dcm_err = []
    fell = False
    terrain = ctrl.terrain
    for i in range(n):
        t = i * env.dt
        if push is not None and push_t <= t < push_t + push_dur:
            env.apply_external_force(base, np.array(push, float))
        else:
            env.clear_external_forces()
        ref, res = ctrl.step(plan, t)
        peak_tau = max(peak_tau, float(np.max(np.abs(res["tau"]))))
        R = env.data.xmat[base].reshape(3, 3)
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        com = env.data.subtree_com[base]
        dcm_err.append(np.linalg.norm(com[:2] + (ctrl.com_task.jacobian(env.data) @ env.data.qvel)[:2] / ref["omega"] - ref["dcm"]))
        gh = terrain.height(com[0], 0.0) if terrain is not None else 0.0
        if env.data.qpos[2] - gh < 0.45:
            fell = True; break
    env.clear_external_forces()
    com = env.data.subtree_com[base]
    return {"fell": fell, "dist": float(com[0] - com0[0]),
            "rise_mm": float((com[2] - com0[2]) * 1000),
            "max_tilt": float(max_tilt), "peak_tau": float(peak_tau),
            "dcm_rms": float(np.sqrt(np.mean(np.square(dcm_err)))) if dcm_err else 0.0,
            "dur": float((i + 1) * env.dt)}


def _walk_plan(params, terrain_name="flat", angle=8.0, velocity=None, stairs=False):
    if stairs:
        g = params["gait"]; g["step_length"] = 0.16; g["swing_apex"] = 0.06; g["n_steps"] = 18
    env, ctrl, terrain = build_on_terrain(params, terrain_name, angle,
                                          dict(rise=0.025, run=0.16, n_steps=6, x0=0.30))
    settle(env, ctrl, terrain, 0.8)
    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[ctrl.base_id].copy()
    plan = WalkPlan(params, il, ir, com0, com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"], terrain=terrain, velocity=velocity)
    return env, ctrl, plan


def main():
    rows = []
    scenarios = [
        ("walk_flat",        dict()),
        ("walk_incline_8",   dict(terrain_name="incline", angle=8.0)),
        ("walk_incline_12",  dict(terrain_name="incline", angle=12.0)),
        ("walk_incline_16",  dict(terrain_name="incline", angle=16.0)),
        ("stairs_6x2.5cm",   dict(terrain_name="stairs", stairs=True)),
        ("omni_forward",     dict(velocity=(0.12, 0.0, 0.0))),
        ("omni_strafe",      dict(velocity=(0.0, 0.08, 0.0))),
        ("omni_curve",       dict(velocity=(0.10, 0.0, 0.12))),
    ]
    for name, kw in scenarios:
        params = load_params()
        env, ctrl, plan = _walk_plan(params, **kw)
        m = _metrics(env, ctrl, plan)
        m["name"] = name
        rows.append(m)
        print(f"{name:18s} fell={m['fell']!s:5s} dist={m['dist']:+.2f} "
              f"rise={m['rise_mm']:+.0f}mm tilt={m['max_tilt']:.1f} tau={m['peak_tau']:.0f}")

    # push recovery (walk + lateral / sagittal shoves)
    push_rows = []
    for label, F in [("push_lateral_50N", (0, 50, 0)), ("push_sagittal_100N", (100, 0, 0))]:
        params = load_params()
        env, ctrl, plan = _walk_plan(params)
        m = _metrics(env, ctrl, plan, push=F)
        m["name"] = label
        push_rows.append(m)
        print(f"{label:18s} fell={m['fell']!s:5s} dist={m['dist']:+.2f} tilt={m['max_tilt']:.1f}")

    _write_report(rows, push_rows)


def _write_report(rows, push_rows):
    out = Path(__file__).resolve().parents[1] / "logs" / "eval_report.md"
    out.parent.mkdir(exist_ok=True)
    L = ["# Merged G1 — Evaluation Report",
         "",
         "Automated battery (`scripts/evaluate.py`). Torque whole-body QP, DCM planning.",
         "",
         "## Walking & terrain",
         "",
         "| scenario | fell | dist [m] | rise [mm] | max_tilt [deg] | dcm_rms [m] | peak_tau [Nm] |",
         "|---|---|---|---|---|---|---|"]
    for m in rows:
        L.append(f"| {m['name']} | {m['fell']} | {m['dist']:+.2f} | {m['rise_mm']:+.0f} | "
                 f"{m['max_tilt']:.1f} | {m['dcm_rms']:.3f} | {m['peak_tau']:.0f} |")
    L += ["", "## Push robustness (mid-walk shove, 100 ms)", "",
          "| scenario | survived | dist [m] | max_tilt [deg] |", "|---|---|---|---|"]
    for m in push_rows:
        L.append(f"| {m['name']} | {not m['fell']} | {m['dist']:+.2f} | {m['max_tilt']:.1f} |")
    L += ["", "_Metrics: dist/rise of CoM; max_tilt = pelvis tilt from vertical;",
          "dcm_rms = DCM tracking error; peak_tau = max joint torque._"]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport written: {out}")


if __name__ == "__main__":
    main()
