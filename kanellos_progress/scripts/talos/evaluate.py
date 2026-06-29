"""Automated evaluation battery for the PAL Talos (rubric-mapped metrics).

Mirrors scripts/evaluate.py but for Talos.  Runs walking scenarios headless and
reports the graded metrics: continuous walking, terrain (inclines), control
comparison (one-step DCM feedback vs preview-MPC), and push robustness.

    python scripts/talos/evaluate.py            # -> logs/talos_eval_report.md
    python scripts/talos/evaluate.py --quick
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biped.robot import Robot, load_talos, TALOS_CONFIG
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import WalkingController


def simulate(terrain, step_len=0.15, n_steps=10, reactive=False, mpc=False,
             push=None, tail=1.0):
    model, data = load_talos(terrain=terrain)
    robot = Robot(model, data, TALOS_CONFIG)
    q_nom = reset_to_crouch(robot, terrain)
    ctrl = WalkingController(robot, terrain, q_nom, reactive=reactive, mpc=mpc,
                            n_steps=n_steps, step_len=step_len)
    dt = model.opt.timestep
    n = int((ctrl.plan.total_time + tail) / dt)
    x0 = data.qpos[0]
    pstep = int(push[2] / dt) if push else -1
    pdur = int(0.1 / dt)
    tilt_max = 0.0
    dcm_err, comz_err, taus, vprev, jerk = [], [], [], None, []
    fell, t_fall = False, None
    for i in range(n):
        t = i * dt
        tau, sol = ctrl(t)
        data.ctrl[:] = tau
        data.xfrc_applied[:] = 0.0
        if push and pstep <= i < pstep + pdur:
            data.xfrc_applied[robot.pelvis_id, push[1]] = push[0]
        mujoco.mj_step(model, data)
        z = data.xmat[robot.pelvis_id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(z[2], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        sp = sol["plan"]
        dcm_err.append(np.linalg.norm(sol["xi"] - sp["xi"]))
        comz_err.append(abs(robot.com()[2] - ctrl.z_des))
        taus.append(np.abs(tau).max())
        v = robot.com_vel()
        if vprev is not None:
            jerk.append(np.linalg.norm(v - vprev) / dt)
        vprev = v
        if data.qpos[2] < 0.6 or tilt > 40:
            fell, t_fall = True, t
            break
    dur = t_fall if fell else n * dt
    return {
        "fell": fell, "duration": round(dur, 2),
        "distance": round(float(data.qpos[0] - x0), 3),
        "max_tilt": round(tilt_max, 1),
        "dcm_rms": round(float(np.sqrt(np.mean(np.square(dcm_err)))), 4),
        "comz_err_rms": round(float(np.sqrt(np.mean(np.square(comz_err)))), 4),
        "com_accel_rms": round(float(np.sqrt(np.mean(np.square(jerk)))) if jerk else 0, 3),
        "peak_tau": round(float(max(taus)), 1),
    }


def push_rate(axis, mags, phases, mpc):
    total, ok, detail = 0, 0, {}
    for mg in mags:
        s = 0
        for ph in phases:
            r = simulate(terrain_mod.Flat(), push=(mg, axis, ph), mpc=mpc)
            total += 1
            s += 0 if r["fell"] else 1
        ok += s
        detail[mg] = f"{s}/{len(phases)}"
    return detail, ok, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    phases = [3.0] if args.quick else [2.6, 3.0, 3.4]
    mags = [60, 90] if args.quick else [60, 90, 120, 150]

    print("Running Talos evaluation battery...\n")
    results = {}
    terrains = {
        "flat": terrain_mod.Flat(),
        "incline_4deg": terrain_mod.Incline(angle=np.deg2rad(4)),
        "incline_8deg": terrain_mod.Incline(angle=np.deg2rad(8)),
    }
    for name, terr in terrains.items():
        sl = 0.12 if "incline" in name else 0.15
        results[f"walk_{name}"] = simulate(terr, step_len=sl, n_steps=10)
    results["ctrl_baseline_flat"] = simulate(terrain_mod.Flat(), mpc=False)
    results["ctrl_mpc_flat"] = simulate(terrain_mod.Flat(), mpc=True)

    cols = ["fell", "duration", "distance", "max_tilt", "dcm_rms",
            "comz_err_rms", "com_accel_rms", "peak_tau"]
    hdr = f"{'scenario':22s} " + " ".join(f"{c:>12s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:22s} " + " ".join(f"{str(m[c]):>12s}" for c in cols))

    print("\nPush robustness (survived / phases):")
    push_res = {}
    for axis, label in ((0, "sagittal"), (1, "lateral")):
        for mpc in (False, True):
            detail, ok, total = push_rate(axis, mags, phases, mpc)
            key = f"push_{label}_{'mpc' if mpc else 'baseline'}"
            push_res[key] = {"by_magnitude": detail, "rate": f"{ok}/{total}"}
            print(f"  {key:24s} {detail}  total {ok}/{total}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/talos_eval_results.json", "w") as f:
        json.dump({"walking": results, "push": push_res}, f, indent=2)
    L = ["# Talos Walking — Evaluation Report", "",
         "Automated battery (`scripts/talos/evaluate.py`). 94 kg, box feet, "
         "32 torque motors. Same DCM + whole-body-QP stack as the G1.", "",
         "## Walking & terrain", "",
         "| scenario | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for name, m in results.items():
        L.append("| " + name + " | " + " | ".join(str(m[c]) for c in cols) + " |")
    L += ["", "## Push robustness (survived / trials)", ""]
    for k, v in push_res.items():
        L.append(f"- **{k}**: {v['rate']} — by magnitude (N): {v['by_magnitude']}")
    with open("logs/talos_eval_report.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nsaved logs/talos_eval_results.json and logs/talos_eval_report.md")


if __name__ == "__main__":
    main()
