"""Automated evaluation harness mapped to the project rubric.

Runs a battery of walking scenarios headless and reports the metrics the project
is graded on -- stable continuous walking, single-support stabilization, support
transitions, distance/duration, gait smoothness, robustness to disturbances, and
performance on unseen terrain (inclines, stairs).

    python scripts/evaluate.py            # full battery -> logs/eval_report.md
    python scripts/evaluate.py --quick    # fewer push trials

Outputs a printed table, logs/eval_results.json, and logs/eval_report.md.
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped.robot import Robot, load_g1
from biped import terrain as terrain_mod
from biped.poses import reset_to_crouch
from biped.controllers import WalkingController


def simulate(terrain, step_len=0.15, n_steps=12, reactive=True, mpc=False,
             push=None, mu_scale=1.0, payload=0.0, tail=1.0):
    """Run one walking trial; return a metrics dict.

    push = (force_N, axis[0=x,1=y], time_s) applies a 100 ms shove.
    """
    model, data = load_g1(terrain=terrain)
    if mu_scale != 1.0:
        model.geom_friction[:, 0] *= mu_scale
    robot = Robot(model, data)
    q_nom = reset_to_crouch(robot, terrain)
    if payload:
        tid = model.body("torso_link").id
        model.body_mass[tid] += payload
    ctrl = WalkingController(robot, terrain, q_nom, reactive=reactive, mpc=mpc,
                            n_steps=n_steps, step_len=step_len)
    dt = model.opt.timestep
    n = int((ctrl.plan.total_time + tail) / dt)
    x0 = data.qpos[0]

    pstep = int(push[2] / dt) if push else -1
    pdur = int(0.1 / dt)
    tilt_max = 0.0
    dcm_err, comz_err, taus, vcom_prev, jerk = [], [], [], None, []
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
        vcom = robot.com_vel()
        if vcom_prev is not None:
            jerk.append(np.linalg.norm(vcom - vcom_prev) / dt)
        vcom_prev = vcom
        if data.qpos[2] < 0.4 or tilt > 50:
            fell, t_fall = True, t
            break

    dur = (t_fall if fell else n * dt)
    return {
        "fell": fell,
        "duration": round(dur, 2),
        "distance": round(float(data.qpos[0] - x0), 3),
        "max_tilt": round(tilt_max, 1),
        "dcm_rms": round(float(np.sqrt(np.mean(np.square(dcm_err)))), 4),
        "comz_err_rms": round(float(np.sqrt(np.mean(np.square(comz_err)))), 4),
        "com_accel_rms": round(float(np.sqrt(np.mean(np.square(jerk)))) if jerk else 0, 3),
        "peak_tau": round(float(max(taus)), 1),
    }


def push_success_rate(direction_axis, mags, phases, reactive, mpc=False):
    """Fraction of (magnitude, phase) push trials survived while walking."""
    total, ok = 0, 0
    detail = {}
    for mg in mags:
        survived = 0
        for ph in phases:
            r = simulate(terrain_mod.Flat(), step_len=0.15, n_steps=12,
                         reactive=reactive, mpc=mpc,
                         push=(mg, direction_axis, ph))
            total += 1
            survived += (0 if r["fell"] else 1)
        ok += survived
        detail[mg] = f"{survived}/{len(phases)}"
    return detail, ok, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    phases = [3.0] if args.quick else [2.6, 3.0, 3.4]
    mags = [40, 50, 60] if args.quick else [30, 40, 50, 60, 70]

    print("Running evaluation battery (this takes a few minutes)...\n")
    results = {}

    # --- core walking + terrain ---
    terrains = {
        "flat": terrain_mod.Flat(),
        "incline_4deg": terrain_mod.Incline(angle=np.deg2rad(4)),
        "incline_8deg": terrain_mod.Incline(angle=np.deg2rad(8)),
        "incline_12deg": terrain_mod.Incline(angle=np.deg2rad(12)),
    }
    for name, terr in terrains.items():
        sl = 0.12 if "incline" in name else 0.15
        results[f"walk_{name}"] = simulate(terr, step_len=sl, n_steps=12, reactive=True)

    # --- control comparison: one-step DCM feedback vs preview-MPC ---
    results["ctrl_baseline_flat"] = simulate(terrain_mod.Flat(), step_len=0.15,
                                             n_steps=12, reactive=False, mpc=False)
    results["ctrl_mpc_flat"] = simulate(terrain_mod.Flat(), step_len=0.15,
                                        n_steps=12, reactive=False, mpc=True)
    results["ctrl_mpc_incline_8deg"] = simulate(
        terrain_mod.Incline(angle=np.deg2rad(8)), step_len=0.12,
        n_steps=12, reactive=False, mpc=True)
    # reactive stepping targets forward walking; march/stairs use the base gait
    results["march_in_place"] = simulate(terrain_mod.Flat(), step_len=0.0,
                                         n_steps=8, reactive=False)
    results["stairs_4cm"] = simulate(terrain_mod.Stairs(rise=0.04, run=0.30, x0=0.5),
                                     step_len=0.28, n_steps=14, reactive=False)

    # --- robustness (modeling errors) ---
    results["low_friction_0.6"] = simulate(terrain_mod.Flat(), mu_scale=0.6)
    results["payload_+1.5kg"] = simulate(terrain_mod.Flat(), payload=1.5)

    # --- print table ---
    cols = ["fell", "duration", "distance", "max_tilt", "dcm_rms",
            "comz_err_rms", "com_accel_rms", "peak_tau"]
    hdr = f"{'scenario':22s} " + " ".join(f"{c:>12s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:22s} " + " ".join(f"{str(m[c]):>12s}" for c in cols))

    # --- push robustness ---
    print("\nPush robustness (survived / phases tested):")
    push_res = {}
    # variants: one-step feedback, reactive stepping, preview-MPC
    variants = [("baseline", dict(reactive=False, mpc=False)),
                ("reactive", dict(reactive=True, mpc=False)),
                ("mpc", dict(reactive=False, mpc=True))]
    for axis, label in ((0, "sagittal"), (1, "lateral")):
        for vname, kw in variants:
            detail, ok, total = push_success_rate(axis, mags, phases, **kw)
            key = f"push_{label}_{vname}"
            push_res[key] = {"by_magnitude": detail, "rate": f"{ok}/{total}"}
            print(f"  {key:28s} {detail}  total {ok}/{total}")

    # --- save ---
    out = {"walking": results, "push": push_res}
    with open("logs/eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    write_report(results, push_res, cols)
    print("\nsaved logs/eval_results.json and logs/eval_report.md")


def write_report(results, push_res, cols):
    L = ["# G1 Walking — Evaluation Report", "",
         "Automated battery (`scripts/evaluate.py`). Torque whole-body QP control.",
         "", "## Walking & terrain", "",
         "| scenario | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for name, m in results.items():
        L.append("| " + name + " | " + " | ".join(str(m[c]) for c in cols) + " |")
    L += ["", "## Push robustness (survived / trials)", ""]
    for k, v in push_res.items():
        L.append(f"- **{k}**: {v['rate']}  — by magnitude (N): {v['by_magnitude']}")
    L += ["", "_Metrics: distance/duration before fall; max_tilt (deg); dcm_rms "
          "(DCM tracking, m); comz_err_rms (CoM-height tracking, m); com_accel_rms "
          "(gait smoothness); peak_tau (Nm)._",
          "", "### Notes / known limits",
          "- Reactive (capture-point) stepping targets *forward* walking; the "
          "in-place march and stairs use the base gait.",
          "- Stairs (4 cm risers) climb several steps then tip — full robust "
          "stair-climbing needs tread-center footstep planning.",
          "- Payload tolerance ≈ +1.5 kg clean (marginal beyond +2 kg) with "
          "fixed gains; lateral push tolerance ~40–50 N (no step-timing "
          "adaptation yet), sagittal ≥60 N."]
    with open("logs/eval_report.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
