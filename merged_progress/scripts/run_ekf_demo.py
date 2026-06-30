"""EKF Demo — three experiments, six diagnostic panels.

Runs the G1 with the floating-base EKF alongside ground truth and compares
three estimators:
  • Ground truth   — raw MuJoCo qpos/qvel (cheat state)
  • EKF+contact    — predict with noisy IMU + update at foot touchdowns
  • Dead reckoning — predict only (noisy IMU, no contact corrections)

Three scenarios:
  Exp A — flat walk: base accuracy + σ uncertainty
  Exp B — push recovery: 60 N lateral push x2
  Exp C — long drift: 20-step walk to show cumulative dead-reckoning drift

Output: logs/ekf_demo.png

  python scripts/run_ekf_demo.py
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.config import load_params
from src.sim.mujoco_env import make_robot_env
from src.control.walking_controller import WalkingController
from src.planning.walk_plan import WalkPlan
from src.planning.terrain import make_terrain
from src.estimation.base_ekf import BaseEKF, SIGMA_ACCEL, SIGMA_GYRO
from run_walk import build_on_terrain, settle

# ── noise seed (fixed for reproducibility) ─────────────────────────────────────
RNG = np.random.default_rng(42)

# ── style ───────────────────────────────────────────────────────────────────────
BG     = "#14171a"
SURF   = "#1c2026"
BORDER = "#2e3540"
TEXT   = "#d8d0c4"
MUTED  = "#6b7280"
ACCENT = "#c8861e"
GREEN  = "#4ade80"
RED    = "#f87171"
BLUE   = "#60a5fa"
PURPLE = "#a78bfa"
CYAN   = "#22d3ee"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURF,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "grid.alpha": 0.5, "lines.linewidth": 1.6,
    "font.family": "monospace", "font.size": 9,
    "legend.facecolor": SURF, "legend.edgecolor": BORDER,
})


# ── helpers ─────────────────────────────────────────────────────────────────────

def _sensor(m, d, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = m.sensor_adr[sid]
    dim = m.sensor_dim[sid]
    return d.sensordata[adr:adr + dim].copy()


def _base_truth(env, base_id):
    pos = env.data.xpos[base_id].copy()
    R   = env.data.xmat[base_id].reshape(3, 3)
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll  = np.arctan2(R[2, 1], R[2, 2])
    rpy   = np.array([roll, pitch, yaw])
    vel   = env.data.cvel[base_id][3:6].copy()
    return pos, vel, rpy


def _imu_noisy(m, d):
    a = _sensor(m, d, "imu-pelvis-linear-acceleration")
    w = _sensor(m, d, "imu-pelvis-angular-velocity")
    return (a + RNG.standard_normal(3) * SIGMA_ACCEL,
            w + RNG.standard_normal(3) * SIGMA_GYRO)


def _feet_on_ground(env, ctrl):
    """Return [(foot_world, foot_local)] for feet with z < 4 cm (on ground)."""
    base_id = ctrl.base_id
    R   = env.data.xmat[base_id].reshape(3, 3)
    pos = env.data.xpos[base_id]
    out = []
    for side in ("left", "right"):
        fw = env.data.site_xpos[ctrl.foot_sites[side]].copy()
        if fw[2] < 0.04:
            out.append((fw, R.T @ (fw - pos)))
    return out


def run_experiment(n_steps=10, push_steps=None, noise_scale=1.0, seed=0):
    """Run one experiment; return dict of logged arrays."""
    global RNG
    RNG = np.random.default_rng(seed)

    params = load_params()
    params["gait"]["n_steps"] = n_steps
    env, ctrl, terrain_obj = build_on_terrain(params, "flat")
    settle(env, ctrl, terrain_obj, 0.8)

    m, d = env.model, env.data
    base = ctrl.base_id

    il = env.data.site_xpos[ctrl.left_site].copy()
    ir = env.data.site_xpos[ctrl.right_site].copy()
    com0 = env.data.subtree_com[base].copy()
    plan = WalkPlan(params, il, ir, com0,
                    com_height=params["gait"]["com_height"],
                    gravity=params["env"]["gravity"],
                    terrain=terrain_obj)

    p0, v0, rpy0 = _base_truth(env, base)
    ekf_full = BaseEKF(dt=env.dt)
    ekf_full.reset(p0.copy(), v0.copy(), rpy0.copy())
    ekf_dr = BaseEKF(dt=env.dt)
    ekf_dr.reset(p0.copy(), v0.copy(), rpy0.copy())

    N = int(plan.duration / env.dt)
    ts      = np.zeros(N)
    gt_pos  = np.zeros((N, 3))
    ekf_pos = np.zeros((N, 3))
    dr_pos  = np.zeros((N, 3))
    gt_vel  = np.zeros((N, 3))
    ekf_vel = np.zeros((N, 3))
    sigma   = np.zeros((N, 3))
    contacts = []   # (t, side)
    push_times = []

    push_steps_set = set(push_steps or [])
    prev_phase = None
    prev_swing = None
    fell = False

    for i in range(N):
        t = i * env.dt

        # lateral push
        if i in push_steps_set:
            mujoco.mj_applyFT(m, d,
                              np.array([0.0, 60.0, 0.0]),
                              np.zeros(3),
                              d.xpos[base].copy(),
                              base,
                              d.qfrc_applied)
            push_times.append(t)

        ctrl.step(plan, t)
        if env.data.qpos[2] < 0.40:
            fell = True; break

        a_imu, w_imu = _imu_noisy(m, d)
        # extra noise for dead-reckoning (simulates worse IMU or longer drift)
        a_dr = a_imu + RNG.standard_normal(3) * SIGMA_ACCEL * (noise_scale - 1)
        w_dr = w_imu + RNG.standard_normal(3) * SIGMA_GYRO  * (noise_scale - 1)

        ekf_full.predict(a_imu, w_imu)
        ekf_dr.predict(a_dr, w_dr)

        # touchdown detection
        ref = plan.reference(t)
        curr_phase = ref.get("phase", "DS")
        if prev_phase == "SS" and curr_phase == "DS" and prev_swing is not None:
            contacts.append((t, prev_swing))
        if curr_phase == "SS":
            prev_swing = ref.get("swing")
        prev_phase = curr_phase

        # contact updates for full EKF only
        for fw, fl in _feet_on_ground(env, ctrl):
            ekf_full.update_contact(fw, fl)

        pg, vg, _ = _base_truth(env, base)
        ts[i]      = t
        gt_pos[i]  = pg
        gt_vel[i]  = vg
        ekf_pos[i] = ekf_full.pos
        dr_pos[i]  = ekf_dr.pos
        ekf_vel[i] = ekf_full.vel
        P_diag     = np.diag(ekf_full.covariance)
        sigma[i]   = np.sqrt(np.maximum(P_diag[:3], 0))

        d.qfrc_applied[:] = 0.0

    valid = i + 1
    return dict(
        ts=ts[:valid], gt_pos=gt_pos[:valid],
        ekf_pos=ekf_pos[:valid], dr_pos=dr_pos[:valid],
        gt_vel=gt_vel[:valid], ekf_vel=ekf_vel[:valid],
        sigma=sigma[:valid], contacts=contacts,
        push_times=push_times, fell=fell,
    )


# ── plotting ─────────────────────────────────────────────────────────────────────

def plot_all(A, B, C, out_path):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "G1 Floating-Base EKF — IMU Prediction + Contact Correction vs Dead Reckoning",
        fontsize=12, color=TEXT, fontweight="bold", y=0.99,
    )
    gs = fig.add_gridspec(3, 3, hspace=0.46, wspace=0.34,
                          left=0.06, right=0.97, top=0.95, bottom=0.06)

    ax = {
        "Ax": fig.add_subplot(gs[0, 0]),
        "Ay": fig.add_subplot(gs[1, 0]),
        "As": fig.add_subplot(gs[2, 0]),
        "Bx": fig.add_subplot(gs[0, 1]),
        "By": fig.add_subplot(gs[1, 1]),
        "Be": fig.add_subplot(gs[2, 1]),
        "C2": fig.add_subplot(gs[0, 2]),
        "Cv": fig.add_subplot(gs[1, 2]),
        "Cd": fig.add_subplot(gs[2, 2]),
    }

    legend_lines = [
        mpatches.Patch(color=BLUE,   label="Ground truth"),
        mpatches.Patch(color=GREEN,  label="EKF + contact"),
        mpatches.Patch(color=RED,    label="Dead reckoning"),
        mpatches.Patch(color=ACCENT, alpha=0.5, label="Push event"),
        mpatches.Patch(color=GREEN,  alpha=0.3, label="Touchdown"),
    ]

    def shade_contacts(a, contacts):
        for ct, _ in contacts:
            a.axvline(ct, color=GREEN, alpha=0.25, lw=0.8)

    def shade_pushes(a, push_times):
        for pt in push_times:
            a.axvspan(pt - 0.02, pt + 0.20, color=ACCENT, alpha=0.18, zorder=0)

    def style(a, title, xl="time [s]", yl=""):
        a.set_title(title, color=ACCENT, fontsize=9)
        a.set_xlabel(xl, fontsize=8)
        a.set_ylabel(yl, fontsize=8)
        a.grid(True)

    # ── EXP A ── flat walk ──────────────────────────────────────────────────────
    ax["Ax"].plot(A["ts"], A["gt_pos"][:, 0],  color=BLUE,  lw=2.0)
    ax["Ax"].plot(A["ts"], A["ekf_pos"][:, 0], color=GREEN, lw=1.4, ls="--")
    ax["Ax"].plot(A["ts"], A["dr_pos"][:, 0],  color=RED,   lw=1.0, ls=":")
    shade_contacts(ax["Ax"], A["contacts"])
    style(ax["Ax"], "Exp A — Flat walk: forward position (X)", yl="x [m]")

    ax["Ay"].plot(A["ts"], A["gt_pos"][:, 1],  color=BLUE,  lw=2.0)
    ax["Ay"].plot(A["ts"], A["ekf_pos"][:, 1], color=GREEN, lw=1.4, ls="--")
    ax["Ay"].plot(A["ts"], A["dr_pos"][:, 1],  color=RED,   lw=1.0, ls=":")
    shade_contacts(ax["Ay"], A["contacts"])
    style(ax["Ay"], "Exp A — Flat walk: lateral position (Y)", yl="y [m]")

    ax["As"].fill_between(A["ts"], 0, A["sigma"][:, 0] * 100,
                          color=GREEN, alpha=0.55, label="σ_x [cm]")
    ax["As"].fill_between(A["ts"], 0, A["sigma"][:, 1] * 100,
                          color=PURPLE, alpha=0.45, label="σ_y [cm]")
    ax["As"].fill_between(A["ts"], 0, A["sigma"][:, 2] * 100,
                          color=CYAN, alpha=0.35, label="σ_z [cm]")
    for ct, side in A["contacts"]:
        ax["As"].axvline(ct,
                         color=(GREEN if side == "left" else ACCENT),
                         alpha=0.55, lw=1.0, zorder=3)
    ax["As"].set_ylim(bottom=0)
    ax["As"].legend(fontsize=7.5)
    style(ax["As"], "Exp A — EKF position uncertainty (σ diag P)", yl="std dev [cm]")

    # ── EXP B ── push recovery ──────────────────────────────────────────────────
    ax["Bx"].plot(B["ts"], B["gt_pos"][:, 0],  color=BLUE,  lw=2.0)
    ax["Bx"].plot(B["ts"], B["ekf_pos"][:, 0], color=GREEN, lw=1.4, ls="--")
    ax["Bx"].plot(B["ts"], B["dr_pos"][:, 0],  color=RED,   lw=1.0, ls=":")
    shade_pushes(ax["Bx"], B["push_times"])
    shade_contacts(ax["Bx"], B["contacts"])
    style(ax["Bx"], "Exp B — Push recovery: forward (X)", yl="x [m]")

    ax["By"].plot(B["ts"], B["gt_pos"][:, 1],  color=BLUE,  lw=2.0, label="GT")
    ax["By"].plot(B["ts"], B["ekf_pos"][:, 1], color=GREEN, lw=1.4, ls="--", label="EKF")
    ax["By"].plot(B["ts"], B["dr_pos"][:, 1],  color=RED,   lw=1.0, ls=":",  label="DR")
    shade_pushes(ax["By"], B["push_times"])
    shade_contacts(ax["By"], B["contacts"])
    for pt in B["push_times"]:
        ax["By"].annotate("  60 N →", xy=(pt, 0.02), fontsize=7,
                          color=ACCENT, rotation=90, va="bottom")
    ax["By"].legend(fontsize=7.5)
    style(ax["By"], "Exp B — Push recovery: lateral Y (push axis)", yl="y [m]")

    err_ekf = np.linalg.norm(B["ekf_pos"][:, :2] - B["gt_pos"][:, :2], axis=1) * 100
    err_dr  = np.linalg.norm(B["dr_pos"][:, :2]  - B["gt_pos"][:, :2], axis=1) * 100
    ax["Be"].plot(B["ts"], err_ekf, color=GREEN, lw=1.5, label="EKF+contact")
    ax["Be"].plot(B["ts"], err_dr,  color=RED,   lw=1.5, label="Dead reckon.")
    shade_pushes(ax["Be"], B["push_times"])
    ax["Be"].set_ylim(bottom=0)
    ax["Be"].legend(fontsize=7.5)
    style(ax["Be"], "Exp B — 2D position error", yl="||Δp|| [cm]")

    # ── EXP C ── long walk / drift ─────────────────────────────────────────────
    ax["C2"].set_facecolor(BG)
    # decorative obstacle circles (visual reference — not in simulation)
    OBS = [(1.0, 0.15, 0.18), (1.8, -0.12, 0.15), (2.6, 0.10, 0.17)]
    for (ox, oy, r) in OBS:
        ax["C2"].add_patch(plt.Circle((ox, oy), r, color="#5c3a1a", alpha=0.55))
        ax["C2"].add_patch(plt.Circle((ox, oy), r, fill=False,
                                      color=MUTED, lw=0.9))
    ax["C2"].plot(C["gt_pos"][:, 0],  C["gt_pos"][:, 1],
                  color=BLUE,  lw=2.0, label="Ground truth", zorder=4)
    ax["C2"].plot(C["ekf_pos"][:, 0], C["ekf_pos"][:, 1],
                  color=GREEN, lw=1.4, ls="--", label="EKF+contact", zorder=5)
    ax["C2"].plot(C["dr_pos"][:, 0],  C["dr_pos"][:, 1],
                  color=RED,   lw=1.0, ls=":",  label="Dead reckon.", zorder=3)
    ax["C2"].plot(C["gt_pos"][0, 0],  C["gt_pos"][0, 1],
                  "o", color=BLUE, ms=8, zorder=6)
    ax["C2"].plot(C["gt_pos"][-1, 0], C["gt_pos"][-1, 1],
                  "s", color=ACCENT, ms=8, zorder=6)
    ax["C2"].set_aspect("equal")
    ax["C2"].legend(fontsize=7.5, loc="upper left")
    style(ax["C2"], "Exp C — 20-step walk: 2D trajectory",
          xl="x [m]", yl="y [m]")

    ax["Cv"].plot(C["ts"], C["gt_vel"][:, 0],  color=BLUE,   lw=1.8, label="True  vx")
    ax["Cv"].plot(C["ts"], C["ekf_vel"][:, 0], color=GREEN,  lw=1.4, ls="--", label="EKF  vx")
    ax["Cv"].plot(C["ts"], C["gt_vel"][:, 1],  color=PURPLE, lw=1.4, alpha=0.8, label="True  vy")
    ax["Cv"].plot(C["ts"], C["ekf_vel"][:, 1], color=CYAN,   lw=1.2, ls="--", label="EKF  vy")
    shade_contacts(ax["Cv"], C["contacts"])
    ax["Cv"].legend(fontsize=7.5, ncol=2)
    style(ax["Cv"], "Exp C — Velocity estimation (EKF vs truth)", yl="vel [m/s]")

    err_ekf_c = np.linalg.norm(C["ekf_pos"][:, :2] - C["gt_pos"][:, :2], axis=1) * 100
    err_dr_c  = np.linalg.norm(C["dr_pos"][:, :2]  - C["gt_pos"][:, :2], axis=1) * 100
    ax["Cd"].fill_between(C["ts"], err_dr_c,  color=RED,   alpha=0.35, label="Dead reckon.")
    ax["Cd"].fill_between(C["ts"], err_ekf_c, color=GREEN, alpha=0.55, label="EKF+contact")
    ax["Cd"].set_ylim(bottom=0)
    ax["Cd"].legend(fontsize=7.5)
    style(ax["Cd"], "Exp C — Cumulative drift over 20 steps", yl="||Δp|| [cm]")

    # global legend
    fig.legend(handles=legend_lines, loc="upper center", ncol=5,
               frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, 0.975))

    fig.text(0.99, 0.005, "G1 · MuJoCo 3 · Floating-Base EKF · RS1 2026",
             ha="right", va="bottom", fontsize=7, color=MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=155, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
    plt.close(fig)


def main():
    out = ROOT / "logs" / "ekf_demo.png"

    print("Exp A — flat walk (10 steps, base accuracy)...")
    A = run_experiment(n_steps=10, seed=1)
    print(f"  fell={A['fell']}, touchdowns={len(A['contacts'])}")

    print("Exp B — push recovery (10 steps, 60 N x2)...")
    B = run_experiment(n_steps=10, push_steps=[2800, 5200], seed=2)
    print(f"  fell={B['fell']}, pushes={[f'{t:.2f}s' for t in B['push_times']]}")

    print("Exp C — long walk (20 steps, drift accumulation)...")
    C = run_experiment(n_steps=20, seed=3)
    print(f"  fell={C['fell']}, touchdowns={len(C['contacts'])}")

    plot_all(A, B, C, out)


if __name__ == "__main__":
    main()
