# G1 Biped Locomotion Challenge (MuJoCo, walking-only)

Simulated bipedal walking controller for the **Unitree G1** (29-DOF) in MuJoCo,
built to stay stable on **flat ground, inclines, and small stairs**. The
architecture is the model-based reactive stack used by modern bipeds:

```
  footstep / gait FSM   ->   DCM (capture-point) planning   ->   whole-body QP   ->   torque
   (when to step,             (CoM/ZMP reference +                (tracks CoM, swing
    where, contact state)      reactive step adjustment)           foot, posture; respects
                                                                   friction + torque limits)
        ^------------------------- state feedback (CoM, contacts, IMU) ----------------------|
```

**Why DCM over plain ZMP:** the divergent component of motion ξ = x + ẋ/ω isolates
the unstable part of the CoM dynamics into one quantity with simple dynamics
ξ̇ = ω(ξ − p_zmp). That gives a clean feedback law, makes "where to step to not
fall" ≈ "step toward the DCM", and generalizes to slopes far better than ZMP
preview control.

**Why the incline requirement drives the design:** friction cones are expressed
in the *contact-surface* frame (not world), swing feet land aligned to the
terrain normal, and CoM height is regulated relative to local foot height. These
are built in from the start (see `biped/terrain.py`), not bolted on.

## Install

```bash
pip install -r requirements.txt
# Robot models (sparse clone of the Menagerie):
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set unitree_g1 pal_talos && cd ..
```

**Two robots, one stack.** The controllers, planner and whole-body QP are
robot-agnostic; a `RobotConfig` (in `biped/robot.py`) captures everything that
differs. The **Unitree G1** (33 kg, 29 DoF, sphere feet, position servos →
torque) is the default; the **PAL Talos** (94 kg, 32 actuated DoF, box feet,
ships torque motors, passive coupled grippers) runs the *same* code via
`load_talos(...)` + `TALOS_CONFIG`. Talos scripts live in `scripts/talos/` so its
gait/gains can be tuned in isolation.

The Menagerie G1 ships with *position* actuators; `biped/robot.py` rewrites them
to torque motors via `mjSpec` at load time (and re-applies per-joint torque
limits, which motors don't inherit).

## Run

```bash
python scripts/run_stand.py                         # standing balance (flat)
python scripts/run_stand.py --shift --seconds 8     # lateral weight shift
python scripts/run_stand.py --push 150              # 150 N shove at t=2 s
python scripts/run_stand.py --terrain incline --angle 12
python scripts/run_stand.py --terrain stairs
python scripts/run_stand.py --view                  # interactive viewer
python scripts/run_stand.py --plot                  # save logs/stand_*.png
```

## Layout

| File | Role |
|------|------|
| `biped/robot.py`       | model loading (torque), state/kinematics facade: M, bias, CoM + Jacobian, foot poses/Jacobians/Jdot, contacts, index groups |
| `biped/terrain.py`     | flat / incline / stairs scenes + `height(x,y)` / `normal(x,y)` for terrain-aware control |
| `biped/poses.py`       | bent-knee nominal stance (feet flat, base height solved) |
| `biped/wbc.py`         | whole-body QP (inverse dynamics): CoM/posture/orientation tasks, soft contacts, friction cones, torque limits |
| `biped/controllers/standing.py` | Phase 1 standing + weight-shift controller |
| `scripts/run_stand.py` | Phase 1 demo / test harness with logging + plots |

## Whole-body QP

Decision variables: joint accelerations `qdd` (nv) and 3-D point contact forces
`f` (4 corner points per stance foot).

```
minimize    Σ w_i ||J_i qdd − b_i||²  +  reg
subject to  floating-base dynamics      (6 equalities — underactuation, hard)
            friction pyramid            (surface-frame, hard inequality)
            joint torque limits         (hard inequality)
```

Contact "points don't accelerate" is a **high-weight soft task** (using the
exact `mj_jacDot`·v) rather than a hard equality: 4 corner points per rigid foot
give 12 scalar constraints for 6 DOF, a redundant system that solvers flag as
infeasible. Torques are recovered as `τ = (M qdd + h − Jcᵀf)[actuated]`.

## Roadmap

- [x] **Phase 0** — setup; read M, bias, Jacobians, CoM, contacts; torque model
- [x] **Phase 1** — whole-body QP standing balance + lateral weight shift; push
      recovery in double support; verified on flat / incline / stairs
- [x] **Phase 2** — DS↔SS gait timeline + in-place stepping (quintic swing
      trajectories, planned contact schedule)
- [x] **Phase 3** — DCM planning + forward footstep sequence → continuous flat
      walking (1.4 m / 10 steps, ~3° tilt). Walks on a 6° incline too (rough).
- [~] **Phase 4** — DCM feedback (done) + reactive capture-point stepping
      **(position adjustment done** — `--reactive`: predict DCM to touchdown,
      shift swing target, commit to plan). Lifts lateral push tolerance at many
      gait phases without hurting nominal walking. **Step-timing adaptation**
      (land early to catch a fall) still needed for worst-case lateral pushes —
      requires event-driven DCM replanning (recompute the trajectory for the
      shortened step), not just a clock skip.
- [x] **Phase 5** — inclines **solid** (walks 14° at ~3° tilt: surface-frame
      cones + CoM-height adaptation + terrain-normal foot landing). Stairs
      **solved** via tread-centre footstep planning (`Terrain.footstep_x`): a
      feet-together-per-tread gait snaps each foot onto a tread centre, with a
      real top landing. Both robots climb the full staircase (G1 4 risers ~3°
      tilt, Talos 5 risers ~2°).
- [x] **Phase 6** — natural arm swing (contralateral hip coupling); automated
      evaluation harness (`scripts/evaluate.py`) producing rubric-mapped metrics
      → `logs/eval_report.md`
- [x] **Phase 7** — DCM **preview-MPC** (`--mpc`, `biped/dcm_mpc.py`):
      receding-horizon CoP optimisation with support-polygon constraints over
      the whole horizon. Deviation-form cost with `a^{-2k}` discounting for a
      well-conditioned QP. Drop-in replacement for the one-step CoP law; walks
      flat + inclines identically when undisturbed, anticipates support limits
      under disturbance.

### Walking modules (Phase 2–3)

| File | Role |
|------|------|
| `biped/swing.py`      | min-jerk swing-foot trajectory; `sin²` vertical bump (zero foot velocity at lift-off **and** touchdown — soft landing) |
| `biped/walk_plan.py`  | footstep planner + gait timeline + DCM (capture-point) trajectory (backward recursion, closed-form affine-ZMP DCM); planned support polygon per phase |
| `biped/dcm_mpc.py`    | DCM preview-MPC: receding-horizon CoP optimisation with support-polygon constraints over the horizon (`--mpc`) |
| `biped/gait.py`       | per-terrain gait profiles (flat / incline / stairs): step length, support timings, foot clearance finetuned to the surface |
| `biped/walk_plan.py` (`plan_walk_velocity`) | **omnidirectional** footstep planning from a body-frame velocity command `(vx, vy, vyaw)` — forward / back / strafe / turn / curve |
| `biped/navigation.py` | go-to-goal **navigation** with potential-field obstacle avoidance (+ tangential swirl) → velocity command *(prototype)* |
| `biped/step_timing.py` | capture-point **step-timing + footstep QP** (Khadiv) — adjusts where *and when* to step *(prototype, used by `online_walking`)* |
| `biped/controllers/walking.py` | DCM feedback **or** preview-MPC + support-polygon ZMP clamp + swing-foot task + low-passed CoM-height + arm swing + **heading/yaw** → WBC |
| `biped/controllers/online_walking.py` | event-driven online DCM walker with step-timing QP *(experimental)* |
| `scripts/run_walk.py` | walk demo / test (march or forward), metrics, plots, GIF |
| `scripts/evaluate.py` | automated rubric-mapped evaluation battery (terrain, push, friction, payload) → `logs/eval_report.md` |

```bash
python scripts/run_walk.py --step-len 0.0  --n-steps 8     # march in place
python scripts/run_walk.py --step-len 0.15 --n-steps 10    # walk forward
python scripts/run_walk.py --step-len 0.15 --video --plot  # save GIF + plots
python scripts/run_walk.py --step-len 0.15 --push 30       # mid-walk push test
python scripts/run_walk.py --step-len 0.15 --reactive --push 50   # capture-point stepping
python scripts/run_walk.py --step-len 0.15 --mpc                  # DCM preview-MPC
```

**Reactive stepping (`--reactive`)** predicts the DCM forward to touchdown and
shifts the swing-foot target by (predicted − nominal) DCM, ramped over late
swing with a deadband (so nominal walking is untouched), committing the
adjustment to the plan at landing. Effect: nominal walk unchanged; lateral push
tolerance 40 → 50 N at favorable gait phases; sagittal pushes already absorbed
to 100 N+. Worst-case lateral is unchanged — that needs step-timing adaptation.

**DCM preview-MPC (`--mpc`)** replaces the one-step proportional CoP law with a
receding-horizon QP (`biped/dcm_mpc.py`). Each tick it optimises the
centre-of-pressure over the next ~1.2 s to drive the *measured* DCM back to its
reference, **subject to the CoP staying inside the planned support polygon at
every preview step** — so it anticipates an upcoming single-support phase
(narrow support) and shifts weight early, which a one-step law cannot.

Two numerical points make it work: (1) the open-loop DCM is *unstable*
(`a = e^{ωh} > 1`), so the cost is written in **deviation form** and the per-step
tracking weight is **discounted by `a^{-2k}`** to cancel the `a^N` amplification
that otherwise wrecks the QP conditioning (OSQP → "solved inaccurate"); the
full-horizon *constraints* are kept — that is where the anticipation lives.
(2) the Hessian is constant (built once, sparse), so only the gradient and the
box bounds update per tick; the solve is throttled to 50 Hz and held between
solves so the live viewer stays real-time. On undisturbed flat ground MPC and
the proportional law track identically (both follow the reference); the MPC's
value appears under disturbance and tight support constraints.

### Omnidirectional walking & navigation

The same DCM + WBC stack walks in **any direction** from a body-frame velocity
command `(vx, vy, vyaw)` — the footstep planner places steps in the commanded
direction and the pelvis/foot yaw track the heading:

```bash
python scripts/render_walk.py --vx 0.25            # forward    (≈1.7 m)
python scripts/render_walk.py --vx -0.25           # backward
python scripts/render_walk.py --vy 0.12            # strafe left (pure sideways)
python scripts/render_walk.py --vx 0.15 --vyaw 0.12  # walk a curve
```

Translation (forward / back / strafe / diagonal) is rock-solid; turning is
gentler-limited (≈0.12 rad/s in place) — faster yaw is where this gait gets
fragile.

**Continuous online walker + navigation.** `biped/controllers/online_walking.py`
is an event-driven DCM walker whose velocity command `(vx, vy, vyaw)` can be
updated **every tick with no replanning/restart** (`set_velocity`), so the gait
flows continuously — the basis for reactive navigation. It also hosts the
capture-point **step-timing QP** (`biped/step_timing.py`) which adjusts *when* to
step, not just where.

`scripts/run_navigate.py` drives the robot to a goal around cylindrical obstacles:
a potential field (attraction + repulsion + tangential **swirl** to arc around
rather than stall head-on) sets the velocity command, **low-passed** so the gait
tracks it smoothly, and fed to the continuous walker each tick.

```bash
python scripts/run_navigate.py --goal 2.6 0.0     # arc around an obstacle to the goal
python scripts/run_navigate.py --video            # -> logs/navigate_g1.gif
```

The robot reaches goals while arcing around obstacles (e.g. +2.6 m around a
0.25 m obstacle, no fall). The online walker is robust for ~25 s of straight
walking; sustained large sideways detours are nearer its stability edge, so very
long slaloms remain the harder case.

### Watch it walk live (interactive MuJoCo viewer)

```bash
mjpython scripts/view_walk.py                       # macOS: must use mjpython
mjpython scripts/view_walk.py --mpc                  # DCM preview-MPC
mjpython scripts/view_walk.py --terrain incline --angle 6 --step-len 0.12
mjpython scripts/view_walk.py --terrain stairs      # climb a staircase
mjpython scripts/view_walk.py --march               # step in place
# Linux/Windows: plain `python scripts/view_walk.py`
```

The viewer walks continuously (auto re-plans each cycle).  **Ctrl+drag** a body
to shove the robot and watch it balance — visible proof of dynamic control.

### PAL Talos (same stack, separate scripts)

The 94 kg Talos runs the identical DCM + whole-body-QP code through its own
scripts in `scripts/talos/` (kept separate so its gait/gains tune independently):

```bash
python  scripts/talos/run_stand.py                  # standing balance
python  scripts/talos/run_walk.py  --step-len 0.15  # forward walk + metrics
python  scripts/talos/run_walk.py  --mpc            # preview-MPC
python  scripts/talos/run_balance.py --foot right   # one-leg balance
python  scripts/talos/evaluate.py  --quick          # rubric battery -> logs/talos_eval_report.md
mjpython scripts/talos/view_walk.py                 # watch it walk live
```

Talos walks out-of-the-box on the shared controllers: ~1.13 m / 8 steps at
~1.4° tilt on flat, stands at 0.14° tilt (foot force = body weight), and
balances on one foot. Arm swing is disabled for Talos (its shoulder axes don't
map to a clean sagittal pitch); everything else is shared.

### Dynamic control (not kinematic playback)

All 29 actuators are **torque motors** (verified: gaintype/biastype 0, force-
limited), and every sim step the whole-body QP solves *inverse dynamics* from
the measured mass matrix, bias forces and contact Jacobians, then commands joint
torques.  The DCM feedback closes the loop on the measured CoM.  Evidence: while
walking it recovers from **unplanned** pushes it never saw in any plan — 30 N
lateral / ≥50 N fore-aft (the lateral limit is low pending Phase-4 reactive
stepping).

**Key lessons baked in:** (1) lateral DCM is only correctable in double
support — the commanded ZMP is clamped to the support polygon so the QP never
chases an impossible CoP; (2) the CoM has low lateral authority through the
legs, so the CoM task needs a high weight (≈200) to win against posture/
orientation; (3) start-from-rest needs a longer initial weight-shift (`t_ds0`).

## Status (verified)

Headless, 33.3 kg G1, MuJoCo 3.6, OSQP. Each holds for the full duration:

| Scenario | Result |
|---|---|
| Stand (flat) | tilt 0.16°, foot force 164/163 N = body weight, peak τ 17 Nm |
| Weight shift ±5 cm | tilt 1.05°, stable 8 s |
| Push 80 N / 150 N (100 ms) | recovered in double support |
| Incline 8° / 15° | tilt < 3° |
| Stairs (standing) | stable |
| **Forward walk, flat** (10 × 0.15 m) | +1.42 m, ~3° tilt |
| **Forward walk, incline 14°** | climbs, ~3° tilt, feet land flat |
| Walk push recovery | 30 N lateral / ≥50 N fore-aft (no reactive stepping yet) |
| **Stairs climb** (4×3 cm, tread-centre) | full climb +0.12 m, ~3° tilt |

### Talos (94 kg, same stack)

Headless, MuJoCo 3.6, OSQP. Same controllers/planner/WBC as the G1:

| Scenario | Result |
|---|---|
| Stand (flat) | tilt 0.14°, foot force 460/462 N = body weight, peak τ 109 Nm |
| One-leg balance (right) | 5 s, tilt 0.8°, swing foot raised 14 cm |
| **Forward walk, flat** (10 × 0.15 m) | +1.44 m, ~1.6° tilt, peak τ 160 Nm |
| Forward walk, incline 4° / 8° | +1.14 m, ~1.4–1.6° tilt |
| Walk + preview-MPC (flat) | DCM-tracking RMS 0.0057 vs 0.0071 one-step (better) |
| **Stairs climb** (5×5 cm, tread-centre) | full climb +0.25 m, ~2° tilt |
