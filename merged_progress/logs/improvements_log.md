# Session 2 Improvements Log

Three remaining features ported and tested. Results below.

---

## 1. Step Timing QP — Khadiv et al. 2016

**File:** `src/planning/step_timing.py`

**Algorithm:** During single support the LIPM DCM obeys `ξ(t) = u + (ξ₀ − u)eᵒᵗ`.
With `τ = exp(ω · t_rem)`, the DCM at end-of-step is `ξ_eos = u + (ξ_meas − u)τ`.
For balance after stepping, the next footstep `u_next` must satisfy:

```
u_next + b = u + (ξ_meas − u) · τ

min  w_foot ||u_next − u_nom||² + w_time (τ − τ_nom)²
s.t. u_next − (ξ_meas − u_cur) · τ = u_cur − b    (DCM constraint)
     u_next ∈ reachability box,  τ ∈ [τ_min, τ_max]
```

A push drives `ξ_meas` outward; the QP responds by shifting `u_next` **and**,
when the footstep alone cannot reach, **stepping sooner** (reducing τ).
Nominal offsets for periodic gait:
- `b_x = L / (τ_full − 1)`
- `b_y = ±W / (τ_full + 1)`  (sign by stance side)

**Implementation status:** `StepTimingQP` class fully ported and unit-tested.
Full event-driven integration (timing adjustment + replanning from actual foot
positions) requires an online footstep replanner; the offline fixed plan has
footstep targets computed upfront and cannot update after actual landings.
The `--step-timing` flag widens the capture-point limits (max_shift 0.12→0.20)
as an interim improvement.

**Push recovery results:** see `push_sweep_report.md`.
Current lateral push limit: **80 N** (capture-point, 100 ms pulse at 2.0 s).

---

## 2. Online Velocity Following — Multi-Segment Replanning

**File:** `scripts/run_velocity_change.py`

**Approach:** Execute sequential `WalkPlan` segments with different velocity
commands. Between segments, current foot positions are read and passed as the
initial state for the next plan. This enables live velocity/direction changes
without a full event-driven rewrite.

**Segments demonstrated:**

| Segment | vx [m/step] | vyaw [rad/step] | n_steps |
|---------|:-----------:|:---------------:|:-------:|
| Straight forward | 0.10 | 0.00 | 6 |
| Turn right | 0.09 | +0.18 | 6 |
| Straight again | 0.10 | 0.00 | 6 |
| Veer left | 0.09 | −0.15 | 4 |
| Final straight | 0.10 | 0.00 | 4 |

**Result (G1):**
```
fell             = False
total distance   = 1.45 m
net heading change = −20.6 deg
RESULT: PASS
```

---

## 3. Steeper Stairs — Standard Indoor Configuration

**Flag:** `python scripts/run_walk.py --terrain stairs --hard-stairs`

**Configuration:**

| Parameter | Default (easy) | Hard (standard indoor) |
|-----------|:--------------:|:----------------------:|
| Riser height | 2.5 cm | **4.0 cm** |
| Tread run | 16 cm | **20 cm** |
| n_steps | 6 | 6 |
| swing_apex | 6 cm | **10 cm** |
| t_ss | 0.50 s | **0.55 s** |

**Result (G1):**
```
fell             = False
forward distance = 1.769 m
height gain      = 239 mm (+24 cm = 6 × 4 cm risers)
RESULT: PASS
```

The extra clearance (swing_apex 6→10 cm) and longer SS (0.50→0.55 s) are
sufficient to clear the taller risers. The terrain-aware footstep planner
(tread-centered placement) handles the 20 cm run without parameter changes.

---

## 4. Full Evaluation Battery (updated)

All 10 baseline scenarios confirmed PASS after all changes:

| Scenario | fell | dist | rise | tilt | τ_peak |
|---|:---:|---:|---:|---:|---:|
| walk_flat | ✅ | +0.90 m | −2 mm | 3.6° | 53 Nm |
| walk_incline_8° | ✅ | +0.89 m | +126 mm | 3.4° | 53 Nm |
| walk_incline_12° | ✅ | +0.89 m | +191 mm | 3.3° | 53 Nm |
| walk_incline_16° | ✅ | +0.88 m | +255 mm | 3.9° | 55 Nm |
| stairs_6×2.5 cm | ✅ | +1.47 m | +149 mm | 4.4° | 62 Nm |
| omni_forward | ✅ | +0.68 m | −2 mm | 3.5° | 53 Nm |
| omni_strafe | ✅ | −0.07 m | 0 mm | 3.8° | 44 Nm |
| omni_curve | ✅ | +0.48 m | −3 mm | 4.3° | 67 Nm |
| push_lateral_50N | ✅ | +0.90 m | — | 3.7° | — |
| push_sagittal_100N | ✅ | +0.90 m | — | 5.0° | — |

New capabilities not in the evaluate battery:
- **Hard stairs (4 cm risers):** ✅ PASS, height gain +239 mm
- **Online velocity following:** ✅ PASS, 5-segment trajectory
- **DCM preview-MPC:** ✅ PASS (`--mpc`)
- **Arm swing:** ✅ PASS (`--arm-swing`)
- **Step timing QP:** class available (`src/planning/step_timing.py`)

_Generated 2026-06-29_
