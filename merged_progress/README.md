# Δυναμική Βάδιση Unitree G1 — Merged (Marios + Kanellos)

> **Merged υλοποίηση.** Κρατά την **αρχιτεκτονική/λογική του Μάριου** (καθαρό
> offline → online → WBC pipeline, stage-by-stage, θεωρητική ανάλυση) και
> **υλοποιεί όλα τα tasks του Κανέλλου** (terrain-aware incline/stairs,
> omnidirectional, push recovery, evaluation harness).

**Πλατφόρμα:** MuJoCo 3.4 · Python 3.11 · torque-level Whole-Body Control (QP Inverse Dynamics).

---

## 1. Αρχιτεκτονική (pipeline)

```
OFFLINE  (ανά πλάνο)            ONLINE  (ανά control step)         WBC (ανά step)
─────────────────────          ──────────────────────────        ───────────────
footstep planner   ┐           DCM feedback (capture point)       QP Inverse Dynamics
 + terrain.height  ├─► CoM/ZMP ─► xi = x + v/ω, p_zmp_cmd  ──────► min Σwᵢ‖Jᵢv̇−bᵢ‖²
 + footstep_x      │   (DCM     reactive step adjustment           s.t. dynamics,
gait FSM (DS/SS)   ┘   ref)     CoM-height low-pass                 contacts (6D),
swing splines                  heading/yaw track                   friction cones
                                                                    (surface frame),
                                                                    torque limits  → τ
        ▲────────────── feedback: CoM, foot poses, contacts, base pose ─────────────┘
                                   ↓ τ
                              MuJoCo G1 (torque actuators)
```

- **Offline** (`src/planning/`): footsteps (flat/incline/stairs/omnidirectional),
  ZMP reference, **DCM** trajectory (backward recursion), swing splines, gait timeline.
- **Online** (`src/control/walking_controller.py`): DCM feedback CoM law, capture-point
  step adjustment, terrain-aware friction-cone orientation, CoM-height low-pass, heading.
- **WBC** (`src/control/wbc_qp.py`): QP inverse dynamics, hard 6D contact + surface-frame
  friction cones + torque limits, safe fallback.

---

## 2. Τι κάνει (όλα επαληθευμένα — `scripts/evaluate.py`)

| Δυνατότητα | Αποτέλεσμα | Script |
|---|---|---|
| Gravity-comp sanity (Στ.1) | drift 1.7 mm | `01_gravity_comp.py` |
| WBC standing + weight-shift + single-support | PASS | `02_stand_balance.py` |
| **Flat walking** | 0.90 m, tilt 3.6° | `run_walk.py --terrain flat` |
| **Incline walking** | **8/12/16° (rise +126/191/255 mm)** | `run_walk.py --terrain incline --angle 16` |
| **Stairs climb** | **6×2.5 cm, +149 mm, full** | `run_walk.py --terrain stairs` |
| **Omnidirectional** | fwd/back/strafe/**curve +42°** | `run_omni.py --vx 0.1 --vyaw 0.12` |
| **Push recovery (walk)** | lateral 50 N / sagittal 100 N | `05_push_recovery.py` |
| Standing on incline (slip-limited) | **stable έως 26° = arctan μ** | `06_walk_incline.py --sweep` |
| **2ο ρομπότ — Talos (94 kg)** | **όλα τα tasks** (stand/walk/incline/push/fun) | `--robot talos` |
| 🍽 **FUN: Σερβιτόρος** (slalom + δίσκος/φραπέ) | **weaving** γύρω από τραπέζια, frappe σταθερό | `run_navigate.py` |
| 🪨 **FUN: Σίσυφος** (σπρώχνει μεγάλη μπάλα) | μπάλα ~ύψος χεριών +0.1 m uphill | `run_sisyphus.py` |
| **Επιλογή ταχύτητας** | slow/normal/fast (0.11/0.19/0.23 m/s) | `run_walk.py --step-len` ή menu `[s]` |
| **DCM Preview-MPC** (`--mpc`) | receding-horizon QP, N≈60 steps, anticipates narrow support | `run_walk.py --mpc` |
| **Arm swing** (`--arm-swing`) | contralateral hip→shoulder coupling, φυσική βάδιση | `run_walk.py --arm-swing` |

**Σημαντικό:** το incline walking πήγε από **3° → 16°** στο merge, χάρη στο
**terrain-aware design** (footsteps ON the surface, friction cones σε surface frame,
swing feet aligned to normal) — η κύρια συνεισφορά που υιοθετήθηκε από τον Κανέλλο.
**Όλα τα tasks τρέχουν και στα δύο ρομπότ** (G1 + Talos) μέσω του robot-agnostic stack.

---

## 2b. Fun tasks — αλγόριθμοι (πίστα & planned trajectory)

### 🍽 Σερβιτόρος: slalom course + planned trajectory
**Παραγωγή πίστας (`navigation.make_slalom`).** Αντί για ευθεία, χτίζουμε **slalom**:
1. **Τραπέζια (gates):** `n` τραπέζια τοποθετούνται **εναλλάξ** στην `+y` / `−y` πλευρά της
   κεντρικής γραμμής, ένα ανά «πύλη», με βήμα `dx` κατά `x`. Κάθε run προστίθεται **τυχαίο
   jitter** σε θέση/μέγεθος (random κάθε φορά). Κάθε τραπέζι = στρογγυλή ξύλινη επιφάνεια + 4 πόδια.
2. **Planned trajectory:** κατασκευάζεται **απευθείας** ως ημιτονοειδές **αντίθετης φάσης** από τα
   τραπέζια: `y(x) = −sign·A·cos(π(x−x₀)/dx)·env(x)`. Έτσι βυθίζεται στο `−A` ακριβώς εκεί που
   υπάρχει `+amp` τραπέζι (και αντίστροφα) → το ρομπότ **περνά κάθε πύλη από την ελεύθερη πλευρά**.
   Ένα `env(x)` (envelope) μηδενίζει το πλάτος πριν την 1η και μετά την τελευταία πύλη (ομαλή
   είσοδος/έξοδος).
3. **Γιατί δουλεύει (όριο στροφής):** το `dx` επιλέγεται ώστε η **καμπυλότητα** του ημιτόνου
   `κ = A·(π/dx)²` να μένει κάτω από τον ρυθμό στροφής που αντέχει το gait (≈0.11 rad/βήμα).
   Πολύ μικρό `dx` → απότομες στροφές → πτώση· γι' αυτό η πίστα **δεν** παράγεται με potential
   field (που σε στενό slalom δίνει χαοτικές 180° στροφές).
4. **Walkability retry:** ξαναδειγματίζουμε (νέο seed/jitter) μέχρι η πορεία **να καθαρίζει** κάθε
   τραπέζι (`min dist > r + 0.27 m`) **και** το curviness ανά βήμα `< 0.11` — ώστε κάθε τυχαίο run
   να είναι βατό.

**Από το trajectory στη βάδιση (pipeline):**
`slalom path` → `footstep_planner.plan_path` (βήματα κατά μήκος της πορείας, pelvis κοιτά την
εφαπτομένη, **per-step heading clamp** ώστε να μη «υπερ-στρίβει» το gait) → **DCM** CoM ref →
**WBC QP** → torques. Ο δίσκος+φραπές είναι geoms κολλημένα στον κορμό· επειδή ο κορμός μένει
κατακόρυφος, ο δίσκος μένει **επίπεδος** (ο φραπές δεν χύνεται — μετριέται το torso tilt < 12°).
Plot: `logs/navigate_*_seed*.png` (slalom weave + τραπέζια + **INITIAL/FINAL** θέση σερβιτόρου).

### 🪨 Σίσυφος
Το ρομπότ **τεντώνει τα χέρια μπροστά** (override του arm posture) και σπρώχνει μια **μεγάλη
μπάλα στο ύψος των χεριών** (`r≈0.36`). Η μπάλα είναι δεσμευμένη σε **slide joint κατά μήκος της
κλίσης** (+ damping) ώστε να μένει μπροστά σαν εκχιονιστήρας. Μια **forward CoM lean** (bias στην
επιτάχυνση CoM) αντισταθμίζει τη ροπή ανατροπής από την ψηλή επαφή — όπως ένας άνθρωπος γέρνει για
να σπρώξει. Η μπάλα ανεβαίνει κατά βήματα (push–rollback–push, ο Σισύφειος ρυθμός). Plot:
`logs/sisyphus_*deg.png`.

## 3. Ανάλυση ορίων κλίσης (απλά μαθηματικά)

- **Ολίσθηση:** `mg sinα ≤ μ mg cosα` ⟹ `α_slip = arctan(μ) = arctan(0.5) = 26.6°`.
- **Ανατροπή:** `α_tip = arctan(d/H)`· ο ενεργός WBC κρατά το CoM πάνω από τα πέλματα,
  οπότε η ανατροπή **δεν δεσμεύει** → standing **slip-limited**.
- **Πείραμα:** standing σταθερό **έως 26°, ολισθαίνει στις 27°** — ταιριάζει με τη θεωρία.
- Incline **walking** (16°) περιορίζεται δυναμικά (single-support), όχι από slip/tip.

---

## 4. Πώς τρέχει

### Διαδραστικό μενού (συνιστάται)
```bash
pip install -r requirements.txt
python main.py    # [1]..[d] task · [v] viewer · [r] robot G1<->Talos · [s] speed · [ESC] έξοδος
```
Κάθε task τρέχει σε **απομονωμένο subprocess** (robust: σφάλμα σε ένα δεν ρίχνει το μενού)
και τυπώνει «τι θα δεις». `[v]` = ζωντανός MuJoCo viewer · `[r]` = εναλλαγή **ρομπότ**
(G1 ↔ **Talos 94 kg**) · `[s]` = **ταχύτητα** (slow/normal/fast = step 0.10/0.16/0.20) ·
`[c]`/`[d]` = τα **fun tasks** (σερβιτόρος / Σίσυφος). Τα walking tasks τρέχουν στο επιλεγμένο ρομπότ.

**Robot-agnostic stack:** ο ίδιος controller/WBC/planner τρέχει G1 και Talos μέσω ενός
`RobotConfig` (`src/sim/robots.py`)· το Talos έχει box feet (προστίθενται corner contact
sites μέσω mjSpec) και είναι ήδη torque-controlled.

### Ταχύτητα βάδισης
- **~0.11 m/s** μέση (καθαρή απόσταση / συνολικό χρόνο)· **~0.15 m/s** σταθερό forward
  κατά τη συνεχή βάδιση (`step_length/T_step = 0.10/0.65`)· peak CoM ~0.38 m/s (με το sway).
- Ρυθμίζεται από `gait.step_length` / `t_ss` στο `config/params.yaml`, ή `vx` στο `run_omni.py`.

### Απευθείας scripts
```bash
# Stage-by-stage (η ραχοκοκαλιά της λογικής)
python scripts/00_inspect_model.py         # DOF, actuators (torque), frames
python scripts/01_gravity_comp.py          # dynamics sanity (gravity comp)
python scripts/02_stand_balance.py         # WBC standing / weight-shift / single-support
python scripts/03_plan_walk.py             # offline planner plots (no robot)

# Terrain-aware walking
python scripts/run_walk.py --terrain flat
python scripts/run_walk.py --terrain incline --angle 16
python scripts/run_walk.py --terrain stairs
python scripts/run_walk.py --terrain incline --angle 12 --viewer

# Omnidirectional
python scripts/run_omni.py --vx 0.12               # forward
python scripts/run_omni.py --vy 0.08               # strafe
python scripts/run_omni.py --vx 0.10 --vyaw 0.12   # curve

# Push recovery + incline analysis + full evaluation
python scripts/05_push_recovery.py
python scripts/06_walk_incline.py --sweep          # slip-limit vs theory
python scripts/evaluate.py                         # full battery -> logs/eval_report.md

# FUN tasks (G1 ή Talos via --robot)
python scripts/run_navigate.py                     # waiter: random slalom around tables + frappe
python scripts/run_navigate.py --tables 6 --viewer # more gates, live viewer
python scripts/run_sisyphus.py --viewer            # push a big ball up the slope
python scripts/run_sisyphus.py --robot talos --angle 4

# Lecture-style plots: path planning, footstep placement, CoM height, ZMP/DCM
python scripts/plot_walk.py --terrain flat
python scripts/plot_walk.py --terrain incline --angle 12
python scripts/plot_walk.py --terrain stairs       # CoM height climbs the treads
python scripts/plot_walk.py --omni curve           # curved path + rotated footsteps

# DCM Preview-MPC (replaces one-step proportional law with receding-horizon QP)
python scripts/run_walk.py --terrain flat --mpc
python scripts/run_walk.py --terrain incline --angle 12 --mpc
python scripts/run_walk.py --terrain flat --mpc --viewer  # watch it live

# Natural arm swing (contralateral coupling, configurable in config/params.yaml)
python scripts/run_walk.py --terrain flat --arm-swing
python scripts/run_walk.py --terrain flat --mpc --arm-swing  # both together
```

**Plots (`logs/plot_walk_*.png`)** — 4 panels ανά scenario:
(A) path planning & **footstep placement** top-down (foot rectangles + CoM/DCM/ZMP),
(B) **CoM height** vs forward position με την επιφάνεια του terrain (φαίνεται να σκαρφαλώνει),
(C) CoM/DCM/ZMP vs χρόνο, (D) swing-foot height (clearance).

---

## 5. Τι κρατήθηκε από τον καθένα (merge map)

| Από Μάριο (αρχιτεκτονική/λογική) | Από Κανέλλο (tasks/features) |
|---|---|
| offline/online/WBC pipeline, stage-by-stage scripts | terrain abstraction (`terrain.py`: height/normal/footstep_x) |
| hard **6D contact constraint** WBC (full-rank, quadprog) | terrain-aware **incline walking** (3°→16°) |
| DCM feedback CoM law + capture-point | **stairs** (tread-centre, feet-together-per-tread) |
| gravity-comp sanity test, bent-knee crouch, tray arms | **omnidirectional** (vx, vy, vyaw) |
| **slip/tip physics analysis**, README changelog | mjSpec terrain loading, low-pass CoM-height |
| self-contained menagerie | **evaluation harness** idea |

**Δομικές διαφορές που συμφιλιώθηκαν:** ο Μάριος χρησιμοποιεί hard 6D contact +
quadprog· ο Κανέλλος soft contact + OSQP. Το merged κράτησε το hard-6D του Μάριου
(ακριβές) και πρόσθεσε το terrain-aware layer του Κανέλλου από πάνω.

---

## 2b. DCM Preview-MPC — αλγόριθμος

Το `--mpc` αντικαθιστά τον one-step νόμο `p_cmd = p_ref + k(ξ − ξ_ref)` με ένα
**receding-horizon QP** που βελτιστοποιεί την ακολουθία CoP πάνω από ~60 preview βήματα.

**Deviation form (ανά άξονα — LIPM decoupling):**
```
delta_{k+1} = a·delta_k + (1−a)·dp_k ,   a = exp(ω·h) > 1
```
Το κόστος **αποκλιμακώνεται με a^{-2k}** ώστε να ακυρωθεί η εκθετική ανάπτυξη της
ασταθούς DCM δυναμικής (αλλιώς OSQP → «solved inaccurate»). Τα **constraints παραμένουν
πλήρη** — εκεί είναι η αντίληψη: το MPC βλέπει ότι μια επερχόμενη φάση SS στενεύει το
support box και μετατοπίζει το βάρος νωρίτερα.

Το Hessian είναι **σταθερό** (κατασκευάζεται μία φορά ως sparse matrix)· ανά tick
ανανεώνονται μόνο το gradient + τα box bounds. Ο solver τρέχει στα **50 Hz** (κρατάει
την τελευταία λύση ανάμεσα) ώστε ο viewer να παραμένει real-time.

## 2c. Arm swing — αλγόριθμος

Φυσική **αντιφασική ώθηση ώμου**: κατά τη SS φάση, το περιστρόφηση ώμου του *αντίθετου*
ποδιού τίθεται στο `q_nom[shoulder] = q_nom_base[shoulder] + k·sin(π·s)`, ενώ το
ομόπλευρο πηγαίνει `−k·sin(π·s)`. Η αρμονική μορφή εξασφαλίζει μηδενική ταχύτητα
στην έναρξη και λήξη κάθε βήματος. Στη DS φάση, επαναφέρεται η nominal στάση.

## 6. Περιορισμοί & μελλοντική δουλειά
- **Incline walking ≤ ~16°** (standing 26°)· steeper walking → slope-projected DCM/MPC.
- **Lateral push ~50 N** (sagittal 100 N+)· step-timing QP (Khadiv) θα το βελτίωνε
  (υπάρχει ήδη στο `kanellos_progress/biped/step_timing.py`).
- **Stairs**: δουλεύει για ήπια σκαλιά (2.5 cm)· πιο απότομα χρειάζονται μεγαλύτερο
  βήμα/clearance και per-tread gait tuning.
- **Arm swing sign**: το πρόσημο κέρδους εξαρτάται από τον άξονα κάθε άρθρωσης ώμου
  στο MJCF· αν φαίνεται ανάποδο, βάλε `gain: -0.08` στο `config/params.yaml`.
- **Online navigation**: η τρέχουσα υλοποίηση χρησιμοποιεί pre-planned sinusoidal
  trajectory (slalom)· online potential-field με live velocity command υπάρχει στο
  `kanellos_progress/biped/controllers/online_walking.py`.

## 7. Credits
- **Marios** — pipeline architecture, WBC (hard 6D contact), DCM feedback, gravity-comp,
  incline physics analysis, documentation.
- **Kanellos** — terrain-aware design (incline/stairs), omnidirectional planning,
  evaluation methodology, mjSpec terrain loading.

## 8. Αναφορές
Lecture 8 (CoM balancing), 9 (WBC via QP), 10 (ZMP/LIPM/DCM). Kuindersma 2016 (QP WBC).
Englsberger 2015 (DCM). Pratt 2006 (capture point). Del Prete (TSID). Caron (scaron.info).
