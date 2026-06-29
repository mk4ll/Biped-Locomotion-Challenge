# Δυναμική Βάδιση Unitree G1 σε MuJoCo με Torque-level Whole-Body Control

> **Ζωντανή αναφορά (report backbone).** Ενημερώνεται με ΚΑΘΕ ουσιαστική αλλαγή (DESCRIPTION.md §9).
> Δομή: Τίτλος → Abstract → Πρόβλημα → Θεωρία → Αρχιτεκτονική → Υλοποίηση → Παράμετροι →
> Πειράματα/Αποτελέσματα → Decisions & Changelog → Πώς τρέχει → Περιορισμοί → Αναφορές.

**Ομάδα:** 2 άτομα · **Μάθημα:** Robotic Systems 1 · **Πλατφόρμα:** MuJoCo 3.4 + Python 3.11

---

## 2. Abstract

Κάνουμε το **Unitree G1** (humanoid, 29 actuated DOF + floating base) να **στέκεται** και να
**περπατά δυναμικά** σε MuJoCo, πρώτα σε επίπεδο έδαφος και μετά σε **κεκλιμένο** επίπεδο.
Η προσέγγιση είναι **dynamics-first / torque-level Whole-Body Control**: ανά control step λύνουμε
ένα **QP Inverse Dynamics** (Lecture 9) με μεταβλητές `[v̇; τ; f]`, και στέλνουμε **ροπές τ** στους
torque actuators (όχι IK+PD). Ο planner (footsteps + DCM/ZMP) είναι ανεξάρτητος από τον controller.
Όλη η balance-λογική εκφράζεται **ως προς τη βαρύτητα / την επιφάνεια** ώστε να γενικεύεται στην κλίση.

**Κατάσταση:** Στάδια 0–6 ✅ (setup, gravity comp, WBC standing, DCM planner, flat walking,
push recovery, **incline: stand 26°=arctan μ / walk 3°**) · Στάδιο 7 (παράδοση: video, polish) ⏳.

---

## 3. Πρόβλημα & Στόχοι

- Δυναμική, σταθερή βάδιση ενός humanoid σε MuJoCo με έλεγχο σε επίπεδο ροπής.
- Robust σε flat, σε ήπιες διαταραχές (push), και σε **κεκλιμένο επίπεδο** (το challenge).
- Καθαρά interfaces: ο WBC controller αντικαταστάσιμος, ο planner ανεξάρτητος.
- **Done όρια ανά στάδιο** ορίζονται στο `CHECKLIST.md`.

---

## 4. Θεωρητικό υπόβαθρο

### 4.1 Floating-base δυναμική
```
M(q) v̇ + h(q,v) = Sᵀ τ + Σ_c J_cᵀ f_c
```
- `M(q)` (nv×nv): mass matrix — από `mj_fullM`.
- `h(q,v)` (nv): bias (Coriolis + φυγόκεντρες + βαρύτητα) — από `data.qfrc_bias`.
- `S` (nu×nv): selection matrix· τα **6 DOF της βάσης είναι μη ενεργοποιούμενα**.
- `J_c`, `f_c`: contact Jacobian & δύναμη επαφής ανά σημείο επαφής πέλματος.

### 4.2 Whole-Body Control ως QP (Lecture 9)
Μεταβλητές `x = [v̇ ; τ ; f]`.
```
min_x  ½ xᵀ Q x + qᵀ x      με  Σ_i w_i ‖W_i x − t_i‖²   →  Q = ΣwWᵀW,  q = −ΣwWᵀt
s.t.   [M  −Sᵀ  −Jᵀ] x = −h               (δυναμική, equality)
       J_c v̇ + J̇_c v = 0                  (stance πέλματα, equality)
       friction cones, τ_min ≤ τ ≤ τ_max  (inequalities)
```

### 4.3 Tasks (acceleration-level με PD)
```
a_des = a_ref + Kp (x_ref − x) + Kd (ẋ_ref − ẋ)
residual:  J_task v̇ + J̇_task v − a_des ≈ 0
```
Tasks: **CoM** (μεγάλο βάρος), **swing/stance-foot**, **torso orientation** (κάθετο ως προς
βαρύτητα), **posture** (regularization). Προαιρετικά angular momentum.

### 4.4 Friction cones (κλειδί για την κλίση)
Γραμμικοποιημένος κώνος (πυραμίδα) **στο frame της επιφάνειας** (n = surface normal):
```
f_n ≥ f_min ,   |f_t1| ≤ μ f_n ,   |f_t2| ≤ μ f_n
```
Σε κλίση, `n` = κάθετος της επιφάνειας (όχι world-z) → ο solver «ξέρει» τι είναι εφικτό.

### 4.5 Walking pattern (Lecture 10)
- **LIPM**: `p_ZMP = p_CoM − (z/g) p̈_CoM`.
- **DCM / capture point**: `ξ = p_CoM + ṗ_CoM/ω`, `ω = √(g/z)` — foot placement & push recovery.
- **Support polygon**: convex hull σημείων επαφής· ZMP εντός.

### 4.6 FSM βάδισης
`DS_init → SS_L → DS → SS_R → DS → …`, σταθερός χρονισμός (π.χ. SS≈0.7s, DS≈0.2s).

---

## 5. Αρχιτεκτονική συστήματος (pipeline)
```
Offline:  footsteps + ZMP reference
   → Online: DCM / capture-point re-plan (CoM traj + foot paths + contact schedule)
   → WBC QP Inverse Dynamics (tasks: CoM+feet+torso+posture ; constraints: dynamics,
                              contacts, friction cones, torque limits) → τ
   → MuJoCo G1 (torque actuators)
   → feedback: base pose/twist, foot/CoM poses, contact state
```
Δομή κώδικα: βλ. `DESCRIPTION.md §5`. Modules: `src/sim`, `src/dynamics`, `src/control`,
`src/planning`, `src/utils`.

---

## 6. Λεπτομέρειες υλοποίησης (ανά module)

### Στάδιο 0 — Setup & μοντέλο ✅
- **Μοντέλο:** `models/unitree_g1/g1.xml` = το G1 του `mujoco_menagerie` με **μετατροπή των
  actuators από `position` → `motor` (torque)**. `ctrlrange` κάθε motor = `actuatorfrcrange`
  της άρθρωσης (π.χ. hip_pitch ±88, knee ±139, ankle ±50, wrist ±5 N·m). `meshdir` δείχνει στα
  assets του menagerie (no copy). Προστέθηκαν 4 **corner sites** ανά πέλμα για contact Jacobians.
- **Scenes:** `scene_flat.xml` (επίπεδο), `scene_incline.xml` (κεκλιμένο plane, euler about-y).
- **Στάση keyframe `stand`:** (α) **χέρια σε "δίσκο"** — βραχίονας κάτω, αντιβράχιο οριζόντιο
  μπροστά, **90° στον αγκώνα** (`shoulder_pitch=0.1, elbow=-0.15`· σημείωση: το elbow≈0 του G1
  είναι το διπλωμένο, ≈1.57 το ίσιο). (β) **bent-knee crouch** (`hip=-0.369, knee=0.914,
  ankle=-0.527`, base z=0.728, CoM≈0.65 m, πέλματα επίπεδα) → φυσική βάδιση με ορατά
  λυγισμένα γόνατα.
- **Discovery (`scripts/00_inspect_model.py`):** `nq=36, nv=35, nu=29`· floating base = dof 0–5
  (μη ενεργοποιούμενα)· **ΟΛΟΙ οι actuators motor/torque** (gain=fixed, bias=none)· μάζα
  **33.34 kg** → βάρος **327 N**· base height 0.79 m. Frames: `pelvis`, `torso_link`,
  `left_foot`/`right_foot` (+ corners).
- **`src/sim/mujoco_env.py`:** load/reset/step/set_torques (+ ctrl clamp), selection matrix `S`.
- **`src/utils/config.py`:** φορτώνει το `config/params.yaml` (single source of truth).

### Στάδιο 1 — Dynamics plumbing & gravity compensation ✅
- **`src/dynamics/model_terms.py`:** εξάγει `M` (`mj_fullM`), `h` (`qfrc_bias` = Coriolis+βαρύτητα),
  selection matrix `S`, contact Jacobians `J_c` (translational `mj_jacSite` στα 8 corner points),
  και drift `J̇_c v` (spatial site accel με `q̈=0` μέσω `mj_rnePostConstraint` — αναλυτικό, =0 στο στατικό).
- **`src/control/gravity_comp.py`:** στατική Inverse Dynamics. Στο `v=0,v̇=0`: `Sᵀτ + Σ J_cᵀ f = h`.
  Οι 6 base (μη ενεργοποιούμενες) εξισώσεις λύνονται για contact forces `f` ως **equality-constrained
  least squares** γύρω από ομοιόμορφη κατακόρυφη κατανομή βάρους (`f₀ = W/8` ανά σημείο):
  `f = f₀ + Aᵀ(AAᵀ+εI)⁻¹(b − A f₀)`, `A = (J_cᵀ)|base`. Μετά `τ = (h − J_cᵀ f)|actuated`.
  Επαναϋπολογίζεται κάθε control step → quasi-static feedback, απορρίπτει numerical drift.
- **Αποτέλεσμα (`scripts/01_gravity_comp.py`, 5 s @ 500 Hz):** base drift **1.6 mm**, CoM drift
  **0.66 mm**, base |vel|→~0, dynamics residual **2.3e-4**, max |τ| 1.4 N·m. → **PASS** (όρθιο & ακίνητο
  μόνο με feedforward ροπές, όχι glued). Plot: `logs/stage1_gravity_comp.png`.

### Στάδιο 2 — WBC QP & standing balance ✅
- **`src/control/wbc_qp.py`:** το QP Inverse Dynamics. Μεταβλητές `x=[v̇(35); τ(29); f(3·ncp)]`.
  - Objective: `Σ w_i‖J_i v̇ − b_i‖²` + tiny regularization (στρικτά PD για `quadprog`).
  - Equalities: δυναμική `[M −Sᵀ −J_fᵀ]x = −h` (35) + **rigid contact 6D ανά stance foot** `J_con v̇ = −J̇v` (6/πόδι).
  - Inequalities: friction pyramid στα corner forces + `τ` limits (lb/ub).
  - **Safe fallback** σε infeasibility (clarabel → +reg → gravity-comp τ).
- **`src/control/tasks.py`:** CoM (`mj_jacSubtreeCom`), Orientation (pelvis upright vs gravity),
  Posture (regularization προς keyframe), Foot (swing). Όλα PD acceleration-level.
- **`src/dynamics/contacts.py`:** `ContactSet` (double/left/right) + `friction_pyramid` στο
  **surface frame** (R_surface, flat=I).
- **Κρίσιμη σχεδιαστική επιλογή:** οι 4 corner accelerations/πόδι είναι **redundant** (rigid foot,
  rank 6) → το `quadprog` αποτυγχάνει. Λύση: contact equality στο **6D foot site** (full-rank),
  ενώ οι **δυνάμεις** μένουν στα 4 corners (για friction cone/CoP). Spec-faithful hard equality.
- **Αποτέλεσμα (`scripts/02_stand_balance.py`):** stand σταθερό (Σfz=327 N), weight-shift CoM range
  111 mm (err 15.7 mm), single-support balance με δεξί πόδι airborne (28 mm). **QP 100% feasible.**
  → **PASS**. Plot: `logs/stage2_stand_balance.png`.
### Στάδιο 3 — Planning layer ✅
Πλήρως **ανεξάρτητο από τον controller** (παράγει references, δεν κινεί το ρομπότ):
- **`footstep_planner.py`:** εναλλασσόμενα βήματα (left/right lanes), advance `step_length`/βήμα,
  kinematic clamp `1.5·step_length`. Χτίζει το **timeline φάσεων** `DS_init → SS → DS → … → DS_final`
  με ZMP endpoints ανά φάση (SS: στο stance foot· DS: ράμπα μεταξύ διαδοχικών stance feet).
- **`fsm.py`:** `phase_at(t)` → φάση + progress `s∈[0,1]`, `support_mode` → ContactSet mode.
- **`com_planner.py` (DCM):** `ω=√(g/z)`, `ξ=p+ṗ/ω`. **Backward** DCM recursion (ευσταθές)
  `ξ[k]=p_zmp[k]+(ξ[k+1]−p_zmp[k])e^{−ωΔt}` με `ξ_end=p_zmp_end`, μετά **forward** CoM
  `ṗ=ω(ξ−p)`. Κρατά το ZMP εντός support polygon εξ ορισμού.
- **`swing_planner.py`:** smoothstep για xy (μηδενική ταχύτητα στα άκρα), `(1−cos)` bump για z
  (μηδενική ταχύτητα σε lift-off/apex/touchdown).
- **`walk_plan.py`:** orchestrator → `reference(t)` = {com, com_vel, zmp, support, swing_pos/vel}
  (αυτό καταναλώνει το Στάδιο 4).
- **Αποτέλεσμα (`scripts/03_plan_walk.py`):** 8 βήματα, 8.6 s, `ω=3.76`, CoM x: 0→1.15 m, lateral
  sway ±0.08 m, swing apex 0.05 m, **ZMP-in-foot 8/8 SS**, DCM bounded (max|DCM−ZMP|=0.16 m).
  → **PASS**. Plot: `logs/stage3_plan.png`.
### Στάδιο 4 — Δυναμική βάδιση σε flat ✅
- **`src/control/walking_controller.py`:** δένει planner → WBC. Ανά step: `reference(t)` → CoM/swing
  targets + contact mode → WBC solve → torques. Επαναχρησιμοποιείται σε Στάδια 5/6.
- **CoM command (DCM feedback):** αντί position-PD, xy μέσω `ξ=com+v/ω`,
  `ξ̇_cmd=ξ̇_ref+k_dcm(ξ_ref−ξ)`, `p_zmp_cmd=ξ−ξ̇_cmd/ω`, `a_com=ω²(com−p_zmp_cmd)`· z με PD ύψους.
  Σταθεροποιεί το ασταθές DCM mode (το καθαρό position-PD αποκλίνει).
- **Capture-point foot placement:** στην αρχή κάθε SS, μετατόπιση landing swing foot κατά
  `clip(gain·(ξ_meas−ξ_ref))` (ramp-in με το progress) → «πιάνει» την πλευρική απόκλιση. Καθοριστικό
  για τα **στενά πέλματα** του G1 (CoP authority ±0.03 m, `e^{ω·t_ss}≈12×` divergence/βήμα).
- **Swing foot 6D:** θέση (spline) + προσανατολισμός (επίπεδο πέλμα) → καθαρή επαφή στο touchdown.
- **Contact-set management:** SS → stance foot 6D constraint + friction μόνο εκεί· swing foot εκτός·
  στο DS επανεντάσσονται και τα δύο.
- **Tuning:** `t_ss=0.5, t_ds=0.15, step=0.10` (πιο γρήγορα/μικρά βήματα → λιγότερη ανά-βήμα divergence).
- **Αποτέλεσμα (`scripts/04_walk_flat.py`):** **10 βήματα, 7.9 s, 0.97 m**, χωρίς πτώση, CoM track err
  35 mm (max 57), QP 100% feasible, max|τ| 40 N·m. → **PASS**. Plot: `logs/stage4_walk_flat.png`.
### Στάδιο 5 — Robustness / push recovery ✅
- **Push στο sim:** `MujocoEnv.apply_external_force` → wrench στο pelvis (`xfrc_applied`) για ~100 ms.
- **Μηχανισμός ανάκαμψης:** το **capture-point foot placement** (συνεχής ενημέρωση κατά το SS):
  `foot_offset = clip(gain·(ξ_meas − ξ_ref))`. Μια ώθηση μετατοπίζει το μετρούμενο `ξ`, οπότε το
  swing foot προσγειώνεται προς την κατεύθυνση της ώθησης και «πιάνει» την πτώση.
- **Κρίσιμη βελτίωση:** η ενημέρωση του capture έγινε **συνεχής** (όχι μόνο στην αρχή του SS) — αλλιώς
  μια ώθηση στη μέση του βήματος αντιδρά με ~1 βήμα καθυστέρηση και το ρομπότ πέφτει.
- **Envelope ανοχής (single 100 ms push):**
  - Lateral (±y): **~80 N / 8 N·s** (Δv≈0.24 m/s) σε ευνοϊκή φάση· ~50 N robust σε όλες τις φάσεις.
  - Forward (+x): **~200 N / 20 N·s** (μεγάλη ανοχή — μακρύ πέλμα μπροστά).
  - Backward (−x): ασθενές (το πέλμα εκτείνεται 0.12 m μπροστά αλλά 0.05 m πίσω → λίγη CoP authority).
- **Αποτέλεσμα (`scripts/05_push_recovery.py`):** βάδιση με 3 ωθήσεις (50 N ±y, 100 N +x) → **καμία
  πτώση**, max lateral deviation 83 mm, ανάκαμψη σε 13 mm, ολοκλήρωση 0.90 m. → **PASS**.
  Plot: `logs/stage5_push_recovery.png`. (Ταχύτητα/κατεύθυνση παραμετροποιήσιμα μέσω `gait`.)
### Στάδιο 6 — Κεκλιμένο επίπεδο ✅ (το challenge)
Η γενίκευση στην κλίση γίνεται χωρίς re-write, γιατί όλη η balance λογική είναι ως προς τη βαρύτητα/επιφάνεια:
- **Friction cones στο surface frame:** το `R_surface = R_y(α)` περνά στο `friction_pyramid` ώστε
  ο κώνος να είναι γύρω από τον **surface normal** (όχι world-z). Έτσι ο solver «ξέρει» τι εφαπτομενικές
  δυνάμεις είναι εφικτές → δεν γλιστράει (όσο `tan α < μ`).
- **Torso vertical w.r.t. gravity** (το orientation task κρατά `R_des=I`, όχι normal επιφάνειας).
- **Feet flat on slope:** pre-tilt αστραγάλων κατά `α` (το 6D contact constraint κλειδώνει τον
  προσανατολισμό, οπότε τα πέλματα πρέπει να ξεκινούν επίπεδα)· posture nominal ενημερωμένο αναλόγως·
  swing foot `R_des=R_surface`. Footsteps/CoM ακολουθούν την κλίση (`z = x·tan α`).
**Ανάλυση ορίων (απλά μαθηματικά) — δύο μηχανισμοί αστοχίας:**
- **Ολίσθηση (slip):** η εφαπτομενική βαρύτητα συγκρατείται από τριβή μόνο αν
  `mg sinα ≤ μ mg cosα` ⟹ **`α_slip = arctan(μ)`**. Με `μ=0.5` → **26.6°**.
- **Ανατροπή (tip):** περιστροφή γύρω από την κατάντη ακμή όταν η γραμμή βαρύτητας βγει εκτός:
  `mg cosα · d ≥ mg sinα · H` ⟹ **`α_tip = arctan(d/H)`**, `d`=απόσταση CoM→κατάντη ακμή,
  `H`=ύψος CoM. **Όμως** ο ενεργός controller κρατά το CoM πάνω από τα πέλματα (το CoM χρειάζεται
  μόνο `|offset|<0.09 m`, ενώ το support είναι `0.17 m`), οπότε η ανατροπή **δεν δεσμεύει** —
  πάντα `α_slip < α_tip` για ρεαλιστικό μοντέλο. **Άρα το standing είναι slip-limited.**

**Αποτελέσματα (`scripts/06_walk_incline.py`):**
- **Standing: σταθερό έως 26°, ολισθαίνει στις 27°** (slip 4→71 mm), πέφτει 28°+. **Συμφωνεί με τη
  θεωρία `arctan(0.5)=26.6°`** — επιβεβαιώνει ότι οι surface-frame friction cones δουλεύουν σωστά.
- **Walking: ανηφορίζει σε 3°** (0.90 m, rise 47 mm), αποτυγχάνει στις 4°. Το walking **ΔΕΝ**
  περιορίζεται από slip/tip (επιτρέπουν ~26°) αλλά από τη **δυναμική single-support**: ο
  horizontal-LIPM/DCM planner δεν μοντελοποιεί την επίδραση της κλίσης στη δυναμική, και η γεωμετρία
  επαφής (πέλματα στην κλίση) μειώνει το περιθώριο. Slope-projected DCM → μελλοντική δουλειά (§11).
- Plot: `logs/stage6_walk_incline.png`.

---

## 7. Παράμετροι (από `config/params.yaml`)

| Παράμετρος | Τιμή | Νόημα |
|---|---|---|
| `model.base_body` | `pelvis` | floating base |
| `model.torso_body` | `torso_link` | torso orientation task |
| `wbc.friction_mu` | 0.5 | συντελεστής τριβής (conservative vs foot geom 0.6, margin) |
| `wbc.f_min` | 1 N | ελάχιστη normal δύναμη ανά corner point |
| `wbc.solver` | quadprog | QP backend (full-rank· proxqp δεν έχει Windows wheel) |
| `wbc.reg.{qddot,torque,force}` | 1e-4/1e-4/1e-3 | regularization (στρικτά PD QP) |
| `tasks.com.{w,kp,kd}` | 100 / 100 / 20 | CoM task (μεγάλο βάρος) |
| `tasks.orientation.{w,kp,kd}` | 20 / 80 / 18 | pelvis upright vs gravity |
| `tasks.posture.{w,kp,kd}` | 1 / 20 / 8 | regularization προς keyframe |
| `tasks.swing_foot.{w,kp,kd}` | 120 / 300 / 34 | swing foot (single/SS) |
| `gravity_comp.hold_kp/kd` | 60 / 8 | posture-hold PD (κρατά crouch ακίνητη, Στ.1) |
| keyframe knee | 0.914 rad | bent-knee crouch (φυσική βάδιση) |
| `env.gravity` | 9.81 | — |
| `env.incline_deg` | 3.0 | αρχική κλίση (Στάδιο 6) |
| `sim.control_rate_hz` | 1000 | ρυθμός-στόχος WBC QP (sim @ 500 Hz, ~490 Hz πραγματικό) |
| `gait.n_steps` | 8 | πλήθος βημάτων |
| `gait.step_length` | 0.15 m | advance/βήμα |
| `gait.swing_apex` | 0.05 m | ύψος swing foot |
| `gait.t_ss / t_ds` | 0.5 / 0.15 s | διάρκειες single / double support |
| `gait.step_length` (Στ.4) | 0.10 m | μικρότερο βήμα για ευστάθεια |
| `tasks.com.k_dcm` | 3.0 | DCM feedback gain (xy) |
| `capture.gain / max_shift` | 1.0 / 0.12 m | capture-point step adjustment |
| `tasks.swing_foot.kp_ori` | 150 | swing foot flat (orientation) |

---

## 8. Πειράματα & Αποτελέσματα

| Πείραμα | Μετρική | Αποτέλεσμα | Plot |
|---|---|---|---|
| Gravity comp (Στ.1) | base drift / residual | 1.6 mm / 2.3e-4 | `logs/stage1_gravity_comp.png` |
| Standing balance (Στ.2) | weight-shift / single support | CoM 111 mm / foot airborne | `logs/stage2_stand_balance.png` |
| Walk plan (Στ.3) | ZMP-in-foot / DCM bounded | 8/8 / 0.16 m | `logs/stage3_plan.png` |
| **Flat walking (Στ.4)** | **βήματα / απόσταση / track err** | **10 / 0.97 m / 35 mm** | `logs/stage4_walk_flat.png` |
| **Push recovery (Στ.5)** | **max push lat/fwd** | **80 N / 200 N (100 ms)** | `logs/stage5_push_recovery.png` |
| **Incline (Στ.6)** | **stand / walk max** | **26° stand (=arctan μ) / 3° walk** | `logs/stage6_walk_incline.png` |

---

## 9. Decisions & Changelog

- **2026-06-24 — Στάδιο 0 (setup):**
  - Στήθηκε δομή φακέλων (`src/`, `scripts/`, `config/`, `models/`, `logs/`, `report/`).
  - **Απόφαση:** μετατροπή των G1 actuators `position → motor (torque)` — απαιτείται για
    torque-level WBC (DESCRIPTION §6). `ctrlrange = actuatorfrcrange`.
  - **Απόφαση:** corner sites ανά πέλμα (4×2) για μελλοντικά contact Jacobians/friction cones.
  - **Απόφαση solver:** `proxsuite/proxqp` δεν χτίζεται σε Windows → χρήση `quadprog` (default,
    dense PD) με fallback `osqp`. Δεν αλλάζει η μέθοδος, μόνο το backend.
  - **Επαλήθευση:** `00_inspect_model.py` → όλοι οι actuators motor/torque· headless test:
    εντολή 20 N·m στο knee → `actuator_force=20`, clamp στα limits. ✅ Done όρος Σταδίου 0.

---

- **2026-06-24 — Στάδιο 1 (gravity compensation):**
  - Νέα modules `src/dynamics/model_terms.py` (M, h, S, J_c, J̇v) και
    `src/control/gravity_comp.py` (static ID).
  - **Απόφαση:** point contacts στα 8 foot corner sites (3D δύναμη/σημείο), friction στο world-z
    (flat)· θα γενικευτεί σε surface normal στο Στάδιο 6.
  - **Απόφαση:** `J̇v` μέσω `mj_objectAcceleration` με `q̈=0` (αναλυτικό) αντί finite-diff.
  - **Επαλήθευση:** base drift 1.6 mm, CoM 0.66 mm, residual 2.3e-4 → PASS (Done όρος Σταδίου 1).

- **2026-06-24 — Στάδιο 2 (WBC QP standing balance):**
  - Νέα modules: `wbc_qp.py`, `tasks.py`, `contacts.py`. Tasks/βάρη/gains στο `params.yaml`.
  - **Bug #1 (rank):** 4 corner-point contact constraints/πόδι είναι redundant (rank 6) →
    `quadprog`/interior-point αποτυγχάνουν (PrimalInfeasible). **Fix:** rigid **6D foot-site**
    contact equality (full-rank), δυνάμεις στα corners. Solver: `quadprog` (γρήγορος, full-rank).
  - **Bug #2 (gravity in drift):** `mj_objectAcceleration` περιλαμβάνει το gravity offset στο
    `cacc`, μολύνοντας το `J̇v` με ~9.81 m/s² ακόμη και σε ηρεμία → ο contact constraint ανάγκαζε
    «πτώση» (`Σfz=8 N` αντί 327). **Fix:** μηδενισμός **βαρύτητας ΚΑΙ q̈** στον υπολογισμό του `J̇v`.
  - **Απόφαση:** single-support schedule = αργό CoM shift πάνω από το πόδι (2 s) → αργό lift (1 s)·
    το απότομο schedule έριχνε το ρομπότ (στενά πέλματα G1).
  - **Επαλήθευση:** stand/weight-shift/single-support όλα PASS, QP 100% feasible (Done όρος Σταδίου 2).

- **2026-06-24 — Model: χέρια στις 90° στους αγκώνες:**
  - Δεν ελέγχουμε τα χέρια· το keyframe `stand` άλλαξε ώστε `left/right_elbow_joint = 1.5708 rad`
    (90°) και το posture task τα κρατά εκεί. `ctrl` του keyframe → 0 (torque actuators, neutral).
  - Επαληθεύτηκε ότι το Στάδιο 2 εξακολουθεί να περνά με τη νέα στάση.
- **2026-06-24 — Στάδιο 3 (planning layer):**
  - Νέα modules: `footstep_planner.py`, `fsm.py`, `com_planner.py` (DCM), `swing_planner.py`,
    `walk_plan.py`. Όλες οι gait παράμετροι στο `params.yaml`.
  - **Απόφαση:** DCM (capture-point) αντί απλού LIPM preview — κλειστού-τύπου ανά βήμα + καθαρό
    feedback, ιδανικό για push recovery (Στάδιο 5). Backward recursion για ευστάθεια.
  - **Απόφαση:** ZMP στο stance foot (SS) και ράμπα στο DS → πάντα εντός support polygon.
  - **Επαλήθευση:** plots (footsteps/CoM/DCM/ZMP, swing height), ZMP-in-foot 8/8, DCM bounded
    → PASS (Done όρος Σταδίου 3, χωρίς κίνηση ρομπότ).

- **2026-06-24 — Στάδιο 4 (δυναμική βάδιση flat):**
  - Νέο `src/control/walking_controller.py` (planner↔WBC). FootTask επεκτάθηκε σε 6D (θέση+orientation).
  - **Bug/insight:** καθαρό CoM position-PD → πλευρική απόκλιση & πτώση στο ~4ο βήμα (ασταθές DCM mode).
    **Fix:** DCM feedback law για a_com (xy) + **capture-point foot placement**. Χωρίς αυτά, τα στενά
    πέλματα του G1 δεν έχουν αρκετή CoP authority.
  - **Απόφαση:** capture-point (Στάδιο 5 τεχνική) εισήχθη ήδη εδώ για robustness· θα επεκταθεί σε push
    recovery. Πιο γρήγορα/μικρά βήματα (`t_ss=0.5, step=0.10`).
  - **Επαλήθευση:** 10 βήματα, 0.97 m, χωρίς πτώση, QP 100% feasible → PASS (Done όρος Σταδίου 4).
- **2026-06-24 — Model: "δίσκος" χέρια + bent-knee crouch (αίτημα χρήστη):**
  - Χέρια: 90° στον αγκώνα, αντιβράχιο οριζόντιο μπροστά (βρέθηκε αριθμητικά: η γεωμετρία του G1
    elbow είναι ανεστραμμένη — 0=διπλωμένο). 
  - **Crouch keyframe** (knee=0.914) ώστε η βάδιση να έχει λυγισμένα γόνατα· κατά το walk τα γόνατα
    flex 50°–67°. Σκέτο χαμήλωμα CoM δεν αρκούσε (το WBC γέρνει στο ισχίο αντί να λυγίσει γόνατο)
    → χρειάστηκε bent-knee **posture nominal**.
  - **Bug:** το open-loop gravity comp (Στ.1) αποκλίνει σε crouch (μη παθητικά ευσταθής στάση).
    **Fix:** `hold_posture` joint-PD (ήδη στο spec) → κρατά την crouch ακίνητη (drift 1.7 mm).
  - Επαληθεύτηκε ότι Στάδια 1/2/4 περνούν με τη νέα στάση.

- **2026-06-24 — Στάδιο 5 (push recovery):**
  - `MujocoEnv.apply_external_force` (pelvis xfrc) + `scripts/05_push_recovery.py` (scheduled + sweep).
  - **Bug/insight:** capture-point μόνο στην αρχή του SS → ~1 βήμα καθυστέρηση → πτώση ακόμη και σε
    60 N. **Fix:** συνεχής ενημέρωση του foot_offset κατά το SS → ανάκαμψη έως ~80 N lateral.
  - **Εύρημα:** έντονη ασυμμετρία: forward ~200 N, backward ασθενές (γεωμετρία πέλματος).
  - **Επαλήθευση:** 3 ωθήσεις χωρίς πτώση, recovery 83→13 mm → PASS (Done όρος Σταδίου 5).

- **2026-06-24 — Στάδιο 6 (κεκλιμένο επίπεδο, το challenge):**
  - `scripts/06_walk_incline.py` (stand/walk/sweep). Friction cones στο `R_surface`, feet pre-tilt,
    slope-aware footsteps/CoM στον `WalkPlan` (`incline_rad`).
  - **Insight:** το 6D contact constraint κλειδώνει τον προσανατολισμό του πέλματος → τα πέλματα
    πρέπει να ξεκινούν επίπεδα στην κλίση (pre-tilt αστραγάλων + posture nominal).
  - **Αποτέλεσμα:** standing έως 18° (surface-frame friction cones δουλεύουν), walking 3°.
  - **Όριο:** incline walking >4° απαιτεί slope-projected DCM (μελλοντική δουλειά, §11).

- **2026-06-25 — Στάδιο 6: ανάλυση ορίων + push (αίτημα χρήστη):**
  - **Bug:** λάθος πρόσημο κλίσης — το `+x` ήταν κατηφόρα ενώ ο planner υπέθετε ανηφόρα → πέλματα
    διείσδυαν/αιωρούνταν. **Fix:** floor tilt `-α` (+x ανηφόρα), `R_surface=R_y(-α)`, ankle pre-tilt `-α`,
    ακριβές drop ώστε τα πέλματα να ακουμπούν.
  - **Ανάλυση φυσικής:** slip `arctan(μ)=26.6°` vs tip `arctan(d/H)`· ο ενεργός controller κρατά CoM
    πάνω από τα πέλματα → tipping δεν δεσμεύει → **slip-limited**.
  - **Πείραμα:** standing σταθερό **έως 26°, slip στις 27°** — ταιριάζει με τη θεωρία (έσπρωξε το
    όριο από την παλιά συντηρητική εκτίμηση 18° στο πραγματικό 26°). Walking 3° (δυναμικό όριο, όχι slip/tip).

```bash
pip install -r requirements.txt            # qpsolvers + quadprog/osqp/clarabel (Windows wheels)
python scripts/00_inspect_model.py         # Στάδιο 0: DOF, actuators, frames
python -m mujoco.viewer --mjcf=models/unitree_g1/scene_flat.xml   # οπτικός έλεγχος μοντέλου
python scripts/01_gravity_comp.py            # Στάδιο 1: gravity comp (headless metrics, PASS/FAIL)
python scripts/01_gravity_comp.py --viewer   # ίδιο με οπτικό παράθυρο
python scripts/02_stand_balance.py           # Στάδιο 2: WBC stand/weight-shift/single-support
python scripts/02_stand_balance.py --viewer  # ίδιο με οπτικό παράθυρο
python scripts/03_plan_walk.py               # Στάδιο 3: planner (plots, χωρίς ρομπότ)
python scripts/04_walk_flat.py               # Στάδιο 4: δυναμική βάδιση σε flat
python scripts/04_walk_flat.py --viewer      # ίδιο με οπτικό παράθυρο
python scripts/05_push_recovery.py           # Στάδιο 5: push recovery (scheduled pushes)
python scripts/05_push_recovery.py --viewer  # ίδιο με οπτικό παράθυρο
python scripts/05_push_recovery.py --sweep   # εύρεση max ανεκτού push
python scripts/06_walk_incline.py --deg 3    # Στάδιο 6: stand + walk σε 3° κλίση
python scripts/06_walk_incline.py --sweep    # max γωνία standing
python scripts/06_walk_incline.py --deg 8 --viewer   # στέκεται σε 8° (οπτικά)
```

---

## 11. Περιορισμοί & μελλοντική δουλειά
- **Incline walking ≤ ~3°** (ενώ standing φτάνει **26°=arctan μ**). Το walking δεν περιορίζεται από
  slip/tip αλλά από τη δυναμική single-support: ο οριζόντιος LIPM/DCM planner δεν μοντελοποιεί την
  κλίση στη δυναμική, και η γεωμετρία επαφής στην κλίση μειώνει το περιθώριο. Για ανηφορική βάδιση σε
  μεγαλύτερες γωνίες → **slope-projected LIPM/DCM** ή MPC, και 5D contact constraint (compliant foot
  pitch) ώστε τα πέλματα να conform-άρουν δυναμικά.
- **Push recovery ασύμμετρο**: ασθενές προς τα πίσω (γεωμετρία πέλματος, 0.05 m heel vs 0.12 m toe).
- **QP backend `quadprog`** αντί `proxqp` (δεν χτίζεται σε Windows) — ίδια μέθοδος, άλλο backend.
- **Feet pre-tilt για κλίση**: το 6D contact constraint δεν αφήνει το πέλμα να conform δυναμικά·
  εναλλακτικά θα μπορούσε contact constraint που αφήνει ελεύθερο το pitch (compliant foot).
- Χωρίς angular-momentum task (προαιρετικό Στ.5) — θα βελτίωνε περαιτέρω το push recovery.

---

## 12. Αναφορές
- Lecture 8 (CoM/biped balancing), Lecture 9 (WBC via QP), Lecture 10 (ZMP/LIPM/DCM).
- Kuindersma et al. 2016 — QP whole-body control (Atlas). Englsberger et al. 2015 — DCM.
- Pratt et al. 2006 — capture point. Del Prete — TSID. Caron — scaron.info.
