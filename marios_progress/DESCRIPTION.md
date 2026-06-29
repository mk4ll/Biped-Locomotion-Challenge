# DESCRIPTION.md — Δίποδη Δυναμική Βάδιση Unitree G1 (MuJoCo) με Whole-Body Control

> **Master document & single source of truth για το Claude Code.**
> Αντικαθιστά κάθε προηγούμενο spec. Διάβασέ το ΟΛΟ πριν γράψεις γραμμή κώδικα.
> Η αναλυτική λίστα με checkboxes ανά στάδιο βρίσκεται στο `CHECKLIST.md` (συνοδευτικό).
>
> ⚠️ **ΚΑΝΟΝΑΣ ΤΕΚΜΗΡΙΩΣΗΣ (δες §9): με ΚΑΘΕ αλλαγή κώδικα ενημερώνεις το `README.md`,
> που είναι η ραχοκοκαλιά της τελικής αναφοράς και τεκμηριώνει ΟΛΟ το project.**

---

## 1. Στόχος

Ομάδα 2 ατόμων. Κάνουμε το **Unitree G1** να **στέκεται** και να **περπατάει δυναμικά** σε
περιβάλλον **MuJoCo**, πρώτα σε επίπεδο έδαφος και μετά σε **κεκλιμένο** (το challenge του μαθήματος).
Στόχος: **robust & λειτουργικό**, όχι demo που πέφτει.

## 2. Προσέγγιση & Φιλοσοφία (δεν αλλάζει)

- **Dynamics-first / torque-level Whole-Body Control.** Λύνουμε ένα QP Inverse Dynamics ανά
  control step και στέλνουμε **ροπές `τ`** (Lecture 9). ΟΧΙ καθαρό IK+PD.
- **Σειρά υλοποίησης (αυστηρή):**
  `dynamics plumbing → gravity-compensation standing → balance/weight-shift → planner →
  flat walking → robustness → κλίση`.
- **Όλη η balance λογική εκφράζεται ως προς τη βαρύτητα / την επιφάνεια**, ΟΧΙ ως προς τον
  world άξονα z. Έτσι το ίδιο σύστημα γενικεύεται στην κλίση χωρίς ξαναγράψιμο.
- **Ο planner είναι ανεξάρτητος από τον controller.** Footsteps/CoM/DCM/swing μένουν ίδια·
  ο WBC είναι το κάτω, αντικαταστάσιμο επίπεδο.
- **Πάντα safe fallback** όταν το QP γίνεται infeasible (relax constraints, μη crash-άρεις).

## 3. Θεωρητικό υπόβαθρο

### 3.1 Δυναμική floating-base ρομπότ
```
M(q) v̇ + h(q,v) = Sᵀ τ + Σ_c J_cᵀ f_c
```
- `M(q)` : mass matrix (nv×nv)
- `h(q,v)` : bias term (Coriolis + φυγόκεντρες + βαρύτητα) — στη Lecture γράφεται `C(q,v)`
- `S` : selection matrix· τα 6 DOF της βάσης (freejoint) είναι **μη επενεργούμενα**
- `J_c`, `f_c` : contact Jacobian & δύναμη επαφής ανά πέλμα

### 3.2 Whole-Body Control ως QP (Lecture 9)
Μεταβλητές: `x = [v̇ ; τ ; f]`.
```
min_x  ½ xᵀ Q x + qᵀ x        όπου  Σ_i w_i ‖W_i x − t_i‖²  →  Q = WᵀW,  q = −Wᵀt
s.t.   [M  −Sᵀ  −Jᵀ] x = −h           (δυναμική — equality)
       J_c v̇ + J̇_c v = 0              (stance πέλματα δεν επιταχύνονται — equality)
       friction cones, torque limits   (inequalities, βλ. 3.4)
```

### 3.3 Tasks σε επίπεδο επιτάχυνσης
Κάθε reference του planner → επιθυμητή επιτάχυνση μέσω PD:
```
a_des = a_ref + Kp (x_ref − x) + Kd (ẋ_ref − ẋ)
residual:  J_task v̇ + J̇_task v − a_des ≈ 0     (γραμμικό στο v̇)
```
Tasks: **CoM** (μεγάλο βάρος), **swing-foot** (όταν στον αέρα), **stance-foot** (κράτημα),
**torso/base orientation** (κάθετος ως προς βαρύτητα), **posture** (regularization, μικρό βάρος),
προαιρετικά **angular momentum**.

### 3.4 Friction cones — το κλειδί για την κλίση
Κάθε `f_c` μέσα στον κώνο τριβής, γραμμικοποιημένο ως πυραμίδα, **στο frame της επιφάνειας**:
```
f_z ≥ f_min ,   |f_x| ≤ μ f_z ,   |f_y| ≤ μ f_z
```
Σε κλίση, `n` = **κάθετος της επιφάνειας** (όχι world-z). Έτσι ο solver «ξέρει» τι εφαπτομενικές
δυνάμεις είναι εφικτές και δεν γλιστράει. Μαζί: **torque limits** `τ_min ≤ τ ≤ τ_max`,
προαιρετικά CoP εντός πέλματος.

### 3.5 Walking pattern generation (Lecture 10)
- **LIPM**: `p_ZMP = p_CoM − (z_CoM/g) p̈_CoM` (σταθερό CoM height).
- **DCM / capture point** (προτιμώμενο): `ξ = p_CoM + ṗ_CoM/ω`, `ω = √(g/z_CoM)` — κλειστού-τύπου
  CoM ανά βήμα + καθαρό feedback. Capture point για foot placement & push recovery.
- **Support polygon**: convex hull των σημείων επαφής· το CoM/ZMP πρέπει να μένει εντός.

### 3.6 FSM / φάσεις βάδισης
`DS_init → SS_L → DS → SS_R → DS → …` με σταθερό χρονισμό (π.χ. SS≈0.7s, DS≈0.2s).
Σε κάθε φάση: support foot, swing foot, phase progress, **contact set** (ποια πέλματα έχουν
contact-constraint + friction cone ενεργά).

### 3.7 Γιατί γενικεύεται στην κλίση
(1) Friction cones ως προς surface normal → σέβεται τι μπορούν τα contacts. (2) Πλήρη `M, h` →
λαμβάνει υπόψη τη συνιστώσα βαρύτητας κατά μήκος της κλίσης + αδρανειακά. (3) Torque limits.
(4) Force-awareness → ομαλότερο, ανθεκτικότερο σε disturbances/model error.

## 4. Αρχιτεκτονική συστήματος (pipeline)
```
Offline:  footsteps + ZMP reference
   →  Online: DCM / capture-point re-plan  (CoM trajectory + foot paths + contact schedule)
   →  WBC QP Inverse Dynamics  (tasks: CoM + feet + torso + posture ; constraints: dynamics,
                                contacts, friction cones, torque limits)  →  τ
   →  MuJoCo G1 (torque actuators)
   ↑  feedback: base pose/twist, foot/CoM poses, contact state
```

## 5. Δομή αρχείων / Directory architecture
```
biped_g1_walking/
├── DESCRIPTION.md             # αυτό — master spec (single source of truth)
├── CHECKLIST.md               # στάδια υλοποίησης με checkboxes
├── README.md                  # ΖΩΝΤΑΝΗ αναφορά — ενημερώνεται με ΚΑΘΕ αλλαγή (βλ. §9)
├── requirements.txt
├── config/
│   └── params.yaml            # ΟΛΕΣ οι παράμετροι (gains, βάρη tasks, χρόνοι φάσης, μ, step size)
├── models/unitree_g1/
│   ├── g1.xml
│   ├── scene_flat.xml
│   └── scene_incline.xml      # plane υπό κλίση (το φτιάχνουμε εμείς)
├── src/
│   ├── sim/
│   │   ├── mujoco_env.py       # load, step, viewer, apply torques
│   │   └── robot_state.py      # q, qdot, base pose/twist, foot/CoM poses, contact state
│   ├── dynamics/
│   │   ├── model_terms.py      # M (mj_fullM), h (qfrc_bias), S, contact Jacobians, J̇v
│   │   └── contacts.py         # contact set management, friction-cone constraints
│   ├── control/
│   │   ├── wbc_qp.py           # QP Inverse Dynamics: μεταβλητές, equalities, inequalities, tasks
│   │   ├── tasks.py            # CoM / foot / torso / posture tasks (residual + Jacobian)
│   │   └── gravity_comp.py     # Στάδιο 1: gravity compensation (sanity test)
│   ├── planning/
│   │   ├── fsm.py              # gait phases + contact schedule
│   │   ├── footstep_planner.py # εναλλασσόμενα βήματα + capture point
│   │   ├── com_planner.py      # LIPM / DCM CoM trajectory
│   │   └── swing_planner.py    # swing-foot splines
│   └── utils/
│       ├── math_utils.py       # rotations, splines, transforms
│       └── logger.py           # logging + plots (CoM, ZMP, forces, τ)
├── scripts/
│   ├── 00_inspect_model.py     # DOF, actuator type, foot/pelvis frame names
│   ├── 01_gravity_comp.py      # Στάδιο 1
│   ├── 02_stand_balance.py     # Στάδιο 2
│   ├── 03_walk_flat.py         # Στάδιο 4
│   └── 04_walk_incline.py      # Στάδιο 6
├── logs/                       # plots, csv (πηγή για τα αποτελέσματα της αναφοράς)
└── report/                     # τελικό video, εικόνες, εξαγόμενο pdf
```

## 6. Εργαλεία & εξαρτήσεις
- **MuJoCo** (sim + δυναμικοί όροι: `mj_fullM`, `qfrc_bias`, `mj_jac`).
- **QP solver**: `qpsolvers` με `proxqp`/`quadprog` (γρήγορο, ~1 kHz).
- **Δρόμος robustness (συνιστάται): TSID + Pinocchio** — δίνει floating-base dynamics, contacts &
  friction cones έτοιμα. Προσοχή: URDF (Pinocchio) και MJCF (MuJoCo) να έχουν ίδιες μάζες/αδράνειες.
- numpy, scipy, matplotlib, pyyaml.
- ⚠️ **Απαιτούνται torque actuators** στο μοντέλο G1. Αν είναι position, άλλαξέ τους.

## 7. Στάδια υλοποίησης (περίληψη — αναλυτικά στο CHECKLIST.md)
0. Setup & απόφαση εργαλείου (torque actuators!)
1. Dynamics plumbing & **gravity compensation** (κρίσιμο sanity test)
2. QP Inverse Dynamics & standing balance (weight shift, single support)
3. Planning layer (footsteps, FSM, DCM, swing)
4. Δυναμική βάδιση σε flat (+ contact set management)
5. Robustness (capture point, push recovery)
6. Κεκλιμένο επίπεδο (surface-aligned friction cones)
7. Παραδοτέα

## 8. Κανόνες κώδικα
- Όλες οι παράμετροι στο `config/params.yaml`, ΠΟΤΕ hardcoded.
- Μικρές, δοκιμασμένες μονάδες· κάθε module με σαφή ευθύνη (βλ. §5).
- Μην προχωράς στάδιο πριν περάσει το «✅ Done όταν» του `CHECKLIST.md`.
- Commit μετά από κάθε στάδιο που περνά το κριτήριο.
- Καθαρά interfaces ώστε ο controller να είναι αντικαταστάσιμος.

## 9. ⚠️ ΚΑΝΟΝΑΣ ΤΕΚΜΗΡΙΩΣΗΣ — README.md ως ζωντανή αναφορά

**Με ΚΑΘΕ ουσιαστική αλλαγή (νέο module, νέα εξίσωση/μέθοδος, πείραμα, αλλαγή παραμέτρων,
αποτέλεσμα), ΕΝΗΜΕΡΩΣΕ το `README.md` στην ίδια εργασία.** Το `README.md` ΔΕΝ είναι οδηγός
εγκατάστασης — είναι **η ραχοκοκαλιά της τελικής αναφοράς** και πρέπει να τεκμηριώνει ΟΛΟ το project.

Δομή που πρέπει να διατηρεί το `README.md` (report backbone):
1. **Τίτλος & ομάδα**
2. **Abstract** — τι κάναμε, με τι μέθοδο, τι πετύχαμε
3. **Πρόβλημα & στόχοι**
4. **Θεωρητικό υπόβαθρο** — δυναμική, WBC QP, DCM/ZMP, friction cones (με τις εξισώσεις, σε δικά μας λόγια)
5. **Αρχιτεκτονική συστήματος** — pipeline + περιγραφή διαγράμματος
6. **Λεπτομέρειες υλοποίησης** — ανά module: τι κάνει, βασικές εξισώσεις, σχεδιαστικές επιλογές & γιατί
7. **Παράμετροι** — πίνακας από `params.yaml` με νόημα κάθε τιμής
8. **Πειράματα & αποτελέσματα** — flat / push recovery / κλίση· μετρικές (distance, duration, max κλίση)· γραφήματα από `logs/`
9. **Decisions & Changelog** — χρονολογημένο log κάθε σημαντικής αλλαγής + αιτιολόγηση
10. **Πώς τρέχει** — εντολές
11. **Περιορισμοί & μελλοντική δουλειά**
12. **Αναφορές**

Οδηγίες ενημέρωσης:
- Πρόσθεσε εγγραφή στο **Changelog (§9 του README)** σε κάθε αλλαγή: *τι, γιατί, ποιο στάδιο*.
- Όταν ολοκληρώνεται στάδιο του `CHECKLIST.md`, γράψε στο README τι αποδείχθηκε & βάλε τα plots.
- Κράτα τις εξισώσεις του README συγχρονισμένες με τον κώδικα (αν αλλάξει η μέθοδος, άλλαξε και το κείμενο).
- Στόχος: στο τέλος του project, το `README.md` να μπορεί να γίνει copy-paste σκελετός της αναφοράς.

## 10. Αναφορές
- Lecture 8 (CoM/biped balancing), Lecture 9 (WBC via QP), Lecture 10 (ZMP/LIPM/DCM/walking).
- Kuindersma et al. 2016 — QP whole-body control (Atlas), contacts + friction cones.
- Englsberger et al. 2015 — DCM. Pratt et al. 2006 — capture point.
- Del Prete — TSID (slides & video lessons). Caron — scaron.info (WBC & walking).
