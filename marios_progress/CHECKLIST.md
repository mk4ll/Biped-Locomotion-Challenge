# Checklist Υλοποίησης — G1 Δυναμική Βάδιση με Whole-Body Control (Torque-level)

> Προσέγγιση: **dynamics-first**. Λύνουμε QP Inverse Dynamics ανά control step και στέλνουμε
> **ροπές `τ`** στο ρομπότ (Lecture 9). Στόχος: robust & λειτουργικό σε flat ΚΑΙ σε κλίση.
>
> Κανόνας: μην προχωράς στάδιο αν δεν τσεκαριστεί το «✅ Done όταν».
> Εργαλείο: συνιστάται **TSID (Pinocchio)** για robustness — σου δίνει floating-base dynamics,
> contacts & friction cones έτοιμα. Εναλλακτικά hand-rolled QP με όρους MuJoCo + `qpsolvers`.

---

## Στάδιο 0 — Setup & Απόφαση εργαλείου

- [ ] Virtual env + εγκατάσταση: `mujoco`, `numpy`, `scipy`, `qpsolvers`, `proxsuite`, `matplotlib`, `pyyaml`
- [ ] Αν Δρόμος TSID: εγκατάσταση `pinocchio` + `tsid` (μέσω conda/robotpkg) και τρέξιμο ενός official humanoid example τους
- [ ] Βάλε τα αρχεία G1 (mujoco_menagerie) στο `models/unitree_g1/`
- [ ] **ΚΡΙΣΙΜΟ: τσέκαρε τύπο actuators.** Αν είναι position actuators → άλλαξέ τους σε **motor/torque**
      (ή χρησιμοποίησε `data.qfrc_applied` / `data.ctrl` ως γενικευμένη δύναμη). Χωρίς torque control δεν γίνεται τίποτα.
- [ ] Αν Δρόμος TSID: εξασφάλισε ότι το URDF (Pinocchio) ΚΑΙ το MJCF (MuJoCo) έχουν **ίδιες μάζες/αδράνειες** (no model mismatch)
- [ ] Δημιούργησε τη δομή φακέλων (όπως στο CLAUDE.md)

**✅ Done όταν:** το scene φορτώνει, ανοίγει viewer, οι actuators δέχονται ροπές.

---

## Στάδιο 1 — Dynamics plumbing & gravity compensation (η «καρδιά»)

- [ ] Συνάρτηση που εξάγει ανά βήμα: `M` (`mj_fullM`), `h` = bias (`data.qfrc_bias`), selection matrix `S`
- [ ] Συναρτήσεις για contact Jacobians `J_c` των πελμάτων (`mj_jac` σε foot sites/corners)
- [ ] Όρος `J̇_c v`: ξεκίνα με προσέγγιση ≈ 0, βελτίωσε αργότερα (finite-diff ή `mj_objectAcceleration`)
- [ ] **Gravity compensation test** (Lecture 8): υπολόγισε `τ` ώστε `Sᵀτ + Σ J_cᵀf = h` σε διπλή στήριξη και κράτα το ρομπότ ακίνητο
- [ ] Επαλήθευσε ότι το ρομπότ ΔΕΝ καταρρέει υπό βαρύτητα όταν εφαρμόζεται το `τ`

**✅ Done όταν:** το ρομπότ μένει όρθιο & ακίνητο μόνο με τις υπολογισμένες ροπές (όχι «κολλημένο» στο sim).
Αυτό αποδεικνύει ότι `M, h, J_c` και το μοντέλο επαφών είναι σωστά. **Μη συνεχίσεις πριν περάσει.**

---

## Στάδιο 2 — QP Inverse Dynamics & Standing Balance

- [ ] Μεταβλητές QP: `x = [v̇ ; τ ; f]`
- [ ] Equality 1 — δυναμική: `[M  −Sᵀ  −Jᵀ] x = −h`
- [ ] Equality 2 — επαφή (stance): `J_c v̇ + J̇_c v = 0` για τα πέλματα σε επαφή
- [ ] Inequality — **friction cones** (πυραμίδα): `f_z ≥ f_min`, `|f_x| ≤ μ f_z`, `|f_y| ≤ μ f_z`
- [ ] Inequality — **torque limits**: `τ_min ≤ τ ≤ τ_max`
- [ ] Tasks (acceleration-level, με PD `a_des = a_ref + Kp·e + Kd·ė`):
  - [ ] CoM task (βάρος μεγάλο)
  - [ ] Foot pose tasks (διπλή στήριξη: κράτα πέλματα)
  - [ ] Torso/base orientation task (κάθετος ως προς **βαρύτητα**)
  - [ ] Posture task (μικρό βάρος, regularization)
- [ ] Στήσε `Q = WᵀW`, `q = −Wᵀt`· λύσε QP @ ~1 kHz με `proxqp`/`quadprog`
- [ ] **Infeasibility handling**: αν το QP αποτύχει, relax (π.χ. χαμήλωσε `f_min`/βάρη) αντί για crash
- [ ] Standing + **weight shifting** (CoM target αριστερά/δεξιά) + **single support** (σήκωσε 1 πόδι)

**✅ Done όταν:** στέκεται σταθερά, μετατοπίζει βάρος, και ισορροπεί σε ένα πόδι — όλα μέσω WBC ροπών.

---

## Στάδιο 3 — Planning layer (ίδιο με το κινηματικό)

- [ ] Footstep planner (εναλλασσόμενα βήματα από επιθυμητή ταχύτητα + kinematic clamp)
- [ ] FSM φάσεων: `DS_init → SS_L → DS → SS_R → …` (π.χ. SS ≈ 0.7 s, DS ≈ 0.2 s)
- [ ] CoM/ZMP planner με **DCM** (`ξ = p_CoM + ṗ_CoM/ω`, `ω = √(g/z_CoM)`)
- [ ] Swing-foot trajectory (cubic/quintic spline, apex ύψος, μηδενική ταχύτητα στο touchdown)
- [ ] Plots: ZMP μέσα στο support polygon, CoM, footsteps

**✅ Done όταν:** ο planner βγάζει συνεπείς references (CoM, foot paths, contact schedule) — επαληθευμένα σε γραφήματα, χωρίς το ρομπότ ακόμα.

---

## Στάδιο 4 — Δυναμική βάδιση σε επίπεδο έδαφος

- [ ] **Contact set management**: στο SS αφαίρεσε contact-constraint & friction-cone του swing foot· στο touchdown ξαναπρόσθεσέ τα
- [ ] Πρόσθεσε **swing-foot task** που τρακάρει το spline (μόνο όταν το πόδι είναι στον αέρα)
- [ ] Σύνδεσε references planner → task targets (CoM/foot/torso)
- [ ] Ομαλές μεταβάσεις βαρών στα task κατά DS↔SS (ramp, όχι απότομα)
- [ ] Tuning σειρά: PD gains των tasks → βάρη tasks → χρόνοι φάσης → step length/width
- [ ] Logging όλου του walk (CoM, ZMP, contact forces, τ)

**✅ Done όταν:** συνεχόμενη σταθερή βάδιση εμπρός σε flat ≥ 8–10 βήματα χωρίς πτώση.

---

## Στάδιο 5 — Robustness (disturbances)

- [ ] **Capture-point foot placement**: επόμενο βήμα κοντά στο `ξ`
- [ ] Push tests: εξωτερική δύναμη στο pelvis για λίγα ms → μέτρα ανάκαμψη
- [ ] (Προαιρετικά) angular-momentum task για καλύτερη σταθερότητα
- [ ] Δοκίμασε μεταβλητή ταχύτητα/κατεύθυνση βάδισης

**✅ Done όταν:** αντέχει μικρά-μέτρια pushes χωρίς να πέφτει, με αυτόματη διόρθωση βημάτων.

---

## Στάδιο 6 — Κεκλιμένο επίπεδο (το challenge)

- [ ] `scene_incline.xml`: ground plane με κλίση (ξεκίνα 3°, μετά 5°, 8°, …)
- [ ] **Friction cones στο frame της επιφάνειας** (κάθετος = surface normal, ΟΧΙ world-z)
- [ ] Εκτίμηση προσανατολισμού επιφάνειας/βάσης (στο sim: ground truth, interface σαν IMU)
- [ ] Προσαρμογή CoM lean ώστε CoM/ZMP εντός του (γερμένου) support polygon
- [ ] Test προτεραιότητα: πρώτα **στέκεται** σε κλίση → μετά **περπατά** σε κλίση
- [ ] Σταδιακή αύξηση κλίσης μέχρι το όριο· κατάγραψε το μέγιστο

**✅ Done όταν:** στέκεται και περπατά σε ήπια κλίση χωρίς ολίσθηση/πτώση.

---

## Στάδιο 7 — Παραδοτέα

- [ ] Demo video: flat + push recovery + κλίση
- [ ] README: αρχιτεκτονική, εξισώσεις (Lecture 9), πώς τρέχει, παράμετροι
- [ ] Μετρικές: distance, duration, μέγιστη κλίση, ανάκαμψη από push
- [ ] Καθάρισμα repo, σχόλια, `params.yaml` συγυρισμένο
- [ ] Commit ανά στάδιο που πέρασε το κριτήριο

---

## Συνεχείς κανόνες (ισχύουν σε όλα τα στάδια)

- [ ] Όλες οι παράμετροι σε `config/params.yaml`, ΟΧΙ hardcoded
- [ ] Όλη η balance λογική εκφρασμένη **ως προς τη βαρύτητα**, όχι world-z (γενίκευση σε κλίση)
- [ ] Το QP solve να μένει γρήγορο (~1 kHz)· κράτα το πρόβλημα μικρό
- [ ] Πάντα safe fallback αν το QP γίνει infeasible
- [ ] Commit + σύντομο plot review μετά από κάθε στάδιο

## Αναφορές
- Lecture 9 (WBC via QP) + Lecture 8 (gravity compensation / biped balancing)
- Kuindersma et al. 2016 — QP whole-body control του Atlas (contacts + friction cones)
- TSID (Del Prete) — slides & video lessons για inverse-dynamics control
- Caron (scaron.info) — WBC & walking notes
