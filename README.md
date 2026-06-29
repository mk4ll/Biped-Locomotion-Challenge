# RS1 Project — Δυναμική Βάδιση Unitree G1 (MuJoCo, torque-level WBC)

Τρεις υλοποιήσεις του ίδιου project (ομάδα 2 ατόμων, παράλληλη δουλειά):

| Φάκελος | Τι είναι |
|---|---|
| [`marios_progress/`](marios_progress/) | Υλοποίηση Μάριου — καθαρό offline→online→WBC pipeline, stage-by-stage (Στάδια 0–6), θεωρητική ανάλυση ορίων κλίσης (slip/tip), hard 6D contact WBC. Flat walking, push recovery, standing-on-incline 26°, incline walking 3°. |
| [`kanellos_progress/`](kanellos_progress/) | Υλοποίηση Κανέλλου — feature-rich: terrain-aware incline (14°) + **stairs**, **omnidirectional**, **navigation** (obstacle avoidance), **DCM preview-MPC**, **2ο ρομπότ (Talos)**, evaluation harness + GIFs. |
| [`merged_progress/`](merged_progress/) | **Merge**: αρχιτεκτονική/λογική Μάριου + ΟΛΑ τα tasks του Κανέλλου. Incline **3°→16°**, **stairs full climb**, **omnidirectional** (fwd/back/strafe/curve), push recovery, αυτόματο eval — όλα PASS. |

## Πίνακας διαφορών (checklist)

| Δυνατότητα / Χαρακτηριστικό | marios | kanellos | merged |
|---|:---:|:---:|:---:|
| Pipeline offline→online→WBC | ✅ | ✅ | ✅ |
| WBC contact | hard 6D | soft | **hard 6D** |
| Gravity-comp sanity test | ✅ | ✅ | ✅ |
| Standing / weight-shift / single-support | ✅ | ✅ | ✅ |
| Flat walking | ✅ | ✅ | ✅ |
| Incline walking | ⚠️ 3° | ✅ 14° | ✅ **16°** |
| Standing on incline | ✅ 26° | ~15° | ✅ **26°** |
| Stairs | ❌ | ✅ | ✅ **πλήρης** |
| Omnidirectional (strafe/turn/curve) | ❌ | ✅ | ✅ |
| Push recovery | ✅ | ✅ | ✅ |
| Navigation (obstacle avoidance) | ❌ | ✅ | ✅ (slalom) |
| DCM preview-MPC | ❌ | ✅ | ✅ (`--mpc`) |
| 2ο ρομπότ (Talos) | ❌ | ✅ | ✅ ([r] toggle) |
| Evaluation harness | ❌ | ✅ +GIFs | ✅ |
| Θεωρητική ανάλυση (slip/tip) | ✅ | ❌ | ✅ |
| **Lecture-style plots** (path/footsteps/CoM height) | μερικά | ❌ | ✅ |
| Tray-pose χέρια (δίσκος) | ✅ | ❌ | ✅ (G1+Talos) |
| Επιλογή ταχύτητας (slow/normal/fast) | ❌ | ❌ | ✅ |
| 🍽 FUN: σερβιτόρος (navigation + φραπέ) | ❌ | ❌ | ✅ |
| 🪨 FUN: Σίσυφος (βράχος σε κλίση) | ❌ | ❌ | ✅ |
| Arm swing (contralateral coupling) | ❌ | ❌ | ✅ (`--arm-swing`) |

## Σύγκριση (+ / −)

**Κανέλλος (+):** terrain-aware από την αρχή → incline 14° + stairs· omnidirectional·
navigation· MPC· 2 ρομπότ· eval harness + GIFs.
**Κανέλλος (−):** lateral push ~40–50 N (όπως Μάριος)· λιγότερη θεωρητική τεκμηρίωση·
soft contact (όχι ακριβές)· κάποια modules prototype.

**Μάριος (+):** καθαρή stage-by-stage δομή χαρτογραφημένη στις διαλέξεις· **ανάλυση
φυσικής** (slip/tip, standing 26°=arctan μ)· hard 6D contact (ακριβές)· changelog/report.
**Μάριος (−):** λιγότερα features (incline μόνο 3°, όχι stairs/omni/MPC/navigation).

## Merged — αποτέλεσμα

Το `merged_progress/` παίρνει το **terrain-aware design** του Κανέλλου πάνω στο **pipeline +
WBC του Μάριου**, και πετυχαίνει: incline walking **16°**, **stairs** (πλήρης σκάλα),
**omnidirectional** βάδιση, push recovery — με αυτόματο eval (`scripts/evaluate.py`) όλα PASS.
Λεπτομέρειες & merge-map: [`merged_progress/README.md`](merged_progress/README.md).

Δεν portαρίστηκαν ακόμη (follow-up): DCM preview-MPC, navigation, Talos — υπάρχουν στου
Κανέλλου και είναι συμβατά με το merged pipeline.
