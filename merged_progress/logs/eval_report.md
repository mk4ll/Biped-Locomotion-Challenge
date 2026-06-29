# Merged G1 — Evaluation Report

Automated battery (`scripts/evaluate.py`). Torque whole-body QP, DCM planning.

## Walking & terrain

| scenario | fell | dist [m] | rise [mm] | max_tilt [deg] | dcm_rms [m] | peak_tau [Nm] |
|---|---|---|---|---|---|---|
| walk_flat | False | +0.90 | -2 | 3.6 | 0.033 | 53 |
| walk_incline_8 | False | +0.89 | +126 | 3.4 | 0.034 | 53 |
| walk_incline_12 | False | +0.89 | +191 | 3.3 | 0.034 | 53 |
| walk_incline_16 | False | +0.88 | +255 | 3.9 | 0.029 | 55 |
| stairs_6x2.5cm | False | +1.47 | +149 | 4.4 | 0.029 | 62 |
| omni_forward | False | +0.68 | -2 | 3.5 | 0.031 | 53 |
| omni_strafe | False | -0.07 | +0 | 3.8 | 0.025 | 44 |
| omni_curve | False | +0.48 | -3 | 4.3 | 0.028 | 67 |

## Push robustness (mid-walk shove, 100 ms)

| scenario | survived | dist [m] | max_tilt [deg] |
|---|---|---|---|
| push_lateral_50N | True | +0.90 | 3.7 |
| push_sagittal_100N | True | +0.90 | 5.0 |

_Metrics: dist/rise of CoM; max_tilt = pelvis tilt from vertical;
dcm_rms = DCM tracking error; peak_tau = max joint torque._