# G1 Walking — Evaluation Report

Automated battery (`scripts/evaluate.py`). Torque whole-body QP control.

## Walking & terrain

| scenario | fell | duration | distance | max_tilt | dcm_rms | comz_err_rms | com_accel_rms | peak_tau |
|---|---|---|---|---|---|---|---|---|
| walk_flat | False | 13.4 | 1.414 | 3.7 | 0.0229 | 0.0029 | 1.136 | 54.7 |
| walk_incline_4deg | False | 13.4 | 1.072 | 3.3 | 0.0233 | 0.0038 | 1.092 | 52.1 |
| walk_incline_8deg | False | 13.4 | 0.992 | 3.3 | 0.0248 | 0.0052 | 1.26 | 55.1 |
| walk_incline_12deg | False | 13.4 | 0.941 | 3.5 | 0.0265 | 0.0088 | 1.739 | 74.6 |
| march_in_place | False | 9.8 | -0.018 | 2.3 | 0.0114 | 0.0017 | 0.772 | 41.3 |
| stairs_4cm | True | 4.32 | 0.433 | 50.1 | 0.233 | 0.0413 | 2.953 | 88.0 |
| low_friction_0.6 | False | 13.4 | 1.441 | 3.9 | 0.0198 | 0.0021 | 1.443 | 65.3 |
| payload_+2kg | True | 12.89 | 0.87 | 50.5 | 0.1111 | 0.0127 | 1.466 | 88.0 |

## Push robustness (survived / trials)

- **push_sagittal_baseline**: 3/3  — by magnitude (N): {40: '1/1', 50: '1/1', 60: '1/1'}
- **push_sagittal_reactive**: 3/3  — by magnitude (N): {40: '1/1', 50: '1/1', 60: '1/1'}
- **push_lateral_baseline**: 1/3  — by magnitude (N): {40: '1/1', 50: '0/1', 60: '0/1'}
- **push_lateral_reactive**: 1/3  — by magnitude (N): {40: '1/1', 50: '0/1', 60: '0/1'}

_Metrics: distance/duration before fall; max_tilt (deg); dcm_rms (DCM tracking, m); comz_err_rms (CoM-height tracking, m); com_accel_rms (gait smoothness); peak_tau (Nm)._

### Notes / known limits
- Reactive (capture-point) stepping targets *forward* walking; the in-place march and stairs use the base gait.
- Stairs (4 cm risers) climb several steps then tip — full robust stair-climbing needs tread-center footstep planning.
- Payload tolerance ≈ +2 kg (6% of body mass) before the nominal gains go marginal; lateral push tolerance ~40–50 N (no step-timing adaptation yet), sagittal ≥60 N.
