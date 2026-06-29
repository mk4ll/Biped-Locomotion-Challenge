# Talos Walking — Evaluation Report

Automated battery (`scripts/talos/evaluate.py`). 94 kg, box feet, 32 torque motors. Same DCM + whole-body-QP stack as the G1.

## Walking & terrain

| scenario | fell | duration | distance | max_tilt | dcm_rms | comz_err_rms | com_accel_rms | peak_tau |
|---|---|---|---|---|---|---|---|---|
| walk_flat | False | 11.6 | 1.435 | 1.6 | 0.0071 | 0.0022 | 0.812 | 159.7 |
| walk_incline_4deg | False | 11.6 | 1.143 | 1.4 | 0.0064 | 0.0027 | 0.72 | 151.5 |
| walk_incline_8deg | False | 11.6 | 1.14 | 1.6 | 0.0078 | 0.0046 | 0.757 | 143.2 |
| ctrl_baseline_flat | False | 11.6 | 1.435 | 1.6 | 0.0071 | 0.0022 | 0.812 | 159.7 |
| ctrl_mpc_flat | False | 11.6 | 1.433 | 1.6 | 0.0057 | 0.0022 | 0.795 | 161.5 |

## Push robustness (survived / trials)

- **push_sagittal_baseline**: 2/2 — by magnitude (N): {60: '1/1', 90: '1/1'}
- **push_sagittal_mpc**: 2/2 — by magnitude (N): {60: '1/1', 90: '1/1'}
- **push_lateral_baseline**: 2/2 — by magnitude (N): {60: '1/1', 90: '1/1'}
- **push_lateral_mpc**: 2/2 — by magnitude (N): {60: '1/1', 90: '1/1'}
