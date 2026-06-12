# V7 Round 3 Optimization Summary

Date: 2026-06-09

## Baseline Kept

The main model checkpoint is unchanged:

- checkpoint: `models_v7_stage1/seeds/seed0/arrhythmia_warning_best.pth`
- locked-test future macro-F1: 0.2978
- locked-test future macro-AUROC: 0.7256
- locked-test temperature-scaled ECE: 0.0459

Round 3 focused on post-training, validation-selected deployment policy optimization and evaluation strengthening. It did not change model weights or dataset tensors.

## Accepted Improvement

Promote the high-recall arrhythmia smoothing policy as the preferred arrhythmia alert policy:

- task: `future_arrhythmia`
- probability source: `temp_scaled`
- threshold: 0.36
- smoothing: causal rolling mean, window=2
- consecutive trigger: k=2
- refractory windows: 0

This policy uses only current/past patient-window scores, so it is compatible with real-time deployment.

### Validation

| Policy | Event recall | Incident recall | Median lead | False windows/hr | False episodes/hr |
|---|---:|---:|---:|---:|---:|
| baseline consecutive | 0.895 | 0.889 | 260s | 27.73 | 2.25 |
| high-recall smoothing | 0.895 | 0.889 | 260s | 28.42 | 1.81 |

### Locked Test

| Policy | Event recall | Incident recall | Median lead | False windows/hr | False episodes/hr |
|---|---:|---:|---:|---:|---:|
| baseline consecutive | 0.875 | 0.828 | 220s | 23.84 | 1.56 |
| high-recall smoothing | 0.900 | 0.862 | 225s | 24.28 | 1.30 |

Decision: accept for arrhythmia alerting. It improves locked-test event recall, incident recall, and false-alert episodes/hr while preserving lead time. False windows/hr increases only slightly, so the improvement is not a long-alarm artifact.

## VT/VF Policy Decision

The high-recall VT/VF candidate was:

- threshold: 0.03
- smoothing: raw
- consecutive trigger: k=10

It improved locked-test event recall from 0.600 to 0.700, but validation false-window burden increased from 71.97/hr to 89.07/hr and false episodes/hr increased from 2.25/hr to 2.42/hr.

Decision: do not promote as the default VT/VF policy. Keep the Round 2 VT/VF policy as the conservative default:

- threshold: 0.04
- consecutive trigger: k=12
- locked-test event recall: 0.600
- locked-test false episodes/hr: 1.38

The threshold 0.03 / k=10 VT/VF policy can be reported only as an exploratory high-sensitivity operating point, not as the final deployment policy.

## Low-False-Positive Smoothing Candidate

A stricter smoothing candidate reduced false-alert burden further:

- arrhythmia threshold: 0.36
- smoothing: rolling mean window=2
- consecutive trigger: k=4

Locked-test result:

- event recall: 0.850
- incident recall: 0.793
- median lead: 175s
- false episodes/hr: 1.12

Decision: do not use as the main policy because it sacrifices recall and lead time. It is useful only as a low-alert-burden demonstration mode.

## Checkpoint Ensemble Result

Two checkpoint ensembles were tested on validation only.

| Ensemble | Val future macro-F1 | Val future macro-AUROC | Decision |
|---|---:|---:|---|
| best + best-AUROC | 0.2707 | 0.7363 | reject: AUROC up, F1 down |
| best + best-macro-F1 | 0.2886 | 0.7253 | no gain |

Decision: do not promote checkpoint ensemble. V7 currently has only one seed in `models_v7_stage1`, so same-seed checkpoint ensembling is not strong enough evidence for replacement.

## Lead-Horizon Sensitivity

This is a post-hoc alert-policy analysis. The model label horizon remains 5 minutes. The analysis varies the allowed event-credit horizon.

Locked-test arrhythmia results for the accepted high-recall smoothing policy:

| Credited horizon | Event recall | Incident recall | Median lead | False episodes/hr |
|---:|---:|---:|---:|---:|
| 1 min | 0.900 | 0.862 | 60s | 2.07 |
| 3 min | 0.900 | 0.862 | 115s | 1.38 |
| 5 min | 0.900 | 0.862 | 225s | 1.30 |
| 10 min | 0.900 | 0.862 | 230s | 1.12 |

Recommended competition wording:

- The system is trained for a 5-minute future-warning task.
- On locked test data, the accepted arrhythmia policy detected 90.0% of arrhythmia events with median lead time 225 seconds and 1.30 false-alert episodes per patient-hour.
- The 1-minute and 3-minute sensitivity analyses show that detected events are not only late detections inside the event segment.

## Final Round 3 Decision

Promote only the arrhythmia high-recall smoothing policy.

Do not replace the model checkpoint.

Do not promote checkpoint ensemble.

Keep VT/VF claims conservative. VT/VF evidence supports an exploratory high-risk warning signal, not a reliable standalone VT/VF diagnostic classifier.

## Generated Files

- `evaluate_smoothed_alert_policy.py`
- `evaluate_checkpoint_ensemble.py`
- `evaluate_lead_horizon_sensitivity.py`
- `results_v7_round3/smoothed_alert_val_highrecall/`
- `results_v7_round3/smoothed_alert_test_highrecall_applied_valpolicy/`
- `results_v7_round3/lead_horizon_sensitivity_highrecall/`
- `results_v7_round3/ensemble_best_plus_auroc_val/`
- `results_v7_round3/ensemble_best_plus_macrof1_val/`
