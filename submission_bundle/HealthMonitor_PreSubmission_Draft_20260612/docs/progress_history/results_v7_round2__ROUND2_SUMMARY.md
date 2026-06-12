# V7 Round 2 Optimization Summary

Date: 2026-06-09

## Baseline To Beat

Current deployable baseline remains:

- checkpoint: `models_v7_stage1/seeds/seed0/arrhythmia_warning_best.pth`
- selected epoch: 12
- validation future macro-F1: 0.2886
- validation rare recall: 0.3711
- validation future macro-AUROC: 0.7254
- locked test future macro-F1: 0.2978
- locked test future macro-AUROC: 0.7256

The second-round rule was: use validation data to compare candidates; only promote a model to locked test evaluation if it improves the main validation metrics without breaking calibration/risk behavior.

## Training-Side Candidates

Summary CSV: `results_v7_round2/round2_candidate_summary.csv`

| Candidate | Change | Best epoch | Val future macro-F1 | Val rare recall | Val future AUROC | Decision |
|---|---|---:|---:|---:|---:|---|
| stage1_baseline | frozen PTB-XL encoder, original V6 selection | 12 | 0.2886 | 0.3711 | 0.7254 | keep |
| risk_w010_hw10_risksel | risk BCE, risk-aware selection | 9 | 0.2459 | 0.3230 | 0.7376 | reject |
| risk_w003_hw05_v6sel | lighter risk BCE, original selection | 12 | 0.2653 | 0.2999 | 0.7373 | reject |
| soft_future_weights | future head weighted by Y_fut soft mass | 12 | 0.2480 | 0.3221 | 0.7350 | reject |
| encoder_tail_tune | fine-tune encoder stage4 + pool projection | 7 | 0.2204 | 0.2275 | 0.7531 | reject |

## Interpretation

The training-side changes improved some secondary ranking metrics in isolated epochs, especially AUROC, but all four candidates reduced future macro-F1 and/or rare recall compared with the baseline. None should replace the stage1 model.

Risk-aware BCE was not a good main optimization direction in this setting. It tended to reward binary risk separation early while weakening the 6-class future tendency task.

Soft future weighting was more aligned with the current soft-distribution supervision, but still did not improve the primary validation endpoint.

Encoder tail fine-tuning increased model flexibility but caused drift on high-risk behavior and reduced future macro-F1. Keeping the PTB-XL encoder frozen is currently safer.

## Alert Policy Optimization

The useful second-round improvement came from deployment-style postprocessing, not retraining.

Validation policy search was expanded from `k=1..5` to `k=1..12`, using the existing stage1 validation outputs and the same validation-selected operating points.

New validation-selected policies:

| Task | Threshold source | Threshold | Consecutive k | Val event recall | Val false episodes/hr |
|---|---|---:|---:|---:|---:|
| future_arrhythmia | temp_scaled sens_ge_0.90 | 0.36 | 2 | 0.895 | 2.25 |
| future_vt_vf | temp_scaled sens_ge_0.90 | 0.04 | 12 | 0.857 | 2.25 |

Locked test application:

| Task | Consecutive k | Test event recall | Test incident recall | Median lead | False episodes/hr |
|---|---:|---:|---:|---:|---:|
| future_arrhythmia | 2 | 0.875 | 0.828 | 220s | 1.56 |
| future_vt_vf | 12 | 0.600 | 0.600 | 300s | 1.38 |

Compared with the earlier VT/VF policy (`k=5`), the new locked test VT/VF false alert burden decreased from 1.90 to 1.38 episodes/hr while keeping event recall at 0.600.

## Current Decision

Do not promote any round2 training checkpoint to the main model.

Keep `models_v7_stage1/seeds/seed0/arrhythmia_warning_best.pth` as the current model baseline.

Promote the expanded alert postprocessing policy as the only accepted round2 improvement:

- arrhythmia: threshold 0.36, consecutive k=2
- VT/VF: threshold 0.04, consecutive k=12

## Recommended Next Round

Next training work should avoid more single-seed loss tinkering. The next meaningful candidates should be tested with short multi-seed evidence or a clear data-level change:

- improve data labeling/window construction for VT/VF future targets
- add event-balanced sampling based on future label mass, not only current class
- test patient-level sequence smoothing as a validation-selected postprocessor
- run multi-seed confirmation only after a candidate beats stage1 on validation future macro-F1 and rare recall
