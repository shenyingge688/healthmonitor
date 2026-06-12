# V7 Round 5 Progress

Date: 2026-06-10

## 1. 3-Minute Horizon Study

Status: completed on validation only. Locked test was not used.

Dataset:

- directory: `dataset_v7_h180`
- horizon: 180 s
- train trajectories: 13242
- validation trajectories: 2404
- audit: `results_v7_round5/dataset_v7_h180_audit/`
- reference h300 audit: `results_v7_round5/dataset_v7_h300_reference_audit/`

Training:

- output: `models_v7_round5_h180_stage1/seeds/seed0/`
- seed: 0
- epochs: 12
- baseline settings: current sampler, V6 selection, current-label future weights, frozen encoder

Best validation checkpoint:

- epoch: 9
- future macro-F1: 0.2421
- future macro-AUROC: 0.7518
- rare recall: 0.2440
- ECE: 0.1213
- transition recall: 0.2146
- composite: 0.3468

Reference h300 seed0 validation checkpoint:

- epoch: 12
- future macro-F1: 0.2886
- future macro-AUROC: 0.7254
- rare recall: 0.3711
- ECE: 0.1444
- transition recall: 0.2350
- composite: 0.3630

Interpretation:

- h180 improves ranking signal: validation AUROC is higher than h300 seed0.
- h180 hurts thresholded classification and rare-class recall.
- h180 does not improve the current deployment-facing alert policy.

Smoothed alert validation, h180:

- output: `results_v7_round5/h180_seed0_smoothed_alert_val_narrow_h180fixed/`
- script fix: `evaluate_smoothed_alert_policy.py` now infers `event_horizon_sec` from `future_end_sec - future_start_sec`
- selected arrhythmia policy: uncalibrated, threshold 0.20, EWMA alpha 0.80, k=2
- event recall: 0.869
- incident event recall: 0.857
- median lead: 180 s
- false alert episodes/hr: 2.27

Reference h300 validation smoothed alert:

- selected arrhythmia policy: temp-scaled, threshold 0.36, rolling mean window 2, k=2
- event recall: 0.895
- incident event recall: 0.889
- median lead: 260 s
- false alert episodes/hr: 1.81

Decision:

- Do not promote h180 as the main model.
- Do not evaluate h180 on locked test.
- Keep h180 as evidence that shorter-horizon labels can improve AUROC but currently reduce deployable alert quality.

## 2. Next Step

Return to the current h300 core and run seeds 1-4 under the same baseline configuration as seed0.

Reason:

- Current h300 seed0 remains stronger on the main validation endpoints and alert-policy endpoints.
- Multiseed stability is a high-value competition-facing evidence upgrade.

## 2. h300 Multiseed Stability

Status: completed for seeds 0-4.

Training:

- seed0: existing h300 baseline checkpoint
- seeds1-4: `models_v7_stage1/seeds/seed*/`
- aggregate: `models_v7_stage1/multiseed_summary.json`
- settings: h300 `dataset_v7`, current sampler, V6 selection, current-label future weights, frozen encoder

Best-composite validation metrics across 5 seeds:

- future macro-F1: 0.278 +/- 0.014
- future macro-AUROC: 0.731 +/- 0.013
- rare recall: 0.330 +/- 0.034
- transition recall: 0.238 +/- 0.014
- ECE: 0.166 +/- 0.033

Interpretation:

- h300 behavior is reproducible across seeds.
- The strongest single seed by composite score is seed1.
- Final-epoch checkpoints remain weaker than validation-selected checkpoints.
- Seed4 has useful rare/high-risk ranking signal, but weaker calibration and weaker standalone alert behavior.

## 3. Single-Seed Full Validation

Validation-only full evaluations:

- seed1: `results_v7_round5/seed1_best_val_full/`
- seed4: `results_v7_round5/seed4_best_val_full/`

Key findings:

- seed1 best checkpoint: future macro-F1 0.299, macro-AUROC 0.733, temp-scaled ECE 0.026.
- seed4 best checkpoint: future macro-F1 0.262, macro-AUROC 0.753, temp-scaled ECE 0.059.
- seed1 improves classification, but selected low-burden alert policy drops validation event recall to 0.860.
- seed4 has stronger AUROC/high-risk ranking, but does not beat the accepted Round3 alert profile as a standalone model.

Decision:

- Do not promote a single seed by itself.
- Continue to seed ensembles.

## 4. Seed Ensemble Search

Validation-only ensemble evaluations:

- 5-best ensemble: `results_v7_round5/ensemble_5best_val_full/`
- seed0+seed1+seed4 ensemble: `results_v7_round5/ensemble_seed014_best_val_full/`

Best validation model candidate:

- candidate: logits average of seed0, seed1, and seed4 best-composite checkpoints
- future macro-F1: 0.302
- future macro-AUROC: 0.768
- future arrhythmia AUROC: 0.750
- future high-risk AUROC: 0.779
- temp-scaled ECE: 0.044

Selected validation alert policy for future_arrhythmia:

- output: `results_v7_round5/round5_seed014_arrhythmia_policy_selected.json`
- probability source: uncalibrated
- threshold: 0.10
- smoothing: EWMA alpha 0.65
- consecutive trigger: k=2
- validation event recall: 0.912
- validation incident recall: 0.911
- validation median lead: 290 s
- validation false windows/hr: 28.16
- validation false episodes/hr: 1.73

Reference Round3 validation arrhythmia policy:

- event recall: 0.895
- incident recall: 0.889
- median lead: 260 s
- false windows/hr: 28.42
- false episodes/hr: 1.81

Decision:

- Promote seed0+seed1+seed4 ensemble with the validation-selected arrhythmia policy to locked test.

## 5. Locked Test Result

Locked-test model evaluation:

- output: `results_v7_round5/ensemble_seed014_best_test_fixed_valcal/`
- calibration and operating points imported from validation.
- future macro-F1: 0.318
- future macro-AUROC: 0.747
- future macro-F1 patient bootstrap CI: 0.230-0.381
- future macro-AUROC patient bootstrap CI: 0.657-0.849
- future arrhythmia AUROC: 0.713
- future high-risk AUROC: 0.734

Locked-test arrhythmia alert policy:

- output: `results_v7_round5/ensemble_seed014_best_test_round5_policy/`
- policy: uncalibrated, threshold 0.10, EWMA alpha 0.65, k=2
- event recall: 0.925
- incident event recall: 0.897
- median lead: 290 s
- incident median lead: 300 s
- false windows/hr: 22.20
- false episodes/hr: 1.12

Patient bootstrap CI:

- output: `results_v7_round5/round5_seed014_alert_policy_bootstrap_ci/`
- locked-test event recall: 0.925, 95% CI 0.757-1.000
- locked-test incident event recall: 0.897, 95% CI 0.675-1.000
- locked-test median lead: 290 s, 95% CI -10-300 s
- locked-test incident median lead: 300 s, 95% CI 300-300 s
- locked-test false windows/hr: 22.20, 95% CI 6.65-44.07
- locked-test false episodes/hr: 1.12, 95% CI 0.30-2.26

Comparison with accepted Round3 locked-test arrhythmia policy:

- Round3 event recall: 0.900 -> Round5 0.925
- Round3 incident recall: 0.862 -> Round5 0.897
- Round3 median lead: 225 s -> Round5 290 s
- Round3 false episodes/hr: 1.30 -> Round5 1.12

Decision:

- Accept Round5 seed0+seed1+seed4 ensemble as the new default future-arrhythmia warning model/policy.
- Final policy file: `results_v7_round5/final_alert_policy_selected_round5.json`
- Keep VT/VF policy conservative and exploratory; Round5 does not establish reliable VT/VF warning.
