# V7 Round 4 Progress

Date: 2026-06-10

Round 4 goal: strengthen clinical evidence first, then test data-level optimization candidates under the existing validation-first / locked-test-only protocol.

## 1. Final Alert Policy Bootstrap CI

Implemented:

- `bootstrap_alert_policy_ci.py`
- patient/record-level bootstrap resampling
- policy file input: `results_v7_round3/final_alert_policy_selected.csv`
- outputs: `results_v7_round4/alert_policy_bootstrap_ci_final/`

Locked-test observed results with 95% patient bootstrap CI:

| Task | Metric | Observed | 95% CI |
|---|---|---:|---:|
| future_arrhythmia | event recall | 0.900 | 0.743-1.000 |
| future_arrhythmia | incident recall | 0.862 | 0.649-1.000 |
| future_arrhythmia | median lead | 225s | -10s-300s |
| future_arrhythmia | false episodes/hr | 1.296 | 0.302-2.717 |
| future_vt_vf | event recall | 0.600 | 0.222-1.000 |
| future_vt_vf | incident recall | 0.600 | 0.222-1.000 |
| future_vt_vf | false episodes/hr | 1.382 | 0.847-2.093 |

Interpretation:

- The arrhythmia policy remains the main competition-facing result.
- CI is wide because locked test has 21 patient/record units.
- VT/VF CI is especially wide because locked test has only 10 VT/VF events. Keep VT/VF claims conservative.

## 2. Horizon Dataset/Evaluation Plumbing

Implemented:

- `build_dataset_factory.py --predict-sec`
- dataset-level `dataset_config.json`
- `final_validation_analysis.py` reads `dataset_config.json`
- `final_validation_analysis.py --max-records-per-split` for smoke datasets
- `evaluate_checkpoint_ensemble.py` also reads horizon config

Smoke test:

- dataset: `results_v7_round4/horizon_smoke_dataset_180/`
- split: first validation record only
- horizon: 180 seconds
- result: 129 samples
- metadata check: `future_end_sec - future_start_sec == 180`
- evaluation log confirmed: `dataset predict_sec=180; event horizon=180s`

Decision:

- The horizon experiment pipeline is now technically ready.
- Do not yet build full 1/3/10 minute datasets until deciding the exact experiment matrix, because each full V7-style dataset is multi-GB.

## 3. Future-Event-Balanced Sampler Candidate

Implemented:

- `train_trajectory.py --sampler-mode current|future_event`
- default remains `current`, so V7 baseline behavior is unchanged
- sampler audit prints mean/p50/p95/p99/max and event-group means

Sampler audit on V7 train:

| Mode | future normal mean | future arrhythmia mean | future high-risk mean | transition mean | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.510 | 1.491 | 4.158 | 2.765 | 8.334 | 10.734 |
| future_event | 0.181 | 1.370 | 4.684 | 2.550 | 12.000 | 12.000 |

The first version was too aggressive and was adjusted downward before training.

### 12-Epoch Validation Result

Candidate:

- output: `models_v7_round4_future_event_sampler_12ep/seeds/seed0/`
- seed: 0
- epochs: 12
- encoder: frozen
- selection: V6 composite

| Model | Best epoch | Val composite | future macro-F1 | rare recall | future AUROC | ECE | transition recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7 stage1 baseline | 12 | 0.3630 | 0.2886 | 0.3711 | 0.7254 | 0.1444 | 0.2350 |
| future-event sampler | 10 | 0.3483 | 0.2725 | 0.2846 | 0.6914 | 0.1274 | 0.2710 |

Decision:

- Do not promote the sampler model.
- It improves transition recall, but loses the main validation endpoints: future macro-F1, rare recall, and AUROC.
- Do not run locked test for this candidate.

## Round 4 Current Decision

Accepted:

- patient-level bootstrap CI for final alert policy
- horizon-configurable dataset/evaluation plumbing

Rejected / not promoted:

- future-event sampler as a model replacement

Still recommended:

- Build full 3-minute and 10-minute horizon datasets only if we are ready to run a focused comparison.
- Next high-value path is competition-material preparation plus a small, carefully bounded full-horizon study.

## Recommended Next Step

Run a focused full-horizon comparison rather than many open-ended model tweaks:

1. Build full `dataset_v7_h180` for 3-minute warning.
2. Train seed0 for 12 epochs with the original baseline settings.
3. Evaluate validation only.
4. Compare against the 5-minute baseline on the same metrics.
5. If 3-minute validation is clearly stronger or more clinically coherent, then decide whether to build/test locked-test outputs.

Avoid another round of loss/sampler tuning unless the horizon study reveals a clear weakness.
