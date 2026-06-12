# V7 Round 6 Progress

Date: 2026-06-10

## 1. Strategy-Layer Objective

Round6 tested post-training alert policies against the accepted Round5
seed0+seed1+seed4 ensemble. Model weights and dataset tensors were unchanged.

Predefined promotion criteria:

- event recall >= 0.90
- incident event recall >= 0.88
- median lead time >= 260 s
- false alert episodes/hr below the Round5 result
- no validation-selected parameter changes after locked-test application

Accepted Round5 locked-test anchor:

- event recall: 0.925
- incident event recall: 0.897
- median lead time: 290 s
- incident median lead time: 300 s
- false alert windows/hr: 22.20
- false alert episodes/hr: 1.12

## 2. Patient-Adaptive Baseline and Hysteresis

Implementation:

- script: `evaluate_adaptive_alert_policy.py`
- patient baselines use prior windows only
- adaptive conditions affect alert entry only
- exit thresholds use a causal state machine
- Round5 anchor reproduction was exact for every reported metric
- validation grid size: 19,536 policies

Validation-selected policy:

- risk: uncalibrated seed014 ensemble, EWMA alpha 0.65
- enter/exit threshold: 0.08 / 0.08
- consecutive k: 2
- baseline: rolling mean/std, 30 prior windows
- warmup: 10 output windows with global-threshold fallback
- adaptive entry: z-score >= 2.0

Validation result:

- event recall: 0.930
- incident event recall: 0.933
- median lead time: 300 s
- false alert windows/hr: 25.14
- false alert episodes/hr: 1.21

Fixed-policy locked-test result:

- event recall: 0.925
- incident event recall: 0.897
- median lead time: 210 s
- incident median lead time: 300 s
- false alert windows/hr: 19.87
- false alert episodes/hr: 1.04

Decision:

- do not promote
- recall was preserved and false-alert burden fell slightly
- overall median lead time missed the predefined 260 s requirement
- event diagnostics showed that 35 of 37 detected test events had unchanged
  lead time; one event changed from 300 s to 200 s and one from 10 s to 40 s
- this explains the unstable median but does not justify changing the
  promotion rule after seeing test results

## 3. AFib-Limited RR Gating

RR export and alignment:

- script: `attach_rr_features.py`
- nine saved RR features were attached from V7 shards
- row count, current labels, and future labels were checked exactly
- validation and test each contained 2,188 aligned rows

Gating scope:

- only AFib-dominant alert-entry candidates were eligible for suppression
- PVC, VT, VF, and AT/SVT dominant candidates were never RR-gated
- entropy was excluded because its validation direction was inconsistent
- VT/VF regularity gating was not attempted

Implementation:

- script: `evaluate_rr_gating.py`
- accepted Round5 threshold, EWMA, and k were fixed
- validation selection searched AFib share, normalized RMSSD, and RR-CV

Validation-selected policy:

- AFib probability share >= 0.65
- RR support: normalized RMSSD >= 3.5 or RR-CV >= 0.45

Validation result:

- event recall: 0.912
- incident event recall: 0.911
- median lead time: 270 s
- incident median lead time: 300 s
- false alert windows/hr: 26.18
- false alert episodes/hr: 1.56

Fixed-policy test application:

- event recall: 0.925
- incident event recall: 0.897
- median lead time: 210 s
- incident median lead time: 300 s
- false alert windows/hr: 16.41
- false alert episodes/hr: 0.86

Decision:

- do not promote as the default policy because median lead time missed 260 s
- retain only as an exploratory low-burden mode
- the test split had already been viewed for the adaptive-policy experiment,
  so this second strategy result has multiple-comparison risk and is not a new
  confirmatory locked-test claim

## 4. Final Strategy Decision

The accepted Round5 policy remains the deployment and competition default:

- uncalibrated seed0+seed1+seed4 logits ensemble
- threshold 0.10
- EWMA alpha 0.65
- consecutive k=2

Round6 strategy experiments show that false-alert burden can be reduced without
losing event recall, but the current small test event set makes the overall
median lead time sensitive to a single changed event. No Round6 strategy policy
meets every predefined promotion criterion.

Next work should move away from test-threshold iteration and continue with:

1. ensemble uncertainty output
2. ECG-level CAM explanation
3. validation-only model experiments, starting with conservative encoder-tail
   fine-tuning if additional discrimination improvement is still required

## 5. Ensemble Uncertainty and Serving Alignment

Status: implemented and regression-tested.

Serving correction:

- the API previously loaded the legacy single checkpoint under `models/`
- `main.py` now loads the accepted Round5 seed0/seed1/seed4 checkpoints
- future and current probabilities use logits averaging, matching the formal
  `evaluate_checkpoint_ensemble.py` protocol
- missing or incompatible checkpoints now fail fast instead of silently serving
  random model weights

Validation-derived disagreement configuration:

- script: `calibrate_ensemble_uncertainty.py`
- config: `results_v7_round6/ensemble_uncertainty_config.json`
- member risk: `1 - member future-normal probability`
- high-confidence maximum risk std: 0.0895, with unanimous 3/3 class vote
- medium-confidence maximum risk std: 0.2204, with at least 2/3 class vote
- all other outputs are low confidence
- validation distribution: 39.5% high, 47.5% medium, 12.9% low

API additions:

- formal logits-ensemble risk score
- member risk mean
- member risk standard deviation
- per-class probability standard deviation
- individual member risks
- future class vote count
- high/medium/low agreement label
- explicit warning that agreement is not a clinical confidence interval

Regression verification:

- script: `test_ensemble_serving.py`
- formal batch-size-64 ensemble probability max error: 1.23e-7
- member-risk max error: 4.32e-8
- real-record direct API latency on GPU: 0.16-0.54 s for three models
- tested records: MIT-BIH 100, 210, and 223
- Streamlit was opened in the local browser and the uncertainty card was
  verified during live replay

## 6. ECG-Level Attribution

Status: implemented as an offline/on-demand explanation artifact.

Implementation:

- script: `eval_ecg_level_attribution.py`
- level 1: class-conditional Grad-CAM across the 39 history windows
- level 2: integrated gradients inside the selected 30-second ECG window
- ensemble: mean explanation across the accepted seed0/seed1/seed4 models
- RR inputs now match serving preprocessing instead of zero-filling trajectory
  features
- PVC explanations are checked against annotated V/E beat neighborhoods

Held-out MIT-BIH 201 PVC example:

- target probability: 0.177
- selected window: 38, covering 630-660 s
- annotated V/E beats in selected window: 6
- mean window CAM on annotated windows / background: 6.97x
- ECG attribution near V/E beats / background: 1.97x
- beat-neighborhood average precision: 0.178, versus neighborhood prevalence
  of approximately 0.081
- 8-step and 12-step integrated-gradient ratios were effectively unchanged

MIT-BIH 119 PVC example:

- target probability: 0.839
- selected window: 27, covering 465-495 s
- annotated V/E beats in selected window: 5
- mean window CAM on annotated windows / background: 2.66x
- ECG attribution near V/E beats / background: 5.32x
- beat-neighborhood average precision: 0.415

Artifacts:

- `results_v7_round6/ecg_level_attribution/mitdb_201_class1_m11.png`
- `results_v7_round6/ecg_level_attribution/mitdb_119_class1_m11.png`
- matching per-case JSON metrics
- `results_v7_round6/ecg_level_attribution/summary.json`

Deployment decision:

- keep the low-cost 39-window attention strip in the 1 Hz live dashboard
- use ECG-level integrated gradients for saved alerts, demonstrations, and
  clinician review because it requires repeated backward passes
