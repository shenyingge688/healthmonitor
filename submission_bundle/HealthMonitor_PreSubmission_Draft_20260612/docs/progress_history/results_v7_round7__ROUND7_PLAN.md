# V7 Round 7: Six-Day Optimization Plan

Plan date: 2026-06-10

Execution window: 2026-06-11 to 2026-06-16

## 0. Execution Status

Status updated: 2026-06-11

Completed:

- added `evaluate_afib_directional_policy.py`
- audited exact row/label alignment across the accepted ensemble output, all
  three member outputs, and the RR-enriched validation table
- evaluated dominant future AFib, any future AFib mass, current AFib events,
  false alert burden, patient bootstrap intervals, error buckets, and
  positive-record leave-one-out influence
- repeated the complete analysis with identical SHA-256 hashes for all 11
  generated result files
- updated and structurally verified
  `C:\HealthMonitor\HealthMonitor_ECG_Arrhythmia_Competition_Report.docx`;
  the report preserves the locked-test results and labels the AFib rule as an
  external-validation-only candidate

Validation support:

- 2,188 windows from 21 patient/record units
- 349 dominant future AFib windows from 5 records
- 370 current AFib windows and 15 current AFib events from 5 records

Baseline future-class argmax:

- dominant AFib precision: 0.251
- dominant AFib recall: 0.160
- dominant AFib F1: 0.196
- AFib event recall: 0.400
- median lead time: -650 s
- false AFib alert episodes/hr: 4.15

Selected validation candidate:

- score:
  `P(AFib) / max(P(AFib) + P(Normal), 1e-8)`
- smoothing: EWMA alpha 0.65
- threshold: 0.10
- consecutive trigger: k=1
- dominant AFib precision: 0.322
- dominant AFib recall: 0.653
- dominant AFib F1: 0.431
- AFib event recall: 0.733
- median lead time: 280 s
- false AFib alert episodes/hr: 3.71
- positive-record leave-one-out gate pass: 4/5

Decision:

- status: `validation_candidate`
- scope: `external_evaluation_only`
- the accepted Round5 overall arrhythmia alarm remains unchanged
- no test-set threshold selection or rescue analysis was performed
- bootstrap intervals remain wide because only five AFib-positive validation
  records are available
- pause RR-based complex gating and the AFib meta-head; proceed to the
  pre-specified external LTAFDB pilot before any deployment or report claim

Artifacts:

- `results_v7_round7/afib_baseline/observed_metrics.json`
- `results_v7_round7/afib_baseline/threshold_sweep.csv`
- `results_v7_round7/afib_baseline/candidate_stability.csv`
- `results_v7_round7/afib_baseline/patient_bootstrap_ci.csv`
- `results_v7_round7/afib_baseline/leave_one_positive_record_out.csv`
- `results_v7_round7/afib_baseline/error_buckets.csv`
- `results_v7_round7/afib_baseline/decision.json`
- `results_v7_round7/afib_baseline/AFIB_BASELINE.md`

External pilot preparation completed:

- audited all 84 LTAFDB headers and rhythm annotations without downloading
  signal files
- confirmed 128 Hz and two signals for every record
- identified 75 annotation-eligible records and 110 strict AF prediction
  onsets
- selected 12 pilot records before inference, with four records in each
  low/medium/high AF-burden stratum
- confirmed the current V7 sources are AFDB, MIT-BIH Arrhythmia Database,
  SVDB, and VFDB; LTAFDB is a distinct external source
- repeated the cached metadata audit with identical hashes for all six output
  files

Selected pilot records:

- low burden: `120`, `32`, `100`, `113`
- medium burden: `101`, `121`, `28`, `117`
- high burden: `103`, `10`, `01`, `74`

External preparation artifacts:

- `prepare_ltafdb_external.py`
- `results_v7_round7/ltafdb_pilot_audit/audit_summary.json`
- `results_v7_round7/ltafdb_pilot_audit/pilot_manifest.csv`
- `results_v7_round7/ltafdb_pilot_audit/LTAFDB_PILOT_AUDIT.md`

The metadata audit contains no model inference or external performance claim.
Signal download and preprocessing must pass an independent integrity and
normalization audit before the frozen models are run.

External pilot signal audit completed:

- downloaded exactly the 12 preselected signal files (487.7 MiB)
- verified every file against the header-derived byte count
- verified both WFDB checksums and both initial samples for every record
- passed 574/576 deterministic 30-second preprocessing probes
- retained lead 0 for all 12 records under the frozen quality rule; no
  model-output-based lead selection was performed
- identified one isolated near-constant dual-lead interval in record `113`;
  the primary external analysis will retain all time points and a secondary
  analysis will report signal-quality-valid histories
- repeated the complete cached audit with identical SHA-256 hashes for all
  seven result files

Signal audit artifacts:

- `download_audit_ltafdb_signals.py`
- `results_v7_round7/ltafdb_signal_audit/signal_audit_summary.json`
- `results_v7_round7/ltafdb_signal_audit/signal_file_inventory.csv`
- `results_v7_round7/ltafdb_signal_audit/lead_quality.csv`
- `results_v7_round7/ltafdb_signal_audit/preprocessing_probes.csv`
- `results_v7_round7/ltafdb_signal_audit/selected_leads.csv`
- `results_v7_round7/ltafdb_signal_audit/LTAFDB_SIGNAL_AUDIT.md`

No model inference or external threshold selection was performed during the
signal audit.

Frozen LTAFDB pilot inference and evaluation completed:

- protocol SHA-256:
  `366a3ade4a02ea076b2f72bc5a3a7190852591d0c2e6b44920f8358de84263e6`
- 12 records, 98,808 ten-second evaluation rows, 98,610
  signal-quality-valid histories
- all 36 cached-versus-direct forward checks retained identical current and
  future argmax decisions
- maximum probability difference was 0.000976 and maximum AFib-vs-Normal
  score difference was 0.000317
- no external threshold tuning, model selection, record selection, or lead
  selection used model performance

External six-class argmax baseline:

- dominant future AFib AUROC: 0.679
- dominant future AFib recall: 0.115
- AFib event recall: 0.206
- median lead time: 210 s
- false AFib alert episodes/hr: 2.656

Frozen AFib directional candidate:

- dominant future AFib AUROC: 0.719
- dominant future AFib AP: 0.552
- dominant future AFib precision: 0.579
- dominant future AFib recall: 0.478
- AFib event recall: 0.545
- incident AFib event recall: 0.540
- median lead time: 260 s
- false AFib alert episodes/hr: 2.897
- patient-bootstrap 95% interval for AUROC: 0.597-0.815
- patient-bootstrap 95% interval for event recall: 0.345-0.750

External decision:

- status: `preliminary_external_support`
- all four pre-specified support checks passed
- false alert episode burden increased by 9.1%, within the 20% gate
- per-record behavior remains heterogeneous; record `103` has AUROC 0.114
- this is a 12-record pilot, not a clinical claim or proof of complete
  cross-database generalization
- the accepted Round5 overall alarm remains unchanged

External result artifacts:

- `run_ltafdb_external_inference.py`
- `evaluate_ltafdb_external.py`
- `results_v7_round7/ltafdb_external_inference/external_evaluation_protocol.json`
- `results_v7_round7/ltafdb_external_inference/inference_summary.json`
- `results_v7_round7/ltafdb_external_evaluation/observed_metrics.json`
- `results_v7_round7/ltafdb_external_evaluation/patient_bootstrap_ci.csv`
- `results_v7_round7/ltafdb_external_evaluation/per_record_metrics.csv`
- `results_v7_round7/ltafdb_external_evaluation/decision.json`

The competition report was updated with the external pilot result and its
limitations. Standard DOCX PNG rendering could not be run because LibreOffice
is not installed; the final document passed structural checks.

## 1. Fixed Baseline

The accepted deployment and competition baseline remains unchanged:

- model: seed0 + seed1 + seed4 best-composite logits ensemble
- future-arrhythmia threshold: 0.10
- smoothing: EWMA alpha 0.65
- consecutive trigger: k=2
- locked-test event recall: 0.925
- locked-test incident recall: 0.897
- locked-test median lead time: 290 s
- locked-test false alert episodes/hr: 1.12

Round7 must not re-run already rejected directions as if they were new:

- encoder-tail fine-tuning as previously configured
- future-event sampler as a model replacement
- 180-second horizon as the main deployment model
- unrestricted test-set threshold iteration

## 2. Evaluation Discipline

1. All feature, model, and threshold selection uses train/validation data only.
2. The current test split is retained as a historical descriptive benchmark.
3. A Round7 candidate is not promoted based on another look at the current test split.
4. Confirmatory evidence must come from a new untouched external holdout or a newly sealed internal holdout.
5. AFib-specific output is an auxiliary directional warning and must not suppress the accepted overall arrhythmia warning.
6. VT/VF/AT-SVT remain exploratory unless new event-level evidence is obtained.
7. Every completed experiment must update:
   - this plan/status file
   - the relevant result JSON/CSV
   - `HealthMonitor_ECG_Arrhythmia_Competition_Report.docx` when conclusions change

## 3. Day-by-Day Schedule

### Day 1: Thursday, 2026-06-11

Objective: establish a reproducible AFib-specific baseline and identify why recall is low.

Implementation:

- add `evaluate_classwise_alert_policy.py`
- use the accepted seed0/seed1/seed4 validation outputs
- evaluate two AFib labels:
  - dominant future AFib class
  - any future AFib mass greater than zero
- report:
  - AUROC and AP
  - thresholded recall and precision
  - AFib event recall
  - AFib median lead time
  - false AFib alert episodes/hr
  - patient/record-level bootstrap intervals
- split AFib false negatives by:
  - AFib future mass and episode duration
  - RR irregularity
  - ensemble disagreement
  - transition versus already-established AFib

Deliverables:

- `results_v7_round7/afib_baseline/observed_metrics.json`
- `results_v7_round7/afib_baseline/threshold_sweep.csv`
- `results_v7_round7/afib_baseline/error_buckets.csv`
- `results_v7_round7/afib_baseline/AFIB_BASELINE.md`

Completion gate:

- baseline metrics reproduce deterministically
- row and label alignment checks pass exactly
- AFib event and patient/record support are explicitly counted

Stop condition:

- if AFib-positive validation support is concentrated in too few patient/record units for meaningful selection, do not tune a complex strategy; move directly to external-data preparation.

### Day 2: Friday, 2026-06-12

Objective: improve AFib directional recall without changing the main arrhythmia alarm.

Validation-only candidates:

1. AFib class-specific probability threshold.
2. AFib share:
   `afib_probability / max(arrhythmia_probability, epsilon)`.
3. RR-supported AFib confidence using normalized RMSSD, RR-CV, SDNN, pNN50,
   Poincare features, and entropy only when their validation direction is stable.
4. High/medium/low ensemble-agreement triage.
5. Entry/exit hysteresis for the AFib directional tag.

The strategy should add one of these labels:

- `AFib risk direction`
- `AFib risk direction, low agreement; review suggested`

It must not remove an overall arrhythmia warning.

Promotion gate relative to the Day 1 validation anchor:

- AFib event recall improves by at least 0.05 absolute, or thresholded AFib
  recall improves by at least 0.08 absolute
- AFib false alert episodes/hr increases by no more than 20%
- AFib median lead time is at least 240 s
- overall Round5 alert outputs are unchanged
- the selected rule is simple enough to explain in the report

Stop condition:

- if no candidate meets all gates, keep current output and record the negative result.
- do not inspect the current test split to rescue a failed validation result.

Deliverables:

- `evaluate_afib_directional_policy.py`
- `results_v7_round7/afib_policy_validation/`
- one selected-policy JSON or an explicit `not_promoted` decision JSON

### Day 3: Saturday, 2026-06-13

Objective: run one bounded AFib model experiment only if strategy-layer work is insufficient.

Candidate:

- a lightweight regularized AFib meta-head trained on the train split
- frozen inputs:
  - ensemble future logits/probabilities
  - current-head probabilities
  - ensemble disagreement
  - the nine saved RR features
- target:
  - future AFib occurrence from the existing future soft label
- models allowed:
  - class-balanced logistic regression
  - one small gradient-boosting baseline

Constraints:

- no backbone changes
- no broad hyperparameter search
- validation is used for model selection, never fitting
- preserve a fully reproducible feature manifest

Promotion gate:

- validation AFib AUROC improves by at least 0.01
- validation AFib AP improves by at least 0.02
- recall at the Day 1 false-alert burden improves by at least 0.08
- performance is not driven by one patient/record in leave-one-record-out analysis

Stop condition:

- if neither model passes, stop AFib model development for this round.
- do not add a neural hierarchical head during this six-day window.

Deliverables:

- `build_afib_meta_features.py`
- `train_afib_meta_head.py`
- `results_v7_round7/afib_meta_head/`
- serialized model only when the promotion gate passes

### Day 4: Sunday, 2026-06-14

Objective: prepare a genuinely external AFib validation pilot.

Primary database:

- Long-Term AF Database
- 84 two-lead recordings, typically 24-25 hours
- 128 Hz with beat and rhythm annotations
- full uncompressed size is about 3.4 GB

Pilot scope:

- download metadata and 8-12 records first, not all 84 records
- select records before running the model, using annotation-derived AF burden
  strata rather than model performance
- keep each record as an independent patient/record unit
- resample one lead to 250 Hz using the existing preprocessing path
- stream windows or save compact metadata first; do not materialize a large
  duplicate tensor dataset until the audit passes

Checks:

- signal scaling and finite-value audit
- rhythm-label mapping
- window count and AFib prevalence
- normal-to-AFib and AFib-to-normal event extraction
- no overlap with project training sources

Deliverables:

- `prepare_ltafdb_external.py`
- `data/ltafdb_pilot/` or a read-only external-data location
- `results_v7_round7/ltafdb_pilot_audit/`

Stop condition:

- if annotation mapping or signal normalization cannot be validated, do not
  report external performance and do not train on the pilot.

### Day 5: Monday, 2026-06-15

Objective: apply frozen models and frozen policies to the external pilot.

Evaluation order:

1. accepted Round5 ensemble
2. Day 2 AFib directional policy, only if promoted on validation
3. Day 3 AFib meta-head, only if promoted on validation

No external threshold tuning is allowed.

Report:

- AFib AUROC and AP
- AFib event recall and incident recall
- median lead time
- false AFib alert episodes/hr
- calibration shift
- per-record results and leave-one-record-out sensitivity

Interpretation rules:

- treat the pilot as preliminary external evidence, not a full clinical claim
- AUROC around or above 0.65 with stable per-record behavior is useful evidence
- poor fixed-threshold recall with retained AUROC indicates calibration/domain
  shift, not necessarily complete discrimination failure
- if performance collapses, investigate preprocessing and domain shift before
  any retraining

Decision:

- pass: schedule a later full 84-record external evaluation
- partial: keep as transparent domain-shift evidence
- fail: do not add an external-performance claim to the competition report

### Day 6: Tuesday, 2026-06-16

Objective: integrate only validated improvements and close Round7 cleanly.

System work:

- add AFib directional output only if the Day 2 or Day 3 gate passed
- retain Round5 overall warning as the default
- retain RR-gated low-burden mode as explicitly exploratory
- show agreement level and the non-clinical-confidence disclaimer
- add one AFib success case and one failure/review case to the dashboard

Verification:

- run `test_ensemble_serving.py`
- run API tests on at least three local records
- verify dashboard replay in the browser
- verify old Round5 output compatibility
- ensure no missing-checkpoint fallback is possible

Documentation:

- create `results_v7_round7/ROUND7_PROGRESS.md`
- update the competition report only with completed, reproducible results
- copy final policy JSON, selected figures, and the report into
  `C:\HealthMonitor\final_artifacts\`
- record rejected experiments and their stop reasons

## 4. Priority Order

1. AFib measurement and failure analysis.
2. AFib directional policy.
3. One lightweight AFib meta-head experiment.
4. Long-Term AF Database external pilot.
5. Dashboard and report integration.

VT/VF/AT-SVT model redesign is deferred. It should begin only after obtaining
more event/patient diversity or defining a separate high-risk any-occurrence
label with a newly sealed evaluation set.

## 5. Resource Budget

Available machine resources at plan creation:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- free disk on `C:`: approximately 109.5 GB

Round7 limits:

- no more than two AFib meta-model families
- no full neural multi-seed training unless a validation candidate first passes
- external pilot download target below 1 GB
- preserve at least 80 GB free disk during the round
- archive or remove only confirmed temporary outputs

## 6. Round7 Success Definition

Round7 is successful if at least one of the following is achieved:

1. A validation-selected AFib directional output improves recall under a
   controlled false-alert burden and remains stable externally.
2. The external pilot demonstrates useful AFib ranking ability without
   threshold retuning.
3. A negative result clearly establishes that current AFib limitations are
   data/domain-shift limited, preventing further low-value model iteration.

The official Round5 overall arrhythmia baseline remains valid unless a future
candidate is confirmed on an untouched evaluation source.
