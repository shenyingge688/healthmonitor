# V7 Round 5 Optimization Plan

Date: 2026-06-10

Goal: improve performance without weakening the current biomedical-engineering evaluation discipline.

## Non-Negotiable Evaluation Rules

- Patient/record-wise splits stay fixed.
- Validation split selects model, horizon, calibration, thresholds, smoothing, and alert policy.
- Locked test is used only after a candidate is selected on validation.
- Do not promote a candidate that improves one secondary metric by damaging the main validation endpoints.
- Main model endpoints: future macro-F1, future macro-AUROC, rare recall, ECE, transition recall.
- Main alert endpoints: event recall, incident recall, median lead time, false alert episodes per patient-hour, false alert windows per patient-hour.

## Current Reference

Current accepted model:

- checkpoint: `models_v7_stage1/seeds/seed0/arrhythmia_warning_best.pth`
- selected epoch: 12
- validation future macro-F1: 0.2886
- validation future macro-AUROC: 0.7254
- validation rare recall: 0.3711

Current accepted alert policy:

- task: `future_arrhythmia`
- probability source: temperature-scaled
- threshold: 0.36
- smoothing: causal rolling mean, window 2
- consecutive trigger: k=2
- locked-test event recall: 0.900
- locked-test incident recall: 0.862
- locked-test median lead time: 225 s
- locked-test false alert episodes per patient-hour: 1.30

## Step 1: 3-Minute Horizon Study

Hypothesis: a 180 s future-label horizon may improve short-term warning coherence compared with the current 300 s horizon.

Execution:

1. Build `dataset_v7_h180` for train and validation only.
2. Audit sample counts, label distributions, transition weights, and horizon metadata.
3. Train seed0 with the current baseline settings.
4. Evaluate validation only.
5. Compare against the 300 s baseline.

Promotion condition:

- Validation future macro-F1 and AUROC are not meaningfully worse than baseline, and
- alert-level validation behavior is clinically cleaner, especially false alert burden and lead-time coherence.

Do not touch locked test during this step.

## Step 2: Current-Core Multiseed

If 180 s is promising, run seeds 1-4 on the selected horizon. If 180 s is not promising, run seeds 1-4 on the current 300 s core.

Promotion condition:

- mean performance remains stable across seeds, and
- variance is small enough to defend in competition material.

## Step 3: Seed Ensemble

After multiseed results exist, average probabilities across selected seeds and tune alert policy on validation.

Promotion condition:

- validation AUROC/calibration or alert burden improves without reducing event recall.

## Step 4: CUDB Incremental Data Study

Only after the horizon decision is made, test whether adding CUDB to training improves rare/high-risk behavior.

Promotion condition:

- validation rare recall or VT/VF recall improves without a major increase in false alert burden.

## Paused Directions

- Future-event sampler: rejected for now because it reduced future macro-F1, rare recall, and AUROC.
- Encoder-tail fine-tuning: not a main route because the prior trial hurt future macro-F1 and rare recall.
- PTB-XL direct mixing: not suitable for the future-warning label task; keep it as pretraining/backbone material.
