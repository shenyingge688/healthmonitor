# AFib Validation Baseline and Directional Policy

This analysis uses validation data only. The accepted overall arrhythmia alarm is unchanged.

## Support and Audit

- rows: 2188
- patient/record units: 21
- current AFib windows: 370 from 5 records
- dominant future AFib windows: 349 from 5 records
- any future AFib mass windows: 422 from 5 records
- event-credit horizon: 300s
- ensemble member row and label alignment: exact

## Observed Results

| policy | dominant precision | dominant recall | dominant F1 | event recall | median lead | false episodes/hr |
|---|---:|---:|---:|---:|---:|---:|
| future-class argmax | 0.251 | 0.160 | 0.196 | 0.400 | -650s | 4.15 |
| afib_vs_normal_ewma_k1_t0.100 | 0.322 | 0.653 | 0.431 | 0.733 | 280s | 3.71 |

## Patient Bootstrap

- argmax dominant recall 95% CI: 0.082-0.249
- selected dominant recall 95% CI: 0.282-1.000
- argmax event recall 95% CI: 0.123-0.786
- selected event recall 95% CI: 0.200-1.000

## Stability and Decision

- positive-record leave-one-out gate pass: 4/5
- decision: `validation_candidate`
- permitted scope: `external_evaluation_only`
- reason: The simple policy passed the full validation gate and at least 80% of positive-record leave-one-out influence checks, but only five positive records are available.

The candidate is not a replacement for the Round5 overall arrhythmia alarm. External AFib validation remains required before any deployment claim.