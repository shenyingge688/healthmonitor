# LTAFDB Frozen External Pilot Evaluation

- records: 12
- 10-second evaluation rows: 98808
- protocol SHA-256: `366a3ade4a02ea076b2f72bc5a3a7190852591d0c2e6b44920f8358de84263e6`
- external threshold tuning: False

## Baseline

- dominant AFib AUROC: 0.679
- dominant AFib recall: 0.115
- AFib event recall: 0.206
- false episodes/hr: 2.656

## Frozen AFib Directional Candidate

- dominant AFib AUROC: 0.719
- dominant AFib AP: 0.552
- dominant AFib precision: 0.579
- dominant AFib recall: 0.478
- AFib event recall: 0.545
- incident event recall: 0.540
- median lead time: 260.0 s
- false episodes/hr: 2.897

Decision: `preliminary_external_support`.

This is a preliminary 12-record external pilot, not a clinical claim.