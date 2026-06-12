# V7 Round 8 Progress

Updated: 2026-06-11

## Frozen Baseline

- Official model: seed0 + seed1 + seed4 logits ensemble.
- Official overall score: `1 - P(Normal)`.
- Policy: EWMA alpha 0.65, threshold 0.10, consecutive k=2, 10-second inference cadence.
- Locked-test event recall: 0.925.
- Locked-test incident event recall: 0.897.
- Locked-test median lead time: 290 seconds.
- Locked-test false alert episodes per patient hour: 1.12.
- The official Round5 output is unchanged.

## Completed

### Shared serving policy and API

- Added `monitoring_policy.py` as the shared causal alarm implementation.
- Added raw overall risk, AFib direction score, signal quality, agreement level,
  policy version, input status, and history completeness to the API.
- Preserved all previous response fields.
- Ensemble probability regression maximum error: `1.234e-07`.

### Soft-target evidence

- Added soft cross-entropy, KL, Brier, abnormal-mass MAE/RMSE/bias.
- Validation rows: 2,188 from 21 patient/record units.
- Soft cross-entropy: 1.336.
- Soft KL: 1.010.
- Soft Brier: 0.380.
- Abnormal-mass MAE: 0.266.
- Abnormal-mass RMSE: 0.355.
- Abnormal-mass bias: 0.013.

### Rare-class directional gate

- VT vs Normal: stopped because false alert burden was 2.851 episodes/hour.
- VF vs Normal: stopped because only one positive record was available.
- VT/VF vs Normal: passed the minimum validation gate at threshold 0.60,
  raw score, k=2; event recall 0.214, median lead 240 seconds, false burden
  0.432 episodes/hour.
- VT + AT/SVT vs Normal: passed the event gate, but future-window recall was
  only 0.044, so it does not support an AT/SVT performance claim.
- No rare-class score is connected to the official alarm.

### Frontend and demo evidence

- Rebuilt the dashboard as monitoring, case replay, and evidence/boundary pages.
- Official events are generated only by the shared Round5 policy.
- Replaced clinical-confidence and lesion-localization wording with model
  agreement and model-attention wording.
- Added signal-quality and symptom-based safety messages.
- Removed the inaccurate record 210 AFib-success and record 223 VT-success claims.
- Frozen cases:
  - record 100: 0/19 official alarm points, future Normal 19/19.
  - record 119: current and future PVC 19/19.
  - record 201 at 26.5-29.5 minutes: future AFib 18/19, maximum AFib direction
    score 0.937.
  - record 223: future VT 0/10; boundary only.
  - record 209: current/future AT-SVT 0/35; boundary only.
- Frozen replay versus API maximum probability error: 0.
- Browser checks passed at 1280x720 and 1440x900.

### External-data gate

- CUDB contains 35 records of 8.482 minutes each.
- No record satisfies the fixed 10-minute history plus 5-minute future
  protocol.
- CUDB performance evaluation was stopped; zero-padding results will not be
  reported as external evidence.

### Full LTAFDB fixed signal audit

- Completed the full 84-record frozen LTAFDB manifest download and fixed
  quality audit in manifest order.
- Frozen full-manifest SHA-256:
  `b91ad7cfc9c5c5ade9265b0f791851ae6c495a3a7c04a7bd3050afe719f2abd7`.
- Local cache: 84 complete `.dat` files, 3,613,777,920 bytes; no retained
  `.part` files.
- All header-derived file sizes match exactly.
- Byte-range resume was used during download, and a partial file was promoted
  only after reaching the exact header-derived byte count.
- Final audit outputs:
  `results_v7_round8/ltafdb_full_signal_audit/`.
- Audit pass is `False` under the fixed gates:
  - record 20 failed both leads because WFDB checksum and initial sample
    values did not match the header; no quality fallback lead was selected.
  - record 113 has 2 failed preprocessing probes out of 4,032 total probes;
    these are isolated near-constant intervals and the record-level probe pass
    fraction remains above the fixed gate.
- Preprocessing probes passed: 4,030/4,032.
- Lead 0 selected: 83 records.
- Lead 1 quality fallback selected: 0 records.
- Failed fixed-quality records: 1.
- Record 20 was independently verified byte-for-byte against the PhysioNet
  source. Its full-file SHA-256 is
  `1cd790a57f70c7ffbef0f020d04910ff381d22656984aeb601037f71433c2eb3`;
  the mismatch is present in the official source header rather than being a
  local download error.
- External thresholds remain frozen and will not be tuned on LTAFDB.

### Full LTAFDB frozen inference

- Frozen protocol version: `ltafdb-external-v3`.
- Protocol SHA-256:
  `fb7b48c6cce4c3ff3c41b042a0fbc9a87968a541597204aad27b8352d9acd845`.
- Inference completed for all 84 records and 698,256 10-second time points.
- Primary analysis is restricted to the 83 fixed-quality-pass records.
- Record 20 is retained only in an explicitly labeled 84-record sensitivity
  analysis; the fixed signal-quality gate was not relaxed.
- Direct-versus-cached checks: 252/252 member-record checks preserved current
  and future argmax.
- Maximum future probability error: `9.36e-4`.
- Maximum AFib directional-score error: `3.28e-4`.
- Duplicate patient/window rows: 0.
- Non-finite numeric values: 0.
- The 83-record fixed-quality primary analysis is complete:
  - dominant AFib AUROC: 0.721 (patient-bootstrap 95% CI 0.661-0.775).
  - dominant AFib average precision: 0.706.
  - dominant AFib recall: 0.409.
  - AFib event recall: 0.613 (95% CI 0.506-0.720).
  - incident event recall: 0.598.
  - median lead time: 240 seconds.
  - false alert episodes: 2.330 per patient hour.
- The all-84-record sensitivity analysis is materially unchanged:
  AUROC 0.720, event recall 0.614, median lead 240 seconds, and false alert
  burden 2.301 episodes per patient hour.
- All four frozen external support checks passed. The AFib directional score
  is now described as full-database external research support, not clinical
  validation, and it remains disconnected from the official Round5 alarm.

### V8 annotation-only training gate

- AFib h300 remains eligible after rare-validation holdout: 10 positive
  training records.
- VT h180 remains eligible: 12 positive training records.
- Combined VT/VF h180 remains eligible: 12 positive training records.
- Independent VF h60 is stopped: only 1 positive training record remains.
- SVT-like h180 is stopped: only 2 positive training records remain.
- A read-only V7 row index generated multi-horizon any-occurrence labels
  without duplicating waveform tensors.
- Three frozen-feature lightweight auxiliary heads were trained for seeds
  0, 1, and 4; no Round5 or backbone parameter was updated.
- Standard validation:
  - AFib h300 passed the event gate.
  - VT h180 failed with event recall 0.040.
  - VT/VF h180 passed the minimum event gate.
- Preregistered rare validation was opened only after models and policies
  were frozen:
  - AFib h300 stopped because only 4 positive records were available and
    false burden was 1.286 episodes/hour.
  - VT h180 remained stopped because standard validation had failed.
  - VT/VF h180 stopped because rare-validation false burden was
    1.286 episodes/hour.
- All V8 auxiliary heads stopped before locked-test evaluation. None is
  connected to the API or official alarm.

## Promotion Rules

- Keep formal, pilot, and exploratory capabilities separate.
- Do not promote any V8 auxiliary-head claim; all candidates stopped before
  locked-test evaluation.
- Do not overwrite V7 datasets, models, policies, or locked results.
