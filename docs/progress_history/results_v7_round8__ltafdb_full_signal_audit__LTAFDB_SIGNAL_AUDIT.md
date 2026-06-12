# LTAFDB Pilot Signal Integrity and Preprocessing Audit

- records audited: 84
- downloaded signal size: 3446.4 MB
- all file sizes match headers: True
- all WFDB checksums match: False
- all initial samples match headers: False
- preprocessing probes passed: 4030/4032 (1.000)
- records with isolated failed probes: ['113']
- lead 0 selected: 83
- lead 1 quality fallback selected: 0
- failed records: 1
- audit pass: False

## Frozen Data-Quality Rule

Lead 0 is used whenever it passes fixed integrity, amplitude, clipping, flatline, and preprocessing gates. Lead 1 is used only as a quality fallback. Model outputs are not consulted.

Each probe follows the existing project preprocessing path: 0.5-45 Hz Butterworth filtering, 128-to-250 Hz polyphase resampling, and per-window z-normalization. Every 30-second probe must contain 7,500 finite samples after resampling.

Record `113` contains an isolated interval where both leads are nearly constant. It passes the pre-specified record-level 95% probe gate, but external evaluation must retain the full-time primary analysis and separately report a signal-quality-valid sensitivity analysis.

No model inference or external threshold selection was performed.