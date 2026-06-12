# LTAFDB Pilot Signal Integrity and Preprocessing Audit

- records audited: 12
- downloaded signal size: 487.7 MB
- all file sizes match headers: True
- all WFDB checksums match: True
- all initial samples match headers: True
- preprocessing probes passed: 574/576 (0.997)
- records with isolated failed probes: ['113']
- lead 0 selected: 12
- lead 1 quality fallback selected: 0
- failed records: 0
- audit pass: True

## Frozen Data-Quality Rule

Lead 0 is used whenever it passes fixed integrity, amplitude, clipping, flatline, and preprocessing gates. Lead 1 is used only as a quality fallback. Model outputs are not consulted.

Each probe follows the existing project preprocessing path: 0.5-45 Hz Butterworth filtering, 128-to-250 Hz polyphase resampling, and per-window z-normalization. Every 30-second probe must contain 7,500 finite samples after resampling.

Record `113` contains an isolated interval where both leads are nearly constant. It passes the pre-specified record-level 95% probe gate, but external evaluation must retain the full-time primary analysis and separately report a signal-quality-valid sensitivity analysis.

No model inference or external threshold selection was performed.