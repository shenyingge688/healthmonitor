# LTAFDB External Pilot Metadata Audit

- source: https://physionet.org/content/ltafdb/1.0.0/
- database version: 1.0.0
- records audited: 84
- pilot records selected: 12
- downloaded metadata size: 17.6 MB
- signal `.dat` files downloaded: 0
- all records at 128 Hz: True
- all records have two signals: True
- duration range starts at: 6.13 h
- minimum labeled coverage: 0.232
- records eligible for pilot selection: 75
- pilot minimum duration: 20.47 h
- pilot minimum labeled coverage: 0.992
- total valid 10-min-history/5-min-future AF onsets: 110
- pilot burden strata: {'low': 4, 'medium': 4, 'high': 4}
- audit pass: True

## Selection Rule

Records were selected before model inference using only annotation-derived AF burden, valid AF onset count, duration, and annotation coverage. Each record remains an independent patient/record unit. Thresholds must not be reselected on this external pilot.

AF segments separated by at most 60 seconds are merged, episodes shorter than 30 seconds are excluded, and a valid prediction onset requires a previous `N` rhythm, at least 60 seconds of AF, and 10 AF-free minutes of history.

## Selected Records

| record | stratum | AF burden | AF episodes | valid onsets | reason |
|---|---|---:|---:|---:|---|
| 103 | high | 0.519 | 1 | 0 | high_burden_near_0.650 |
| 10 | high | 0.670 | 21 | 5 | high_burden_transition_rich |
| 01 | high | 0.789 | 11 | 6 | high_burden_transition_rich |
| 74 | high | 0.928 | 1 | 0 | high_burden_near_0.900 |
| 120 | low | 0.012 | 9 | 7 | low_burden_transition_rich |
| 32 | low | 0.030 | 45 | 2 | low_burden_near_0.025 |
| 100 | low | 0.079 | 87 | 12 | low_burden_transition_rich |
| 113 | low | 0.083 | 1 | 1 | low_burden_near_0.075 |
| 101 | medium | 0.140 | 56 | 8 | medium_burden_transition_rich |
| 121 | medium | 0.156 | 38 | 7 | medium_burden_transition_rich |
| 28 | medium | 0.225 | 7 | 3 | medium_burden_near_0.200 |
| 117 | medium | 0.383 | 1 | 0 | medium_burden_near_0.400 |

## Rhythm Notes

`AB`=4472, `AFIB`=7358, `B`=2696, `IVR`=137, `N`=22834, `SBR`=11326, `SVTA`=3268, `T`=785, `VT`=828

No external performance is reported at this stage.