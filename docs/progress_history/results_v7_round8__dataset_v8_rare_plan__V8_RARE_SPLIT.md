# V8 Rare Validation Pre-registration

- unique records: 8
- source: V7 training records only
- model performance used for selection: no
- V7 artifacts modified: no
- training started: no
- minimum positive training records: 5

## Selected Records
- afib_h300: afdb:04936, vfdb:615
- vf_h60: vfdb:418, vfdb:419
- vt_h180: vfdb:611, vfdb:607
- svt_like_h180: mitdb:222, mitdb:203

## Annotation-only Training Gate

- afib_h300: 10 positive training records after holdout; eligible=True
- vf_h60: 1 positive training records after holdout; eligible=False
- vt_h180: 12 positive training records after holdout; eligible=True
- svt_like_h180: 2 positive training records after holdout; eligible=False
- vt_vf_combined_h180: 12 positive training records after holdout; eligible=True
