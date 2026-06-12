"""
Plan a V7 patient-wise data split without rebuilding tensor shards.

The script inventories every record under data/, estimates whether it is usable
under the current 10min-history + 5min-future trajectory design, and writes a
non-destructive train/val/test/demo split plan for review.
"""
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import build_dataset_factory as B


DBS = ["mitdb", "afdb", "vfdb", "cudb", "svdb"]
CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]

# Dashboard-facing examples must not leak into V7 training.
DEMO_MITDB_V7 = ["100", "119", "209", "210", "223", "201", "207"]

VAL_HINTS = {
    "mitdb": ["214", "217", "221", "233"],
    "afdb": ["08219", "04126", "06426", "06995"],
    "vfdb": ["424", "428", "430", "602", "612"],
    "svdb": ["802", "804", "841", "857", "864", "880", "887", "894"],
}

TEST_HINTS = {
    "mitdb": ["205", "215", "219", "234"],
    "afdb": ["04908", "07162", "07859", "08378"],
    "vfdb": ["422", "423", "427", "609", "614"],
    "svdb": ["803", "805", "851", "854", "860", "878", "884", "892"],
}


def record_ids(db):
    db_dir = Path(B.DATA_DIR) / db
    if not db_dir.is_dir():
        return []
    return sorted({
        p.stem for p in db_dir.iterdir()
        if p.suffix.lower() in {".dat", ".qrs"}
    })


def extend_svt_tail(labels):
    labels = labels.copy()
    svt_mask = labels == 5
    if svt_mask.any():
        svt_ends = np.where(svt_mask[:-1] & ~svt_mask[1:])[0]
        for end_pos in svt_ends:
            region = labels[end_pos:min(len(labels), end_pos + 15000)]
            region[region == 0] = 5
    return labels


def planned_windows(total_pts, db):
    max_pt = total_pts - (B.HISTORY_SEC + B.PREDICT_SEC) * B.TARGET_FS
    if max_pt <= 0:
        return 0
    stride_sec = B.DB_STRIDE_OVERRIDE.get(db, B.FIXED_STRIDE_SEC)
    stride_pts = int(stride_sec * B.TARGET_FS)
    return int(math.ceil(max_pt / stride_pts))


def walk_targets(labels, db):
    n = planned_windows(len(labels), db)
    if n <= 0:
        return Counter(), Counter(), 0
    stride_sec = B.DB_STRIDE_OVERRIDE.get(db, B.FIXED_STRIDE_SEC)
    stride_pts = int(stride_sec * B.TARGET_FS)
    y_cur, y_fut, transitions = Counter(), Counter(), 0
    for i in range(n):
        current_pt = i * stride_pts
        history_end = current_pt + B.HISTORY_SEC * B.TARGET_FS
        pred_start = history_end
        pred_end = pred_start + B.PREDICT_SEC * B.TARGET_FS
        cur = B.get_current_label(labels, history_end)
        fut = int(np.argmax(B.get_distribution(labels, pred_start, pred_end)))
        y_cur[cur] += 1
        y_fut[fut] += 1
        transitions += int(cur != fut)
    return y_cur, y_fut, transitions


def inspect_record(db, rec_id):
    path = Path(B.DATA_DIR) / db / rec_id
    row = {
        "db": db,
        "record_id": rec_id,
        "patient_id": f"{db}:{rec_id}",
        "path": str(path),
        "readable": False,
        "usable_trajectory": False,
        "exclude_reason": "",
        "fs": np.nan,
        "duration_min": np.nan,
        "planned_stride_sec": B.DB_STRIDE_OVERRIDE.get(db, B.FIXED_STRIDE_SEC),
        "planned_windows": 0,
        "transition_windows": 0,
    }
    try:
        record = wfdb.rdrecord(str(path))
        annotation = wfdb.rdann(str(path), "atr", pn_dir=None)
        signal_len = record.sig_len if hasattr(record, "sig_len") else len(record.p_signal)
        src_fs = record.fs if hasattr(record, "fs") and record.fs else 360
        max_samples = 2 * 3600 * int(src_fs)
        clipped_len = min(int(signal_len), max_samples)
        total_pts = (
            math.ceil(clipped_len * B.TARGET_FS / int(src_fs))
            if src_fs != B.TARGET_FS else clipped_len
        )
        labels = B.parse_annotations(annotation, B.TARGET_FS, src_fs, total_pts)
        labels = extend_svt_tail(labels)
        y_cur, y_fut, transitions = walk_targets(labels, db)
        label_seconds = np.bincount(labels, minlength=6).astype(float) / B.TARGET_FS

        row.update({
            "readable": True,
            "fs": float(src_fs),
            "duration_min": clipped_len / float(src_fs) / 60.0,
            "planned_windows": int(sum(y_cur.values())),
            "transition_windows": int(transitions),
            "usable_trajectory": int(sum(y_cur.values())) > 0,
        })
        if not row["usable_trajectory"]:
            row["exclude_reason"] = "shorter_than_history_plus_future"
        for i, name in enumerate(CLASS_NAMES):
            key = name.lower().replace("/", "_").replace("-", "_")
            row[f"label_seconds_{key}"] = float(label_seconds[i])
            row[f"y_cur_{key}"] = int(y_cur.get(i, 0))
            row[f"y_fut_{key}"] = int(y_fut.get(i, 0))
    except Exception as exc:
        row["exclude_reason"] = f"{type(exc).__name__}: {exc}"
    return row


def assign_split(row):
    db = row["db"]
    rec = row["record_id"]
    if not row["usable_trajectory"]:
        return "excluded"
    if db == "mitdb" and rec in DEMO_MITDB_V7:
        return "demo"
    if rec in set(VAL_HINTS.get(db, [])):
        return "val"
    if rec in set(TEST_HINTS.get(db, [])):
        return "test"
    return "train"


def summarize_split(df):
    rows = []
    for split, g in df.groupby("split"):
        usable = g[g["usable_trajectory"]]
        row = {
            "split": split,
            "records": int(len(g)),
            "usable_records": int(len(usable)),
            "planned_windows": int(usable["planned_windows"].sum()),
            "transition_windows": int(usable["transition_windows"].sum()),
        }
        for c in CLASS_NAMES:
            key = c.lower().replace("/", "_").replace("-", "_")
            row[f"label_seconds_{key}"] = float(usable.get(f"label_seconds_{key}", pd.Series(dtype=float)).sum())
            row[f"y_cur_{key}"] = int(usable.get(f"y_cur_{key}", pd.Series(dtype=int)).sum())
            row[f"y_fut_{key}"] = int(usable.get(f"y_fut_{key}", pd.Series(dtype=int)).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split")


def write_plan(df, out_dir):
    split_plan = defaultdict(lambda: defaultdict(list))
    for _, row in df[df["split"].isin(["train", "val", "test", "demo"])].iterrows():
        split_plan[row["split"]][row["db"]].append(row["record_id"])
    split_plan = {
        split: {db: recs for db, recs in sorted(dbs.items())}
        for split, dbs in sorted(split_plan.items())
    }
    with open(out_dir / "split_plan_v7.json", "w", encoding="utf-8") as f:
        json.dump(split_plan, f, indent=2, ensure_ascii=False)
    return split_plan


def write_markdown(summary, split_plan, out_dir):
    def markdown_table(df):
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Dataset V7 Split Plan",
        "",
        "Non-destructive planning output. Existing dataset shards are not modified.",
        "",
        "## Split Summary",
        markdown_table(summary),
        "",
        "## Design Notes",
        "- Demo records are held out from training, including new dashboard cases 210 and 223.",
        "- AFDB uses the planned 50s stride from DB_STRIDE_OVERRIDE.",
        "- CUDB is excluded under the current 10min history + 5min future trajectory because records are too short.",
        "- SVDB is included mainly as PVC/normal support under the current parser; AT/SVT claims still require parser work.",
        "- VF appears as short current/soft-label evidence but rarely as future-majority; use VT/VF high-risk evaluation instead of a standalone VF argmax claim.",
        "",
        "## Records",
    ]
    for split, dbs in split_plan.items():
        lines.append(f"### {split}")
        for db, recs in dbs.items():
            lines.append(f"- {db}: {', '.join(recs)}")
        lines.append("")
    (out_dir / "dataset_v7_plan_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    out_dir = Path("results") / "dataset_v7_plan"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for db in DBS:
        for rec_id in record_ids(db):
            rows.append(inspect_record(db, rec_id))
    df = pd.DataFrame(rows)
    df["split"] = df.apply(assign_split, axis=1)

    inventory_path = out_dir / "record_inventory.csv"
    summary_path = out_dir / "split_summary.csv"
    df.to_csv(inventory_path, index=False, encoding="utf-8")
    summary = summarize_split(df)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    split_plan = write_plan(df, out_dir)
    write_markdown(summary, split_plan, out_dir)

    print(f"Inventory -> {inventory_path}")
    print(f"Summary   -> {summary_path}")
    print(f"Plan      -> {out_dir / 'split_plan_v7.json'}")
    print(f"Markdown  -> {out_dir / 'dataset_v7_plan_summary.md'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
