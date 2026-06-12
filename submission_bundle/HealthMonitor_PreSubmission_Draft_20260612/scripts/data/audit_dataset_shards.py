"""
Audit sharded ECG trajectory tensors before training/evaluation.

This is intentionally read-only. It summarizes sample counts, tensor shapes,
label distributions, transition weights, soft-label validity, and optional
finite-value checks shard by shard.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
REQUIRED_KEYS = ["X", "Y_cur", "Y_fut", "T_weight"]


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def finite_check(x, mode="sample", sample_rows=64):
    if mode == "none":
        return {"checked": 0, "nonfinite": 0, "mode": mode}
    if mode == "sample":
        x = x[: min(int(sample_rows), len(x))]
    checked = int(x.numel()) if isinstance(x, torch.Tensor) else int(np.asarray(x).size)
    nonfinite = int((~torch.isfinite(x)).sum().item()) if isinstance(x, torch.Tensor) else int((~np.isfinite(x)).sum())
    return {"checked": checked, "nonfinite": nonfinite, "mode": mode}


def audit_shard(path, finite_mode="sample", sample_rows=64):
    data = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"{path}: missing required keys {missing}")

    x = data["X"]
    x_rr = data.get("X_rr")
    y_cur = _as_numpy(data["Y_cur"]).astype(np.int64)
    y_fut = _as_numpy(data["Y_fut"]).astype(np.float32)
    t_weight = _as_numpy(data["T_weight"]).astype(np.float32)
    y_fut_argmax = y_fut.argmax(axis=1) if len(y_fut) else np.array([], dtype=np.int64)

    n = int(len(x))
    if len(y_cur) != n or len(y_fut) != n or len(t_weight) != n:
        raise ValueError(f"{path}: inconsistent first dimension")
    if y_fut.ndim != 2 or y_fut.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"{path}: Y_fut must have shape [N, {len(CLASS_NAMES)}]")

    y_cur_counts = np.bincount(y_cur, minlength=len(CLASS_NAMES))
    y_fut_counts = np.bincount(y_fut_argmax, minlength=len(CLASS_NAMES))
    soft_sum = y_fut.sum(axis=1) if len(y_fut) else np.array([], dtype=np.float32)
    finite_x = finite_check(x, finite_mode, sample_rows)
    finite_rr = finite_check(x_rr, finite_mode, sample_rows) if x_rr is not None else {"checked": 0, "nonfinite": 0, "mode": "missing"}

    row = {
        "path": str(path),
        "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        "samples": n,
        "x_shape": list(x.shape),
        "x_rr_shape": list(x_rr.shape) if x_rr is not None else None,
        "y_fut_shape": list(y_fut.shape),
        "soft_sum_min": float(soft_sum.min()) if len(soft_sum) else None,
        "soft_sum_max": float(soft_sum.max()) if len(soft_sum) else None,
        "soft_sum_mean": float(soft_sum.mean()) if len(soft_sum) else None,
        "t_weight_mean": float(t_weight.mean()) if len(t_weight) else None,
        "t_weight_max": float(t_weight.max()) if len(t_weight) else None,
        "t_weight_gt1_frac": float((t_weight > 1.0).mean()) if len(t_weight) else None,
        "x_nonfinite": finite_x["nonfinite"],
        "x_checked": finite_x["checked"],
        "x_rr_nonfinite": finite_rr["nonfinite"],
        "x_rr_checked": finite_rr["checked"],
    }
    for idx, name in enumerate(CLASS_NAMES):
        key = name.lower().replace("/", "_").replace("-", "_")
        row[f"y_cur_{key}"] = int(y_cur_counts[idx])
        row[f"y_fut_{key}"] = int(y_fut_counts[idx])
    return row


def aggregate(rows, split):
    total = {
        "split": split,
        "shards": len(rows),
        "samples": int(sum(r["samples"] for r in rows)),
        "file_size_mb": round(sum(r["file_size_mb"] for r in rows), 2),
        "x_nonfinite": int(sum(r["x_nonfinite"] for r in rows)),
        "x_checked": int(sum(r["x_checked"] for r in rows)),
        "x_rr_nonfinite": int(sum(r["x_rr_nonfinite"] for r in rows)),
        "x_rr_checked": int(sum(r["x_rr_checked"] for r in rows)),
    }
    if rows:
        total["t_weight_gt1_frac_weighted"] = float(
            sum(r["t_weight_gt1_frac"] * r["samples"] for r in rows if r["t_weight_gt1_frac"] is not None)
            / max(total["samples"], 1)
        )
        total["t_weight_max"] = float(max(r["t_weight_max"] for r in rows if r["t_weight_max"] is not None))
        total["soft_sum_min"] = float(min(r["soft_sum_min"] for r in rows if r["soft_sum_min"] is not None))
        total["soft_sum_max"] = float(max(r["soft_sum_max"] for r in rows if r["soft_sum_max"] is not None))
    for name in CLASS_NAMES:
        key = name.lower().replace("/", "_").replace("-", "_")
        total[f"y_cur_{key}"] = int(sum(r[f"y_cur_{key}"] for r in rows))
        total[f"y_fut_{key}"] = int(sum(r[f"y_fut_{key}"] for r in rows))
    return total


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset_v7")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--output-dir", default="results/dataset_audit")
    ap.add_argument("--finite-check", choices=["none", "sample", "full"], default="sample")
    ap.add_argument("--sample-rows", type=int, default=64)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    split_summaries = []
    all_shards = []

    for split in args.splits:
        paths = sorted(dataset_dir.glob(f"{split}_shard_*.pt"))
        rows = [audit_shard(p, args.finite_check, args.sample_rows) for p in paths]
        for row in rows:
            row["split"] = split
        all_shards.extend(rows)
        split_summaries.append(aggregate(rows, split))

    write_csv(out_dir / "dataset_shard_audit.csv", all_shards)
    write_csv(out_dir / "dataset_split_audit.csv", split_summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dataset_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset_dir": str(dataset_dir), "splits": split_summaries}, f, indent=2)

    print(f"Audit complete: {dataset_dir}")
    for row in split_summaries:
        print(
            f"  {row['split']}: shards={row['shards']} samples={row['samples']} "
            f"T_weight>1={row.get('t_weight_gt1_frac_weighted', 0.0):.3f} "
            f"X_nonfinite={row['x_nonfinite']}/{row['x_checked']}"
        )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
