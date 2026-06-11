"""
Attach saved RR features to a final-validation output CSV with strict alignment.

The dataset shards and inference outputs share deterministic, non-shuffled sample
order. This script verifies row count, current labels, and future labels before
writing a new CSV. It never modifies the source evaluation file.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch


RR_FEATURE_NAMES = (
    "rr_mean_rr_norm",
    "rr_sdnn_norm",
    "rr_rmssd_norm",
    "rr_pnn50",
    "rr_sample_entropy",
    "rr_sd1_norm",
    "rr_sd2_norm",
    "rr_sd1_sd2_ratio",
    "rr_cv",
)


def load_rr_and_labels(dataset_dir, split):
    paths = sorted(glob.glob(str(Path(dataset_dir) / f"{split}_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(
            f"no shards found: {dataset_dir}/{split}_shard_*.pt"
        )

    rr_rows = []
    current_labels = []
    future_labels = []
    for path in paths:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if "X_rr" not in data:
            raise KeyError(f"{path} does not contain X_rr")
        x_rr = data["X_rr"]
        if x_rr.ndim != 3 or x_rr.shape[-1] != len(RR_FEATURE_NAMES):
            raise ValueError(f"unexpected X_rr shape in {path}: {tuple(x_rr.shape)}")

        y_cur = data.get("Y_cur", data.get("Y"))
        y_fut = data.get("Y_fut")
        if y_cur is None or y_fut is None:
            raise KeyError(f"{path} is missing Y_cur/Y_fut labels")

        rr_rows.append(x_rr[:, -1, :].float().numpy())
        current_labels.append(y_cur.long().numpy())
        future_labels.append(y_fut.float().argmax(dim=1).long().numpy())

    return (
        np.concatenate(rr_rows, axis=0),
        np.concatenate(current_labels, axis=0),
        np.concatenate(future_labels, axis=0),
        paths,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--dataset-dir", default="dataset_v7")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    source = pd.read_csv(args.outputs_csv)
    rr, y_cur, y_fut, paths = load_rr_and_labels(args.dataset_dir, args.split)

    if len(source) != len(rr):
        raise RuntimeError(
            f"row-count mismatch: outputs={len(source)}, shard samples={len(rr)}"
        )
    csv_cur = source["target_cur"].to_numpy(dtype=int)
    csv_fut = source["target_fut"].to_numpy(dtype=int)
    if not np.array_equal(csv_cur, y_cur):
        mismatch = int(np.flatnonzero(csv_cur != y_cur)[0])
        raise RuntimeError(f"current-label alignment failed at row {mismatch}")
    if not np.array_equal(csv_fut, y_fut):
        mismatch = int(np.flatnonzero(csv_fut != y_fut)[0])
        raise RuntimeError(f"future-label alignment failed at row {mismatch}")
    if not np.isfinite(rr).all():
        raise RuntimeError("RR features contain non-finite values")

    output = source.copy()
    for feature_i, feature_name in enumerate(RR_FEATURE_NAMES):
        output[feature_name] = rr[:, feature_i]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Attached {len(RR_FEATURE_NAMES)} RR features to {len(output)} rows "
        f"from {len(paths)} shards: {output_path}"
    )


if __name__ == "__main__":
    main()
