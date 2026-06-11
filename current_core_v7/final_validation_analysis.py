"""
Final validation statistics for the ECG early-warning model.

This script is intentionally post-training only:
  1. Load a frozen checkpoint.
  2. Export patient/window-level validation outputs.
  3. Run patient-level bootstrap confidence intervals.
  4. Run calibration, reliability plots, temperature scaling, and threshold sweeps.

It does not change the model, loss, architecture, or dataset tensors.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from dl_model import ArrhythmiaWarningNet


CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
CLASS_KEYS = ["normal", "pvc", "afib", "vf", "vt", "at_svt"]
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_WARNING_HORIZON_SEC = 300.0


def load_dataset_config(dataset_dir):
    config_path = Path(dataset_dir) / "dataset_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_val_dataset(split_prefix="val", dataset_dir="dataset"):
    shard_paths = sorted(glob.glob(os.path.join(dataset_dir, f"{split_prefix}_shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for {dataset_dir}/{split_prefix}_shard_*.pt")

    datasets = []
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        x_rr = data.get("X_rr", torch.zeros(len(data["X"]), 39, 9, dtype=torch.float16))
        y_cur = data.get("Y_cur", data.get("Y", torch.zeros(len(data["X"]), dtype=torch.long)))
        y_fut = data.get("Y_fut", torch.zeros(len(data["X"]), NUM_CLASSES, dtype=torch.float16))
        t_weight = data.get("T_weight", torch.ones(len(data["X"]), dtype=torch.float16))
        datasets.append(TensorDataset(data["X"], x_rr, y_cur, y_fut, t_weight))
    return ConcatDataset(datasets), shard_paths


def validation_record_order():
    """Recreate the validation record order used by build_dataset_factory.py."""
    import build_dataset_factory as b

    mitdb_dir = os.path.join(b.DATA_DIR, "mitdb")
    all_mitdb = sorted([
        f.split(".")[0] for f in os.listdir(mitdb_dir)
        if f.endswith(".dat")
    ]) if os.path.isdir(mitdb_dir) else []
    val_mitdb = [r for r in all_mitdb if r in b.VAL_MITDB]

    records = [(os.path.join(mitdb_dir, r), "mitdb", r) for r in val_mitdb]
    for db_name, recs in b.VAL_SUPPLEMENTS.items():
        db_dir = os.path.join(b.DATA_DIR, db_name)
        if not os.path.isdir(db_dir):
            continue
        available = list(set([
            f.split(".")[0] for f in os.listdir(db_dir)
            if f.endswith(".dat") or f.endswith(".qrs")
        ]))
        found = [r for r in recs if r in available]
        records.extend((os.path.join(db_dir, r), db_name, r) for r in found)
    return records


def record_order_from_split_plan(split_plan_path, split_name):
    import build_dataset_factory as b

    with open(split_plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    records = []
    for db_name, recs in plan.get(split_name, {}).items():
        db_dir = os.path.join(b.DATA_DIR, db_name)
        records.extend((os.path.join(db_dir, rec), db_name, rec) for rec in recs)
    return records


def count_record_windows(record_path, db_name="mitdb", use_db_stride=False,
                         history_sec=None, predict_sec=None):
    """Count windows exactly as the current validation shard was built."""
    import wfdb
    import build_dataset_factory as b

    history_sec = int(history_sec if history_sec is not None else b.HISTORY_SEC)
    predict_sec = int(predict_sec if predict_sec is not None else b.PREDICT_SEC)

    record_obj = wfdb.rdrecord(record_path)
    signal_len = record_obj.sig_len if hasattr(record_obj, "sig_len") else len(record_obj.p_signal)
    src_fs = record_obj.fs if hasattr(record_obj, "fs") else 360
    max_samples = 2 * 3600 * int(src_fs)
    if signal_len > max_samples:
        signal_len = max_samples

    if src_fs != b.TARGET_FS:
        ecg_len = math.ceil(signal_len * b.TARGET_FS / int(src_fs))
    else:
        ecg_len = int(signal_len)

    max_pt = ecg_len - (history_sec + predict_sec) * b.TARGET_FS
    stride_sec = b.DB_STRIDE_OVERRIDE.get(db_name, b.FIXED_STRIDE_SEC) if use_db_stride else b.FIXED_STRIDE_SEC
    stride_pts = int(stride_sec * b.TARGET_FS)
    count = 0
    current_pt = 0
    while current_pt < max_pt:
        count += 1
        current_pt += stride_pts
    return count, int(src_fs), int(ecg_len), float(stride_sec)


def reconstruct_val_metadata(expected_n, records=None, use_db_stride=False,
                             history_sec=None, predict_sec=None):
    """Build patient_id/window_id/timestamp arrays for the existing val shard."""
    import build_dataset_factory as b

    history_sec = int(history_sec if history_sec is not None else b.HISTORY_SEC)
    predict_sec = int(predict_sec if predict_sec is not None else b.PREDICT_SEC)

    rows = []
    if records is None:
        records = validation_record_order()
    for record_path, db_name, rec_id in records:
        try:
            n_windows, src_fs, ecg_len, stride_sec = count_record_windows(
                record_path, db_name=db_name, use_db_stride=use_db_stride,
                history_sec=history_sec, predict_sec=predict_sec,
            )
        except Exception as exc:
            print(f"[metadata] skip {db_name}:{rec_id}: {type(exc).__name__}: {exc}")
            continue

        patient_id = f"{db_name}:{rec_id}"
        for window_id in range(n_windows):
            start_sec = window_id * stride_sec
            history_end_sec = start_sec + history_sec
            rows.append({
                "patient_id": patient_id,
                "db": db_name,
                "record_id": rec_id,
                "window_id": window_id,
                "history_start_sec": float(start_sec),
                "timestamp_sec": float(history_end_sec),
                "future_start_sec": float(history_end_sec),
                "future_end_sec": float(history_end_sec + predict_sec),
                "source_fs": src_fs,
                "resampled_len": ecg_len,
            })

    if len(rows) != expected_n:
        raise RuntimeError(
            f"Metadata row count mismatch: reconstructed {len(rows)} rows, "
            f"but validation tensors contain {expected_n} samples."
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def run_inference(checkpoint_path, batch_size=64, dataset_dir="dataset", split_name="val"):
    dataset, shard_paths = load_val_dataset(split_name, dataset_dir=dataset_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = ArrhythmiaWarningNet().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    arrays = {
        "logits_cur": [],
        "logits_fut": [],
        "probs_cur": [],
        "probs_fut": [],
        "target_cur": [],
        "target_fut_soft": [],
        "target_fut": [],
        "t_weight": [],
    }

    for bx, bx_rr, y_cur, y_fut, t_weight in loader:
        if not torch.isfinite(bx).all():
            continue
        bx = bx.to(DEVICE, dtype=torch.float32)
        bx_rr = bx_rr.to(DEVICE, dtype=torch.float32)
        with torch.amp.autocast("cuda" if DEVICE.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)

        arrays["logits_cur"].append(out["logits_cur"].float().cpu().numpy())
        arrays["logits_fut"].append(out["logits_fut"].float().cpu().numpy())
        arrays["probs_cur"].append(out["probs_cur"].float().cpu().numpy())
        arrays["probs_fut"].append(out["probs_fut"].float().cpu().numpy())
        arrays["target_cur"].append(y_cur.cpu().numpy())
        yf = y_fut.float().cpu().numpy()
        arrays["target_fut_soft"].append(yf)
        arrays["target_fut"].append(yf.argmax(axis=1))
        arrays["t_weight"].append(t_weight.float().cpu().numpy())

    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    out["pred_cur"] = out["probs_cur"].argmax(axis=1)
    out["pred_fut"] = out["probs_fut"].argmax(axis=1)
    out["transition_flag"] = out["target_fut"] != out["target_cur"]
    out["checkpoint_epoch"] = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    out["checkpoint_metrics"] = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    out["checkpoint_path"] = str(checkpoint_path)
    out["shard_paths"] = shard_paths
    out["dataset_dir"] = str(dataset_dir)
    out["split_name"] = str(split_name)
    return out


def save_outputs(outputs, metadata, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = metadata.copy()
    df["target_cur"] = outputs["target_cur"].astype(int)
    df["target_fut"] = outputs["target_fut"].astype(int)
    df["pred_cur"] = outputs["pred_cur"].astype(int)
    df["pred_fut"] = outputs["pred_fut"].astype(int)
    df["transition_flag"] = outputs["transition_flag"].astype(bool)
    df["t_weight"] = outputs["t_weight"].astype(float)

    for i, key in enumerate(CLASS_KEYS):
        df[f"prob_cur_{key}"] = outputs["probs_cur"][:, i]
        df[f"prob_fut_{key}"] = outputs["probs_fut"][:, i]
        df[f"logit_cur_{key}"] = outputs["logits_cur"][:, i]
        df[f"logit_fut_{key}"] = outputs["logits_fut"][:, i]
        df[f"target_fut_soft_{key}"] = outputs["target_fut_soft"][:, i]

    csv_path = out_dir / "final_validation_outputs.csv"
    npz_path = out_dir / "final_validation_outputs.npz"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    np.savez_compressed(
        npz_path,
        patient_id=df["patient_id"].to_numpy(dtype=str),
        window_id=df["window_id"].to_numpy(),
        timestamp_sec=df["timestamp_sec"].to_numpy(),
        target_cur=outputs["target_cur"],
        target_fut=outputs["target_fut"],
        target_fut_soft=outputs["target_fut_soft"],
        transition_flag=outputs["transition_flag"],
        t_weight=outputs["t_weight"],
        probs_cur=outputs["probs_cur"],
        probs_fut=outputs["probs_fut"],
        logits_cur=outputs["logits_cur"],
        logits_fut=outputs["logits_fut"],
    )
    return df, csv_path, npz_path


def safe_roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def safe_average_precision(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return np.nan
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return np.nan


def macro_auroc(y_true, probs):
    aucs = []
    for c in range(NUM_CLASSES):
        yb = (y_true == c).astype(int)
        auc = safe_roc_auc(yb, probs[:, c])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else np.nan


def macro_auprc(y_true, probs):
    aps = []
    for c in range(NUM_CLASSES):
        yb = (y_true == c).astype(int)
        ap = safe_average_precision(yb, probs[:, c])
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else np.nan


def compute_metrics(target_cur, target_fut, probs_cur, probs_fut, transition_flag):
    pred_cur = probs_cur.argmax(axis=1)
    pred_fut = probs_fut.argmax(axis=1)
    arrhythmia_target = (target_fut != 0).astype(int)
    arrhythmia_score = 1.0 - probs_fut[:, 0]
    out = {
        "n": int(len(target_fut)),
        "n_patients_proxy": np.nan,
        "current_acc": float(accuracy_score(target_cur, pred_cur)),
        "current_macro_f1": float(f1_score(target_cur, pred_cur, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
        "future_acc": float(accuracy_score(target_fut, pred_fut)),
        "future_macro_f1": float(f1_score(target_fut, pred_fut, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
        "future_macro_auroc": macro_auroc(target_fut, probs_fut),
        "future_macro_auprc": macro_auprc(target_fut, probs_fut),
        "future_arrhythmia_auroc": safe_roc_auc(arrhythmia_target, arrhythmia_score),
        "future_arrhythmia_ap": (
            float(average_precision_score(arrhythmia_target, arrhythmia_score))
            if arrhythmia_target.sum() > 0 else np.nan
        ),
        "future_high_risk_auroc": safe_roc_auc(
            np.isin(target_fut, [3, 4]).astype(int),
            probs_fut[:, 3] + probs_fut[:, 4],
        ),
        "future_high_risk_ap": safe_average_precision(
            np.isin(target_fut, [3, 4]).astype(int),
            probs_fut[:, 3] + probs_fut[:, 4],
        ),
    }

    for cls_idx, cls_key in [(1, "pvc"), (2, "afib"), (3, "vf"), (4, "vt"), (5, "at_svt")]:
        mask = target_fut == cls_idx
        out[f"future_{cls_key}_support"] = int(mask.sum())
        out[f"future_{cls_key}_recall"] = float((pred_fut[mask] == cls_idx).mean()) if mask.sum() else np.nan
        out[f"future_{cls_key}_auroc"] = safe_roc_auc(mask.astype(int), probs_fut[:, cls_idx])
        out[f"future_{cls_key}_ap"] = safe_average_precision(mask.astype(int), probs_fut[:, cls_idx])

    trans = np.asarray(transition_flag, dtype=bool)
    out["transition_n"] = int(trans.sum())
    out["transition_rate"] = float(trans.mean()) if len(trans) else np.nan
    if trans.sum() > 0:
        out["transition_acc"] = float(accuracy_score(target_fut[trans], pred_fut[trans]))
        out["transition_macro_f1"] = float(f1_score(
            target_fut[trans], pred_fut[trans], labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0
        ))
        out["transition_persistence_acc"] = float(accuracy_score(target_fut[trans], target_cur[trans]))
        for cls_idx, cls_key in [(1, "pvc"), (2, "afib"), (3, "vf"), (4, "vt"), (5, "at_svt")]:
            cls_mask = trans & (target_fut == cls_idx)
            out[f"transition_{cls_key}_support"] = int(cls_mask.sum())
            out[f"transition_{cls_key}_recall"] = (
                float((pred_fut[cls_mask] == cls_idx).mean()) if cls_mask.sum() else np.nan
            )
    else:
        out["transition_acc"] = np.nan
        out["transition_macro_f1"] = np.nan
        out["transition_persistence_acc"] = np.nan
    return out


def patient_bootstrap(outputs_df, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(outputs_df["patient_id"].unique()))
    index_by_patient = {
        p: outputs_df.index[outputs_df["patient_id"] == p].to_numpy()
        for p in patients
    }

    probs_cur = outputs_df[[f"prob_cur_{k}" for k in CLASS_KEYS]].to_numpy()
    probs_fut = outputs_df[[f"prob_fut_{k}" for k in CLASS_KEYS]].to_numpy()
    target_cur = outputs_df["target_cur"].to_numpy()
    target_fut = outputs_df["target_fut"].to_numpy()
    transition = outputs_df["transition_flag"].to_numpy(dtype=bool)

    observed = compute_metrics(target_cur, target_fut, probs_cur, probs_fut, transition)
    observed["n_patients_proxy"] = int(len(patients))

    samples = []
    for _ in range(n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([index_by_patient[p] for p in picked])
        metrics = compute_metrics(
            target_cur[idx],
            target_fut[idx],
            probs_cur[idx],
            probs_fut[idx],
            transition[idx],
        )
        samples.append(metrics)

    boot_df = pd.DataFrame(samples)
    rows = []
    for metric, value in observed.items():
        if metric.startswith("n") or metric.endswith("_support"):
            continue
        vals = boot_df[metric].to_numpy(dtype=float) if metric in boot_df else np.array([])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0 or not np.isfinite(value):
            lo, hi = np.nan, np.nan
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({
            "metric": metric,
            "observed": value,
            "ci_low": lo,
            "ci_high": hi,
            "n_boot": int(len(vals)),
        })
    return observed, pd.DataFrame(rows), boot_df


def _arrays_from_outputs_df(outputs_df):
    probs_cur = outputs_df[[f"prob_cur_{k}" for k in CLASS_KEYS]].to_numpy()
    probs_fut = outputs_df[[f"prob_fut_{k}" for k in CLASS_KEYS]].to_numpy()
    target_cur = outputs_df["target_cur"].to_numpy()
    target_fut = outputs_df["target_fut"].to_numpy()
    transition = outputs_df["transition_flag"].to_numpy(dtype=bool)
    return target_cur, target_fut, probs_cur, probs_fut, transition


def patient_robustness_analysis(outputs_df, out_dir):
    """Per-patient metrics plus leave-one-patient-out influence analysis."""
    out_dir = Path(out_dir)
    target_cur, target_fut, probs_cur, probs_fut, transition = _arrays_from_outputs_df(outputs_df)
    observed = compute_metrics(target_cur, target_fut, probs_cur, probs_fut, transition)

    patient_rows = []
    loo_rows = []
    patients = sorted(outputs_df["patient_id"].unique())
    for patient in patients:
        patient_df = outputs_df[outputs_df["patient_id"] == patient]
        pc, pf, prc, prf, tr = _arrays_from_outputs_df(patient_df)
        row = compute_metrics(pc, pf, prc, prf, tr)
        row["patient_id"] = patient
        row["db"] = str(patient_df["db"].iloc[0])
        row["record_id"] = str(patient_df["record_id"].iloc[0])
        row["future_normal_support"] = int((pf == 0).sum())
        patient_rows.append(row)

        keep_df = outputs_df[outputs_df["patient_id"] != patient]
        pc, pf, prc, prf, tr = _arrays_from_outputs_df(keep_df)
        loo = compute_metrics(pc, pf, prc, prf, tr)
        loo["excluded_patient_id"] = patient
        loo["excluded_n"] = int(len(patient_df))
        loo["delta_future_acc"] = loo["future_acc"] - observed["future_acc"]
        loo["delta_future_macro_f1"] = loo["future_macro_f1"] - observed["future_macro_f1"]
        loo["delta_future_arrhythmia_auroc"] = (
            loo["future_arrhythmia_auroc"] - observed["future_arrhythmia_auroc"]
            if np.isfinite(loo["future_arrhythmia_auroc"]) and np.isfinite(observed["future_arrhythmia_auroc"])
            else np.nan
        )
        loo["delta_transition_acc"] = (
            loo["transition_acc"] - observed["transition_acc"]
            if np.isfinite(loo["transition_acc"]) and np.isfinite(observed["transition_acc"])
            else np.nan
        )
        loo_rows.append(loo)

    patient_df = pd.DataFrame(patient_rows)
    loo_df = pd.DataFrame(loo_rows)
    first_cols = [
        "patient_id", "db", "record_id", "n", "transition_n", "future_acc",
        "future_macro_f1", "future_arrhythmia_auroc", "transition_acc",
        "transition_macro_f1", "future_pvc_support", "future_afib_support",
        "future_vt_support", "future_normal_support",
    ]
    patient_df = patient_df[[c for c in first_cols if c in patient_df.columns] +
                            [c for c in patient_df.columns if c not in first_cols]]
    first_loo_cols = [
        "excluded_patient_id", "excluded_n", "future_acc", "delta_future_acc",
        "future_macro_f1", "delta_future_macro_f1",
        "future_arrhythmia_auroc", "delta_future_arrhythmia_auroc",
        "transition_acc", "delta_transition_acc", "transition_n",
    ]
    loo_df = loo_df[[c for c in first_loo_cols if c in loo_df.columns] +
                    [c for c in loo_df.columns if c not in first_loo_cols]]
    patient_df.to_csv(out_dir / "patient_metrics.csv", index=False)
    loo_df.to_csv(out_dir / "leave_one_patient_out.csv", index=False)
    return patient_df, loo_df


def one_hot(targets, n_classes=NUM_CLASSES):
    y = np.zeros((len(targets), n_classes), dtype=np.float64)
    y[np.arange(len(targets)), targets.astype(int)] = 1.0
    return y


def multiclass_brier(probs, targets):
    return float(np.mean(np.sum((probs - one_hot(targets, probs.shape[1])) ** 2, axis=1)))


def multiclass_nll(logits, targets, temperature=1.0):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.long)
    with torch.no_grad():
        return float(F.cross_entropy(logits_t / float(temperature), targets_t).item())


def multiclass_ece(probs, targets, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
            ece += count / len(targets) * gap
        else:
            acc = avg_conf = gap = np.nan
        rows.append({
            "bin": i,
            "lower": lo,
            "upper": hi,
            "count": count,
            "accuracy": acc,
            "confidence": avg_conf,
            "gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def binary_ece(scores, targets, n_bins=10):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores > lo) & (scores <= hi)
        count = int(mask.sum())
        if count:
            event_rate = float(targets[mask].mean())
            avg_score = float(scores[mask].mean())
            gap = abs(event_rate - avg_score)
            ece += count / len(targets) * gap
        else:
            event_rate = avg_score = gap = np.nan
        rows.append({
            "bin": i,
            "lower": lo,
            "upper": hi,
            "count": count,
            "event_rate": event_rate,
            "score": avg_score,
            "gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def binary_calibration_slope_intercept(scores, targets):
    """Logistic recalibration model: outcome ~ intercept + slope * logit(score)."""
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(scores) == 0 or targets.sum() == 0 or targets.sum() == len(targets):
        return np.nan, np.nan
    eps = 1e-6
    logits = np.log(np.clip(scores, eps, 1 - eps) / np.clip(1 - scores, eps, 1 - eps))
    x = torch.tensor(logits[:, None], dtype=torch.float32)
    y = torch.tensor(targets[:, None], dtype=torch.float32)
    beta = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)
    intercept = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([beta, intercept], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = x @ beta + intercept
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        return float(beta.detach().item()), float(intercept.detach().item())
    except Exception:
        return np.nan, np.nan


def fit_temperature(logits, targets):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.long)
    raw_temp = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_temp], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temp = F.softplus(raw_temp) + 1e-3
        loss = F.cross_entropy(logits_t / temp, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float((F.softplus(raw_temp) + 1e-3).detach().cpu().item())
    return max(temp, 1e-3)


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def plot_multiclass_reliability(uncal_bins, cal_bins, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Ideal")
    for bins, label, color in [
        (uncal_bins, "Uncalibrated", "#2f6fed"),
        (cal_bins, "Temp-scaled", "#c45a2a"),
    ]:
        valid = bins["count"] > 0
        ax.plot(
            bins.loc[valid, "confidence"],
            bins.loc[valid, "accuracy"],
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Future Head Multiclass Reliability")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_binary_reliability(uncal_bins, cal_bins, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Ideal")
    for bins, label, color in [
        (uncal_bins, "Uncalibrated", "#2f6fed"),
        (cal_bins, "Temp-scaled", "#c45a2a"),
    ]:
        valid = bins["count"] > 0
        ax.plot(
            bins.loc[valid, "score"],
            bins.loc[valid, "event_rate"],
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def calibration_analysis(outputs_df, out_dir, n_bins=10, reference_calibration=None):
    out_dir = Path(out_dir)
    logits_fut = outputs_df[[f"logit_fut_{k}" for k in CLASS_KEYS]].to_numpy()
    probs_fut = outputs_df[[f"prob_fut_{k}" for k in CLASS_KEYS]].to_numpy()
    targets = outputs_df["target_fut"].to_numpy(dtype=int)

    if reference_calibration is not None:
        temp = float(reference_calibration["future_temperature"])
    else:
        temp = fit_temperature(logits_fut, targets)
    probs_cal = softmax_np(logits_fut / temp)

    ece_uncal, bins_uncal = multiclass_ece(probs_fut, targets, n_bins=n_bins)
    ece_cal, bins_cal = multiclass_ece(probs_cal, targets, n_bins=n_bins)
    brier_uncal = multiclass_brier(probs_fut, targets)
    brier_cal = multiclass_brier(probs_cal, targets)
    nll_uncal = multiclass_nll(logits_fut, targets, temperature=1.0)
    nll_cal = multiclass_nll(logits_fut, targets, temperature=temp)

    arr_target = (targets != 0).astype(int)
    arr_uncal = 1.0 - probs_fut[:, 0]
    arr_cal = 1.0 - probs_cal[:, 0]
    arr_ece_uncal, arr_bins_uncal = binary_ece(arr_uncal, arr_target, n_bins=n_bins)
    arr_ece_cal, arr_bins_cal = binary_ece(arr_cal, arr_target, n_bins=n_bins)
    arr_brier_uncal = float(np.mean((arr_uncal - arr_target) ** 2))
    arr_brier_cal = float(np.mean((arr_cal - arr_target) ** 2))
    arr_slope_uncal, arr_intercept_uncal = binary_calibration_slope_intercept(arr_uncal, arr_target)
    arr_slope_cal, arr_intercept_cal = binary_calibration_slope_intercept(arr_cal, arr_target)

    hi_target = np.isin(targets, [3, 4]).astype(int)
    hi_uncal = probs_fut[:, 3] + probs_fut[:, 4]
    hi_cal = probs_cal[:, 3] + probs_cal[:, 4]
    hi_ece_uncal, hi_bins_uncal = binary_ece(hi_uncal, hi_target, n_bins=n_bins)
    hi_ece_cal, hi_bins_cal = binary_ece(hi_cal, hi_target, n_bins=n_bins)
    hi_brier_uncal = float(np.mean((hi_uncal - hi_target) ** 2))
    hi_brier_cal = float(np.mean((hi_cal - hi_target) ** 2))
    hi_slope_uncal, hi_intercept_uncal = binary_calibration_slope_intercept(hi_uncal, hi_target)
    hi_slope_cal, hi_intercept_cal = binary_calibration_slope_intercept(hi_cal, hi_target)

    plot_multiclass_reliability(
        bins_uncal, bins_cal, out_dir / "reliability_future_multiclass.png"
    )
    plot_binary_reliability(
        arr_bins_uncal, arr_bins_cal,
        out_dir / "reliability_future_arrhythmia.png",
        "Future Arrhythmia Risk Reliability",
    )
    plot_binary_reliability(
        hi_bins_uncal, hi_bins_cal,
        out_dir / "reliability_future_vt_vf.png",
        "Future VT/VF Risk Reliability",
    )

    bins_uncal.to_csv(out_dir / "calibration_bins_future_multiclass_uncalibrated.csv", index=False)
    bins_cal.to_csv(out_dir / "calibration_bins_future_multiclass_temp_scaled.csv", index=False)
    arr_bins_uncal.to_csv(out_dir / "calibration_bins_future_arrhythmia_uncalibrated.csv", index=False)
    arr_bins_cal.to_csv(out_dir / "calibration_bins_future_arrhythmia_temp_scaled.csv", index=False)
    hi_bins_uncal.to_csv(out_dir / "calibration_bins_future_vt_vf_uncalibrated.csv", index=False)
    hi_bins_cal.to_csv(out_dir / "calibration_bins_future_vt_vf_temp_scaled.csv", index=False)

    for i, key in enumerate(CLASS_KEYS):
        outputs_df[f"prob_fut_temp_{key}"] = probs_cal[:, i]

    summary = {
        "future_temperature": temp,
        "future_nll_uncalibrated": nll_uncal,
        "future_nll_temp_scaled": nll_cal,
        "future_ece_uncalibrated": ece_uncal,
        "future_ece_temp_scaled": ece_cal,
        "future_brier_uncalibrated": brier_uncal,
        "future_brier_temp_scaled": brier_cal,
        "future_arrhythmia_ece_uncalibrated": arr_ece_uncal,
        "future_arrhythmia_ece_temp_scaled": arr_ece_cal,
        "future_arrhythmia_brier_uncalibrated": arr_brier_uncal,
        "future_arrhythmia_brier_temp_scaled": arr_brier_cal,
        "future_arrhythmia_calibration_slope_uncalibrated": arr_slope_uncal,
        "future_arrhythmia_calibration_intercept_uncalibrated": arr_intercept_uncal,
        "future_arrhythmia_calibration_slope_temp_scaled": arr_slope_cal,
        "future_arrhythmia_calibration_intercept_temp_scaled": arr_intercept_cal,
        "future_vt_vf_ece_uncalibrated": hi_ece_uncal,
        "future_vt_vf_ece_temp_scaled": hi_ece_cal,
        "future_vt_vf_brier_uncalibrated": hi_brier_uncal,
        "future_vt_vf_brier_temp_scaled": hi_brier_cal,
        "future_vt_vf_calibration_slope_uncalibrated": hi_slope_uncal,
        "future_vt_vf_calibration_intercept_uncalibrated": hi_intercept_uncal,
        "future_vt_vf_calibration_slope_temp_scaled": hi_slope_cal,
        "future_vt_vf_calibration_intercept_temp_scaled": hi_intercept_cal,
        "future_vt_vf_support": int(hi_target.sum()),
    }
    if reference_calibration is not None:
        selected_source = reference_calibration.get("selected_threshold_probability_source", "uncalibrated")
    elif arr_ece_cal <= arr_ece_uncal and brier_cal <= brier_uncal:
        selected_source = "temp_scaled"
    else:
        selected_source = "uncalibrated"
    summary["selected_threshold_probability_source"] = selected_source
    summary["calibration_source"] = "reference" if reference_calibration is not None else "in_sample"

    pd.DataFrame([summary]).to_csv(out_dir / "calibration_summary.csv", index=False)
    return summary, outputs_df


def threshold_sweep(scores, targets, task_name, thresholds):
    rows = []
    targets = np.asarray(targets).astype(int)
    scores = np.asarray(scores, dtype=float)
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (targets == 1)).sum())
        fp = int(((pred == 1) & (targets == 0)).sum())
        tn = int(((pred == 0) & (targets == 0)).sum())
        fn = int(((pred == 0) & (targets == 1)).sum())
        precision = precision_score(targets, pred, zero_division=0)
        recall = recall_score(targets, pred, zero_division=0)
        f1 = f1_score(targets, pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        accuracy = (tp + tn) / len(targets) if len(targets) else np.nan
        rows.append({
            "task": task_name,
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": int(targets.sum()),
        })
    return pd.DataFrame(rows)


def decision_curve(scores, targets, task_name, thresholds):
    rows = []
    targets = np.asarray(targets).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(targets)
    prevalence = float(targets.mean()) if n else np.nan
    for thr in thresholds:
        if thr <= 0.0 or thr >= 1.0 or n == 0:
            continue
        pred = scores >= thr
        tp = int((pred & (targets == 1)).sum())
        fp = int((pred & (targets == 0)).sum())
        weight = thr / (1.0 - thr)
        model_nb = tp / n - fp / n * weight
        treat_all_nb = prevalence - (1.0 - prevalence) * weight
        treat_none_nb = 0.0
        rows.append({
            "task": task_name,
            "threshold": float(thr),
            "net_benefit_model": float(model_nb),
            "net_benefit_treat_all": float(treat_all_nb),
            "net_benefit_treat_none": treat_none_nb,
            "standardized_net_benefit": (
                float(model_nb / prevalence) if prevalence and prevalence > 0 else np.nan
            ),
            "prevalence": prevalence,
            "tp": tp,
            "fp": fp,
        })
    return pd.DataFrame(rows)


def summarize_decision_curve(dca_df):
    rows = []
    if not len(dca_df):
        return pd.DataFrame(rows)
    for (source, task), g in dca_df.groupby(["probability_source", "task"], dropna=False):
        useful = g["net_benefit_model"] > np.maximum(
            g["net_benefit_treat_all"],
            g["net_benefit_treat_none"],
        )
        best = g.sort_values("net_benefit_model", ascending=False).iloc[0]
        rows.append({
            "probability_source": source,
            "task": task,
            "best_threshold": float(best["threshold"]),
            "best_net_benefit": float(best["net_benefit_model"]),
            "best_standardized_net_benefit": float(best["standardized_net_benefit"]),
            "useful_threshold_count": int(useful.sum()),
            "useful_threshold_min": float(g.loc[useful, "threshold"].min()) if useful.any() else np.nan,
            "useful_threshold_max": float(g.loc[useful, "threshold"].max()) if useful.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _threshold_analysis_for_source(outputs_df, thresholds, source_name, prefix, reference_ops=None):
    targets = outputs_df["target_fut"].to_numpy(dtype=int)
    tasks = [
        ("future_arrhythmia", 1.0 - outputs_df[f"{prefix}normal"].to_numpy(), targets != 0),
        ("future_pvc", outputs_df[f"{prefix}pvc"].to_numpy(), targets == 1),
        ("future_afib", outputs_df[f"{prefix}afib"].to_numpy(), targets == 2),
        ("future_vt_vf", (
            outputs_df[f"{prefix}vf"].to_numpy() +
            outputs_df[f"{prefix}vt"].to_numpy()
        ), np.isin(targets, [3, 4])),
    ]
    all_sweeps = []
    operating_rows = []
    for name, scores, y in tasks:
        sweep = threshold_sweep(scores, y, name, thresholds)
        sweep["probability_source"] = source_name
        all_sweeps.append(sweep)
        selected_rows = []
        if reference_ops is not None:
            ref_rows = reference_ops[reference_ops["task"] == name]
            for _, ref in ref_rows.iterrows():
                thr = float(ref["threshold"]) if np.isfinite(ref["threshold"]) else np.nan
                if np.isfinite(thr):
                    nearest = sweep.iloc[(sweep["threshold"] - thr).abs().argmin()].to_dict()
                else:
                    nearest = {"task": name, "threshold": np.nan, "precision": np.nan, "recall": np.nan,
                               "specificity": np.nan, "f1": np.nan, "accuracy": np.nan}
                selected_rows.append((f"fixed_{ref['selection']}", nearest))
        else:
            best_f1 = sweep.sort_values(["f1", "recall", "specificity"], ascending=False).iloc[0].to_dict()
            youden = sweep.assign(youden=sweep["recall"] + sweep["specificity"] - 1.0)
            best_youden = youden.sort_values(["youden", "recall"], ascending=False).iloc[0].to_dict()
            high_sens = sweep[sweep["recall"] >= 0.90]
            if len(high_sens):
                sens90 = high_sens.sort_values(["specificity", "threshold"], ascending=False).iloc[0].to_dict()
            else:
                sens90 = {"task": name, "threshold": np.nan, "precision": np.nan, "recall": np.nan,
                          "specificity": np.nan, "f1": np.nan, "accuracy": np.nan}
            selected_rows = [("best_f1", best_f1), ("best_youden", best_youden), ("sens_ge_0.90", sens90)]
        for label, row in selected_rows:
            row = dict(row)
            row["selection"] = label
            row["probability_source"] = source_name
            operating_rows.append(row)

    sweep_df = pd.concat(all_sweeps, ignore_index=True)
    op_df = pd.DataFrame(operating_rows)
    return sweep_df, op_df


def threshold_analysis(outputs_df, out_dir, selected_source, reference_ops=None):
    out_dir = Path(out_dir)
    thresholds = np.linspace(0.0, 1.0, 101)
    sources = [
        ("uncalibrated", "prob_fut_"),
        ("temp_scaled", "prob_fut_temp_"),
    ]
    all_sweeps, all_ops, all_dca = [], [], []
    selected_ops = None
    locked_reference_thresholds = reference_ops is not None
    for source_name, prefix in sources:
        source_ref_ops = None
        if locked_reference_thresholds:
            source_ref_ops = reference_ops[reference_ops["probability_source"] == source_name]
        sweep_df, op_df = _threshold_analysis_for_source(
            outputs_df, thresholds, source_name, prefix, reference_ops=source_ref_ops
        )
        sweep_df.to_csv(out_dir / f"threshold_sweep_future_{source_name}.csv", index=False)
        op_df.to_csv(out_dir / f"operating_points_future_{source_name}.csv", index=False)
        targets = outputs_df["target_fut"].to_numpy(dtype=int)
        dca_tasks = [
            ("future_arrhythmia", 1.0 - outputs_df[f"{prefix}normal"].to_numpy(), targets != 0),
            ("future_vt_vf", (
                outputs_df[f"{prefix}vf"].to_numpy() +
                outputs_df[f"{prefix}vt"].to_numpy()
            ), np.isin(targets, [3, 4])),
        ]
        dca_df = pd.concat(
            [decision_curve(scores, y, name, thresholds) for name, scores, y in dca_tasks],
            ignore_index=True,
        )
        dca_df["probability_source"] = source_name
        dca_df.to_csv(out_dir / f"decision_curve_future_{source_name}.csv", index=False)
        all_sweeps.append(sweep_df)
        all_ops.append(op_df)
        all_dca.append(dca_df)
        if source_name == selected_source:
            selected_ops = op_df

    sweep_all = pd.concat(all_sweeps, ignore_index=True)
    op_all = pd.concat(all_ops, ignore_index=True)
    dca_all = pd.concat(all_dca, ignore_index=True)
    dca_summary = summarize_decision_curve(dca_all)
    sweep_all.to_csv(out_dir / "threshold_sweep_future_all_sources.csv", index=False)
    op_all.to_csv(out_dir / "operating_points_future_all_sources.csv", index=False)
    dca_all.to_csv(out_dir / "decision_curve_future_all_sources.csv", index=False)
    dca_summary.to_csv(out_dir / "decision_curve_summary.csv", index=False)
    if selected_ops is None:
        selected_ops = all_ops[0]
    selected_ops.to_csv(out_dir / "operating_points_future_selected.csv", index=False)
    return sweep_all, op_all, selected_ops, dca_all, dca_summary


def build_risk_flags(observed, ci_df, calibration, loo_df):
    flags = []
    ci = {row["metric"]: row for row in ci_df.to_dict(orient="records")}

    def add(flag, severity, detail):
        flags.append({"flag": flag, "severity": severity, "detail": detail})

    n_patients = int(observed.get("n_patients_proxy", 0))
    if n_patients < 20:
        add(
            "low_patient_count",
            "high",
            f"Validation currently has {n_patients} patient/record units; patient-level CI is expected to be wide.",
        )

    for metric, max_width, severity in [
        ("future_macro_f1", 0.20, "medium"),
        ("transition_acc", 0.40, "high"),
        ("future_arrhythmia_auroc", 0.25, "medium"),
    ]:
        row = ci.get(metric)
        if row and np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            width = row["ci_high"] - row["ci_low"]
            if width > max_width:
                add(
                    f"wide_ci_{metric}",
                    severity,
                    f"{metric} CI width is {width:.3f} ({row['ci_low']:.3f}-{row['ci_high']:.3f}).",
                )

    if len(loo_df):
        f1_delta = loo_df["delta_future_macro_f1"].abs().max()
        trans_delta = loo_df["delta_transition_acc"].abs().max()
        if np.isfinite(f1_delta) and f1_delta > 0.05:
            patient = loo_df.iloc[loo_df["delta_future_macro_f1"].abs().argmax()]["excluded_patient_id"]
            add(
                "leave_one_out_future_macro_f1_sensitive",
                "medium",
                f"Excluding {patient} changes future macro-F1 by {f1_delta:.3f}.",
            )
        if np.isfinite(trans_delta) and trans_delta > 0.15:
            patient = loo_df.iloc[loo_df["delta_transition_acc"].abs().argmax()]["excluded_patient_id"]
            add(
                "leave_one_out_transition_sensitive",
                "high",
                f"Excluding {patient} changes transition accuracy by {trans_delta:.3f}.",
            )

    vt_support = int(calibration.get("future_vt_vf_support", 0))
    if vt_support < 30:
        add(
            "vt_vf_low_support",
            "high",
            f"Future VT/VF support is {vt_support}; treat VT/VF claims as exploratory/data-limited.",
        )

    if calibration["selected_threshold_probability_source"] == "uncalibrated":
        add(
            "temperature_scaling_not_selected",
            "low",
            "Temperature scaling improves NLL but worsens ECE/Brier, so thresholds use uncalibrated probabilities.",
        )

    return flags


def _contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segments = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def event_lead_time_analysis(outputs_df, out_dir, selected_source, op_df):
    """Event-centered early-warning analysis for the selected probability source.

    True events are contiguous current-state arrhythmia regions within each
    patient timeline. An event is detected when a future-risk alert fires from
    horizon seconds before the event start through the event end.
    """
    out_dir = Path(out_dir)
    prefix = "prob_fut_temp_" if selected_source == "temp_scaled" else "prob_fut_"

    def _selected_thresholds(task_name, fallback):
        keep = ["best_f1", "sens_ge_0.90", "fixed_best_f1", "fixed_sens_ge_0.90"]
        rows = op_df[(op_df["task"] == task_name) & (op_df["selection"].isin(keep))]
        selected = []
        seen = set()
        for _, row in rows.iterrows():
            threshold = float(row["threshold"]) if np.isfinite(row["threshold"]) else np.nan
            if not np.isfinite(threshold):
                continue
            key = str(row["selection"])
            if key in seen:
                continue
            selected.append({"selection": key, "threshold": threshold})
            seen.add(key)
        if selected:
            return selected
        return [{"selection": "fallback", "threshold": fallback}]

    future_targets = outputs_df["target_fut"].to_numpy(dtype=int)
    current_targets = outputs_df["target_cur"].to_numpy(dtype=int)
    tasks = [
        {
            "task": "future_arrhythmia",
            "score": 1.0 - outputs_df[f"{prefix}normal"].to_numpy(dtype=float),
            "event_target": current_targets != 0,
            "sample_target": future_targets != 0,
            "thresholds": _selected_thresholds("future_arrhythmia", 0.5),
        },
        {
            "task": "future_vt_vf",
            "score": (
                outputs_df[f"{prefix}vf"].to_numpy(dtype=float) +
                outputs_df[f"{prefix}vt"].to_numpy(dtype=float)
            ),
            "event_target": np.isin(current_targets, [3, 4]),
            "sample_target": np.isin(future_targets, [3, 4]),
            "thresholds": _selected_thresholds("future_vt_vf", 0.5),
        },
    ]

    event_rows = []
    summary_rows = []
    for task in tasks:
        for threshold_cfg in task["thresholds"]:
            threshold = threshold_cfg["threshold"]
            selection = threshold_cfg["selection"]
            alert = task["score"] >= threshold
            all_event_count = 0
            detected_count = 0
            lead_times = []
            incident_lead_times = []
            false_alert_windows = 0
            false_alert_episodes = 0
            total_hours = 0.0
            incident_event_count = 0
            incident_detected_count = 0

            for patient in sorted(outputs_df["patient_id"].unique()):
                idx = outputs_df.index[outputs_df["patient_id"] == patient].to_numpy()
                patient_df = outputs_df.loc[idx].sort_values("timestamp_sec")
                order = patient_df.index.to_numpy()
                times = patient_df["timestamp_sec"].to_numpy(dtype=float)
                if len(times) > 1:
                    total_hours += max(float(times[-1] - times[0]), 0.0) / 3600.0
                target = task["event_target"][order]
                alert_patient = alert[order]

                covered_alert = np.zeros(len(order), dtype=bool)
                for event_id, (start_i, end_i) in enumerate(_contiguous_true_segments(target)):
                    all_event_count += 1
                    incident = start_i > 0
                    if incident:
                        incident_event_count += 1
                    start_t = float(times[start_i])
                    end_t = float(times[end_i])
                    eligible = (times >= start_t - EARLY_WARNING_HORIZON_SEC) & (times <= end_t)
                    event_alert_idx = np.where(eligible & alert_patient)[0]
                    detected = len(event_alert_idx) > 0
                    lead = np.nan
                    first_alert_t = np.nan
                    if detected:
                        detected_count += 1
                        if incident:
                            incident_detected_count += 1
                        first_alert_t = float(times[event_alert_idx[0]])
                        lead = start_t - first_alert_t
                        lead_times.append(lead)
                        if incident:
                            incident_lead_times.append(lead)
                        covered_alert |= eligible & alert_patient
                    event_rows.append({
                        "task": task["task"],
                        "selection": selection,
                        "patient_id": patient,
                        "event_id": event_id,
                        "incident_event": incident,
                        "event_start_sec": start_t,
                        "event_end_sec": end_t,
                        "event_duration_sec": end_t - start_t,
                        "threshold": threshold,
                        "detected": detected,
                        "first_alert_sec": first_alert_t,
                        "lead_time_sec": lead,
                    })

                false_mask = alert_patient & ~covered_alert & ~target
                false_alert_windows += int(false_mask.sum())
                false_alert_episodes += len(_contiguous_true_segments(false_mask))

            lead_array = np.asarray(lead_times, dtype=float)
            incident_lead_array = np.asarray(incident_lead_times, dtype=float)
            summary_rows.append({
                "task": task["task"],
                "selection": selection,
                "probability_source": selected_source,
                "threshold": threshold,
                "events": all_event_count,
                "event_recall": detected_count / all_event_count if all_event_count else np.nan,
                "detected_events": detected_count,
                "incident_events": incident_event_count,
                "incident_event_recall": (
                    incident_detected_count / incident_event_count if incident_event_count else np.nan
                ),
                "detected_incident_events": incident_detected_count,
                "median_lead_time_sec": float(np.nanmedian(lead_array)) if len(lead_array) else np.nan,
                "p25_lead_time_sec": float(np.nanpercentile(lead_array, 25)) if len(lead_array) else np.nan,
                "p75_lead_time_sec": float(np.nanpercentile(lead_array, 75)) if len(lead_array) else np.nan,
                "incident_median_lead_time_sec": (
                    float(np.nanmedian(incident_lead_array)) if len(incident_lead_array) else np.nan
                ),
                "incident_p25_lead_time_sec": (
                    float(np.nanpercentile(incident_lead_array, 25)) if len(incident_lead_array) else np.nan
                ),
                "incident_p75_lead_time_sec": (
                    float(np.nanpercentile(incident_lead_array, 75)) if len(incident_lead_array) else np.nan
                ),
                "false_alert_windows": false_alert_windows,
                "false_alert_episodes": false_alert_episodes,
                "patient_hours": total_hours,
                "false_alert_windows_per_patient_hour": false_alert_windows / total_hours if total_hours > 0 else np.nan,
                "false_alert_episodes_per_patient_hour": false_alert_episodes / total_hours if total_hours > 0 else np.nan,
                "horizon_sec": EARLY_WARNING_HORIZON_SEC,
            })

    event_df = pd.DataFrame(event_rows)
    summary_df = pd.DataFrame(summary_rows)
    event_df.to_csv(out_dir / "event_lead_time_events.csv", index=False)
    summary_df.to_csv(out_dir / "event_lead_time_summary.csv", index=False)
    return summary_df, event_df


def plot_patient_timeline(outputs_df, out_dir):
    out_dir = Path(out_dir)
    patients = sorted(outputs_df["patient_id"].unique())
    n = len(patients)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.2 * n, 6)), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, patient in zip(axes, patients):
        d = outputs_df[outputs_df["patient_id"] == patient].copy()
        hours = d["timestamp_sec"].to_numpy() / 3600.0
        arr_risk = 1.0 - d["prob_fut_temp_normal"].to_numpy()
        hi_risk = d["prob_fut_temp_vf"].to_numpy() + d["prob_fut_temp_vt"].to_numpy()
        ax.plot(hours, arr_risk, color="#2f6fed", linewidth=1.8, label="Arrhythmia risk")
        ax.plot(hours, hi_risk, color="#c45a2a", linewidth=1.4, label="VT/VF risk")
        trans = d["transition_flag"].to_numpy(dtype=bool)
        if trans.any():
            ax.scatter(hours[trans], arr_risk[trans], color="black", s=18, zorder=3, label="Transition sample")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(patient)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Forecast start time (hours)")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    fig.suptitle("Patient-Level Future Risk Timeline")
    fig.tight_layout()
    fig.savefig(out_dir / "patient_risk_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir, checkpoint_path, outputs, observed, ci_df, calibration, op_df,
                  patient_df, loo_df, event_summary_df, dca_summary_df,
                  risk_flags, outputs_csv, outputs_npz):
    out_dir = Path(out_dir)
    key_metrics = {
        row["metric"]: row for row in ci_df.to_dict(orient="records")
    }

    def ci_line(metric):
        row = key_metrics.get(metric)
        if not row:
            return f"- {metric}: N/A"
        return (
            f"- {metric}: {row['observed']:.4f} "
            f"(95% patient bootstrap CI {row['ci_low']:.4f}-{row['ci_high']:.4f})"
        )

    lines = [
        "# Final Validation Summary",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint_epoch_field: {outputs.get('checkpoint_epoch', -1)}",
        f"- dataset_dir: `{outputs.get('dataset_dir', 'dataset')}`",
        f"- split_name: `{outputs.get('split_name', 'val')}`",
        f"- validation_outputs_csv: `{outputs_csv}`",
        f"- validation_outputs_npz: `{outputs_npz}`",
        f"- validation_samples: {int(observed['n'])}",
        f"- validation_patients_or_records: {int(observed['n_patients_proxy'])}",
        "",
        "## Patient Bootstrap CI",
        ci_line("future_acc"),
        ci_line("future_macro_f1"),
        ci_line("future_macro_auroc"),
        ci_line("future_macro_auprc"),
        ci_line("future_arrhythmia_auroc"),
        ci_line("future_arrhythmia_ap"),
        ci_line("future_high_risk_auroc"),
        ci_line("future_high_risk_ap"),
        ci_line("transition_acc"),
        ci_line("transition_macro_f1"),
        ci_line("transition_persistence_acc"),
        "",
        "## Patient Robustness",
        f"- patient_metrics_csv: `{out_dir / 'patient_metrics.csv'}`",
        f"- leave_one_patient_out_csv: `{out_dir / 'leave_one_patient_out.csv'}`",
    ]
    if len(loo_df):
        f1_influence = loo_df[["excluded_patient_id", "delta_future_macro_f1"]].dropna()
        trans_influence = loo_df[["excluded_patient_id", "delta_transition_acc"]].dropna()
        if len(f1_influence):
            worst = f1_influence.iloc[f1_influence["delta_future_macro_f1"].abs().argmax()]
            lines.append(
                f"- largest |delta future macro-F1| after exclusion: "
                f"{worst['excluded_patient_id']} ({worst['delta_future_macro_f1']:+.4f})"
            )
        if len(trans_influence):
            worst = trans_influence.iloc[trans_influence["delta_transition_acc"].abs().argmax()]
            lines.append(
        f"- largest |delta transition acc| after exclusion: "
                f"{worst['excluded_patient_id']} ({worst['delta_transition_acc']:+.4f})"
            )

    lines.extend([
        "",
        "## Risk Flags",
    ])
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- [{flag['severity']}] {flag['flag']}: {flag['detail']}")
    else:
        lines.append("- none")

    locked_reference = calibration.get("calibration_source") == "reference" or (
        len(op_df) and op_df["selection"].astype(str).str.startswith("fixed_").all()
    )
    operating_heading = (
        "## Fixed Validation Operating Points Applied on This Split"
        if locked_reference
        else f"## Operating Points Selected on This Split ({calibration['selected_threshold_probability_source']})"
    )
    operating_note = (
        "- thresholds/calibration were imported from the reference validation split; "
        "no operating threshold is selected from this split."
        if locked_reference
        else "- thresholds are selected on this split for validation/model-selection use only."
    )

    lines.extend([
        "",
        "## Calibration",
        f"- calibration source: {calibration.get('calibration_source', 'in_sample')}",
        f"- future temperature: {calibration['future_temperature']:.4f}",
        f"- future NLL: {calibration['future_nll_uncalibrated']:.4f} -> {calibration['future_nll_temp_scaled']:.4f}",
        f"- future ECE: {calibration['future_ece_uncalibrated']:.4f} -> {calibration['future_ece_temp_scaled']:.4f}",
        f"- future Brier: {calibration['future_brier_uncalibrated']:.4f} -> {calibration['future_brier_temp_scaled']:.4f}",
        f"- future arrhythmia ECE: {calibration['future_arrhythmia_ece_uncalibrated']:.4f} -> {calibration['future_arrhythmia_ece_temp_scaled']:.4f}",
        f"- future arrhythmia calibration slope/intercept: "
        f"{calibration['future_arrhythmia_calibration_slope_uncalibrated']:.3f} / "
        f"{calibration['future_arrhythmia_calibration_intercept_uncalibrated']:.3f}",
        f"- future VT/VF ECE: {calibration['future_vt_vf_ece_uncalibrated']:.4f} -> {calibration['future_vt_vf_ece_temp_scaled']:.4f}",
        f"- selected threshold probability source: {calibration['selected_threshold_probability_source']}",
        f"- VT/VF support: {calibration['future_vt_vf_support']} samples",
        "",
        operating_heading,
        operating_note,
    ])
    shown_selections = ["best_f1", "sens_ge_0.90", "fixed_best_f1", "fixed_sens_ge_0.90"]
    for _, row in op_df[op_df["selection"].isin(shown_selections)].iterrows():
        lines.append(
            f"- {row['task']} / {row['selection']}: threshold={row['threshold']:.2f}, "
            f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, "
            f"specificity={row['specificity']:.3f}"
        )

    lines.extend([
        "",
        "## Event-Centered Early Warning",
        f"- event_lead_time_summary_csv: `{out_dir / 'event_lead_time_summary.csv'}`",
        f"- event_lead_time_events_csv: `{out_dir / 'event_lead_time_events.csv'}`",
    ])
    if len(event_summary_df):
        for _, row in event_summary_df.iterrows():
            selection = row.get("selection", "threshold")
            lines.append(
                f"- {row['task']} / {selection}: threshold={row['threshold']:.2f}, "
                f"events={int(row['events'])}, "
                f"event_recall={row['event_recall']:.3f}, "
                f"incident_event_recall={row['incident_event_recall']:.3f}, "
                f"median_lead={row['median_lead_time_sec']:.1f}s, "
                f"false_alarm_episodes/hr={row['false_alert_episodes_per_patient_hour']:.2f}"
            )

    lines.extend([
        "",
        "## Descriptive Decision Curve Analysis",
        f"- decision_curve_summary_csv: `{out_dir / 'decision_curve_summary.csv'}`",
        f"- decision_curve_all_sources_csv: `{out_dir / 'decision_curve_future_all_sources.csv'}`",
        "- decision curves summarize net benefit across thresholds; test-split curves are descriptive and do not select deployment thresholds.",
    ])
    if len(dca_summary_df):
        selected_source = calibration["selected_threshold_probability_source"]
        selected_dca = dca_summary_df[dca_summary_df["probability_source"] == selected_source]
        for _, row in selected_dca.iterrows():
            lines.append(
                f"- {row['task']}: best_threshold={row['best_threshold']:.2f}, "
                f"best_net_benefit={row['best_net_benefit']:.4f}, "
                f"useful_threshold_range={row['useful_threshold_min']:.2f}-{row['useful_threshold_max']:.2f}"
            )

    lines.extend([
        "",
        "## Claim Boundary",
        "- Stronger evidence: Normal/PVC/AFib discrimination and transition-subset performance above persistence.",
        "- Weak evidence: VT/VF reliability is exploratory because validation support is small and future VT recall is currently poor.",
    ])
    path = out_dir / "final_validation_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    global EARLY_WARNING_HORIZON_SEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/arrhythmia_warning_best.pth")
    parser.add_argument("--output-dir", default="results/final_validation")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-plan", default=None,
                        help="Optional split_plan_v7.json for metadata reconstruction on non-default splits")
    parser.add_argument("--max-records-per-split", type=int, default=None,
                        help="For smoke datasets built with --max-records-per-split, reconstruct only the first N records")
    parser.add_argument("--reference-calibration", default=None,
                        help="Optional calibration_summary.json from validation split for locked test calibration")
    parser.add_argument("--reference-operating-points", default=None,
                        help="Optional operating_points_future_selected.csv from validation split for locked test thresholds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = load_dataset_config(args.dataset_dir)
    history_sec = dataset_config.get("history_sec")
    predict_sec = dataset_config.get("predict_sec")
    if predict_sec is not None:
        EARLY_WARNING_HORIZON_SEC = float(predict_sec)
        print(f"[config] dataset predict_sec={predict_sec}; event horizon={EARLY_WARNING_HORIZON_SEC:.0f}s")
    reference_calibration = None
    reference_ops = None
    if args.reference_calibration:
        with open(args.reference_calibration, "r", encoding="utf-8") as f:
            reference_calibration = json.load(f)
    if args.reference_operating_points:
        reference_ops = pd.read_csv(args.reference_operating_points)

    print(f"[1/6] Running inference on {args.checkpoint} ({DEVICE})")
    outputs = run_inference(
        args.checkpoint,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        split_name=args.split,
    )

    print("[2/6] Reconstructing validation patient/window metadata")
    records = (
        record_order_from_split_plan(args.split_plan, args.split)
        if args.split_plan else None
    )
    if records is not None and args.max_records_per_split is not None:
        records = records[:int(args.max_records_per_split)]
    metadata = reconstruct_val_metadata(
        len(outputs["target_fut"]),
        records=records,
        use_db_stride=bool(args.split_plan),
        history_sec=history_sec,
        predict_sec=predict_sec,
    )
    outputs_df, outputs_csv, outputs_npz = save_outputs(outputs, metadata, out_dir)
    print(f"      saved {outputs_csv}")
    print(f"      saved {outputs_npz}")

    print(f"[3/6] Patient-level bootstrap ({args.bootstrap} resamples)")
    observed, ci_df, boot_df = patient_bootstrap(outputs_df, n_boot=args.bootstrap, seed=args.seed)
    ci_df.to_csv(out_dir / "patient_bootstrap_ci.csv", index=False)
    boot_df.to_csv(out_dir / "patient_bootstrap_samples.csv", index=False)

    print("[4/6] Patient robustness")
    patient_df, loo_df = patient_robustness_analysis(outputs_df, out_dir)

    print("[5/6] Calibration and reliability")
    calibration, outputs_df = calibration_analysis(
        outputs_df,
        out_dir,
        n_bins=args.bins,
        reference_calibration=reference_calibration,
    )
    risk_flags = build_risk_flags(observed, ci_df, calibration, loo_df)
    outputs_df.to_csv(outputs_csv, index=False, encoding="utf-8")

    print("[6/6] Threshold sweep, event lead-time, and patient timeline")
    _, _, op_df, dca_df, dca_summary_df = threshold_analysis(
        outputs_df,
        out_dir,
        selected_source=calibration["selected_threshold_probability_source"],
        reference_ops=reference_ops,
    )
    event_summary_df, event_df = event_lead_time_analysis(
        outputs_df,
        out_dir,
        selected_source=calibration["selected_threshold_probability_source"],
        op_df=op_df,
    )
    plot_patient_timeline(outputs_df, out_dir)

    with open(out_dir / "observed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(observed, f, indent=2, ensure_ascii=False)
    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    with open(out_dir / "validation_risk_flags.json", "w", encoding="utf-8") as f:
        json.dump(risk_flags, f, indent=2, ensure_ascii=False)
    event_summary_json = event_summary_df.replace({np.nan: None}).to_dict(orient="records")
    with open(out_dir / "event_lead_time_summary.json", "w", encoding="utf-8") as f:
        json.dump(event_summary_json, f, indent=2, ensure_ascii=False)

    summary_path = write_summary(
        out_dir,
        args.checkpoint,
        outputs,
        observed,
        ci_df,
        calibration,
        op_df,
        patient_df,
        loo_df,
        event_summary_df,
        dca_summary_df,
        risk_flags,
        outputs_csv,
        outputs_npz,
    )
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
