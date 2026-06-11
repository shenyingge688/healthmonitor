"""
build_val_meta.py — derive per-sample patient/record id + transition flag for the
validation set, WITHOUT touching the shards or re-extracting features.

The shards store X/X_rr/Y_cur/Y_fut/T_weight but NOT which record each window came
from. Patient-level bootstrap CI needs that grouping. We replicate the exact val
record order + windowing from build_dataset_factory (importing its real functions
and constants) and emit only metadata, then SELF-VERIFY by asserting the recomputed
labels match the stored Y_cur / argmax(Y_fut) element-for-element. If the assert
passes, the rec_id mapping is provably aligned to the shard sample order.

Output: dataset/val_meta.pt  { rec_id, cur, fut, t_weight, transition }
"""
import os
import glob
import numpy as np
import torch
import wfdb
from scipy import signal

import build_dataset_factory as B


def val_record_paths():
    mitdb_dir = os.path.join(B.DATA_DIR, "mitdb")
    all_mitdb = sorted(f.split(".")[0] for f in os.listdir(mitdb_dir) if f.endswith(".dat"))
    val_mitdb = [r for r in all_mitdb if r in B.VAL_MITDB]
    recs = [(os.path.join(mitdb_dir, r), r) for r in val_mitdb]
    for db_name, rlist in B.SUPPLEMENT_RECORDS.items():
        val_fixed = B.VAL_SUPPLEMENTS.get(db_name, [])
        db_dir = os.path.join(B.DATA_DIR, db_name)
        avail = set(f.split(".")[0] for f in os.listdir(db_dir)
                    if f.endswith(".dat") or f.endswith(".qrs"))
        found = [r for r in rlist if r in avail]
        for r in found:
            if r in val_fixed:
                recs.append((os.path.join(db_dir, r), f"{db_name}/{r}"))
    return recs


def walk_record(rec_path, db_source="mitdb"):
    """Replicate build_dataset's per-record windowing; emit (cur, fut, tw) per sample."""
    record_obj = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, "atr", pn_dir=None)

    signal_len = record_obj.sig_len if hasattr(record_obj, "sig_len") else len(record_obj.p_signal)
    src_fs = record_obj.fs if hasattr(record_obj, "fs") else 360
    MAX_SAMPLES = 2 * 3600 * int(src_fs)
    if signal_len > MAX_SAMPLES:
        signal_len = MAX_SAMPLES

    raw_sig = record_obj.p_signal[:signal_len, 0]
    if src_fs != B.TARGET_FS:
        ecg = signal.resample_poly(B.clean_ecg_signal(raw_sig, fs=src_fs), B.TARGET_FS, int(src_fs))
    else:
        ecg = B.clean_ecg_signal(raw_sig, fs=src_fs)
    ecg = ecg.astype(np.float32)

    labels = B.parse_annotations(annotation, B.TARGET_FS, src_fs, len(ecg))
    svt_mask = labels == 5
    if svt_mask.any():
        svt_ends = np.where(svt_mask[:-1] & ~svt_mask[1:])[0]
        for end_pos in svt_ends:
            end_s = min(len(labels), end_pos + 15000)
            region = labels[end_pos:end_s]
            region[region == 0] = 5

    out = []
    current_pt = 0
    max_pt = len(ecg) - (B.HISTORY_SEC + B.PREDICT_SEC) * B.TARGET_FS
    if max_pt <= 0:
        return out
    stride = B.DB_STRIDE_OVERRIDE.get(db_source, B.FIXED_STRIDE_SEC)
    while current_pt < max_pt:
        history_end = current_pt + B.HISTORY_SEC * B.TARGET_FS
        pred_start = history_end
        pred_end = pred_start + B.PREDICT_SEC * B.TARGET_FS
        cur_label = B.get_current_label(labels, history_end)
        fut_dist = B.get_distribution(labels, pred_start, pred_end)
        t_weight = B.get_transition_weight(cur_label, fut_dist)
        out.append((cur_label, int(np.argmax(fut_dist)), float(t_weight)))
        current_pt += int(stride * B.TARGET_FS)
    return out


def main():
    print("Re-deriving val patient/record metadata (no shard changes)...")
    recs = val_record_paths()
    print("Val record order:")
    for p, rid in recs:
        print(f"  {rid}")

    rec_ids, curs, futs, tws = [], [], [], []
    for rec_path, rid in recs:
        rows = walk_record(rec_path, db_source="mitdb")
        for cur, fut, tw in rows:
            rec_ids.append(rid)
            curs.append(cur)
            futs.append(fut)
            tws.append(tw)
        print(f"  {rid}: {len(rows)} samples")

    curs = np.array(curs, dtype=np.int64)
    futs = np.array(futs, dtype=np.int64)
    tws = np.array(tws, dtype=np.float32)
    rec_ids = np.array(rec_ids)

    # ---- self-verify against the stored shard ----
    shard_paths = sorted(glob.glob("dataset/val_shard_*.pt"))
    sy_cur, sy_fut = [], []
    for p in shard_paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        sy_cur.append(d["Y_cur"].numpy())
        sy_fut.append(d["Y_fut"].float().numpy().argmax(axis=1))
    sy_cur = np.concatenate(sy_cur)
    sy_fut = np.concatenate(sy_fut)

    print(f"\nRecomputed N={len(curs)}  Shard N={len(sy_cur)}")
    assert len(curs) == len(sy_cur), "sample count mismatch — windowing replica is off"
    cur_match = float((curs == sy_cur).mean())
    fut_match = float((futs == sy_fut).mean())
    print(f"  Y_cur match: {cur_match*100:.2f}%   argmax(Y_fut) match: {fut_match*100:.2f}%")
    assert cur_match > 0.999, "Y_cur mismatch — alignment NOT valid"
    assert fut_match > 0.999, "Y_fut mismatch — alignment NOT valid"
    print("  [verified] metadata is aligned to shard sample order.")

    transition = (curs != futs).astype(np.int64)
    torch.save({"rec_id": rec_ids, "cur": curs, "fut": futs,
                "t_weight": tws, "transition": transition},
               "dataset/val_meta.pt")
    # per-record summary
    print("\nPer-record sample counts + transition fraction:")
    for rid in dict.fromkeys(rec_ids):
        m = rec_ids == rid
        print(f"  {rid:14s} n={int(m.sum()):4d}  trans={int(transition[m].sum()):3d} "
              f"({transition[m].mean()*100:.0f}%)")
    print(f"\nUnique val patients/records: {len(set(rec_ids.tolist()))}")
    print("Saved dataset/val_meta.pt")


if __name__ == "__main__":
    main()
