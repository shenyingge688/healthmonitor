"""
test_demo_samples.py
Offline replay of candidate demo records through the served checkpoint, so we
can judge each example WITHOUT spinning up FastAPI + Streamlit + a browser.

Runs on CPU (CUDA hidden) so it never contends with a training run on the GPU.
Reports, for scan points across the playback window, the current-head and
future-head predictions + the probability of the case's target class.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU, never touch the training GPU

import numpy as np
import torch
import wfdb
from scipy import signal as scisig
from scipy.signal import butter, filtfilt

import main  # imports build_window_sequence, model (CPU), device, constants
from constants import API_BUFFER_SIZE, TARGET_FS

ENG = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]


def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype="band")
    return filtfilt(b, a, data)


def load_sim_data(rec_id, db):
    rec_path = os.path.join("data", db, rec_id)
    if not os.path.exists(rec_path + ".dat"):
        return None
    if db == "mitdb":
        record = wfdb.rdrecord(rec_path, sampto=int(30 * 60 * 360))
        src_fs = 360
    elif db == "vfdb":
        record = wfdb.rdrecord(rec_path)
        src_fs = getattr(record, "fs", 250) or 250
    else:
        record = wfdb.rdrecord(rec_path, sampto=int(30 * 60 * 250))
        src_fs = getattr(record, "fs", 250) or 250
    sig = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    if src_fs == 250:
        return sig.astype(np.float32)
    if db == "mitdb":
        sig = clean_ecg_signal(sig, fs=src_fs)
    return scisig.resample_poly(sig, 250, src_fs).astype(np.float32)


@torch.no_grad()
def predict_at(data, current_pts):
    buf = data[max(0, current_pts - API_BUFFER_SIZE):current_pts]
    if len(buf) < API_BUFFER_SIZE:
        buf = np.pad(buf, (API_BUFFER_SIZE - len(buf), 0), "constant")
    ws, rs = main.build_window_sequence(buf)
    bx = torch.from_numpy(ws).unsqueeze(0).to(main.device)
    bx_rr = torch.from_numpy(rs).unsqueeze(0).to(main.device)
    out = main.run_ensemble(bx, bx_rr)
    pf = out["probs_fut"][0].cpu().float().numpy()
    pc = out["probs_cur"][0].cpu().float().numpy()
    return pc, pf


def replay(rec, db, start_min, target_cls, end_min=30.0, step_sec=20):
    data = load_sim_data(rec, db)
    if data is None:
        print(f"  [missing] {db}/{rec}")
        return
    fs = TARGET_FS
    print(f"\n=== {db}/{rec}  target={ENG[target_cls]}  start={start_min}min ===")
    print(f"{'t(min)':>7} | cur (p)        | fut (p)        | p_tgt_cur p_tgt_fut")
    t = start_min
    cur_hits = fut_hits = n = 0
    while t <= end_min:
        cp = int(t * 60 * fs)
        if cp >= len(data):
            break
        pc, pf = predict_at(data, cp)
        ci, fi = int(pc.argmax()), int(pf.argmax())
        n += 1
        cur_hits += (ci == target_cls)
        fut_hits += (fi == target_cls)
        print(f"{t:7.1f} | {ENG[ci]:<8}{pc[ci]:.2f} | {ENG[fi]:<8}{pf[fi]:.2f} | "
              f"{pc[target_cls]:9.2f} {pf[target_cls]:9.2f}")
        t += step_sec / 60.0
    if n:
        print(f"  -> current-head hit {ENG[target_cls]}: {cur_hits}/{n}  "
              f"future-head hit: {fut_hits}/{n}")


if __name__ == "__main__":
    import sys
    print("Checkpoints:")
    for ckpt in main.checkpoint_paths:
        print(f"  - {ckpt}")
    # (rec, db, start_min, target_class)
    cases = [
        ("100", "mitdb", 10.0, 0),   # Normal baseline
        ("119", "mitdb", 10.0, 1),   # PVC
        ("209", "mitdb", 10.0, 5),   # AT/SVT
        ("201", "mitdb", 10.0, 2),   # AFib (current demo - suspect)
        ("221", "mitdb", 10.0, 2),   # AFib candidate (sustained)
        ("219", "mitdb", 10.0, 2),   # AFib candidate (paroxysmal)
        ("207", "mitdb", 12.0, 4),   # VT/VFl (current demo - suspect)
        ("205", "mitdb", 10.0, 4),   # VT candidate
        ("215", "mitdb", 10.0, 4),   # VT candidate
    ]
    pick = sys.argv[1:] if len(sys.argv) > 1 else None
    for rec, db, sm, tc in cases:
        if pick and rec not in pick:
            continue
        replay(rec, db, sm, tc, step_sec=30)
