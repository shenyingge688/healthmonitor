"""
eval_gradcam.py — P5: verify the Grad-CAM actually lands on real ectopic beats.

For a demo record with annotated V/E (PVC) beats, build the 10-min history
window, run class-conditional Grad-CAM for the PVC/VT class, and check whether
the high-saliency history windows coincide with windows that actually contain
annotated ventricular ectopic beats. This turns the explanation from a pretty
strip into a claim verifiable against ground-truth annotations.
"""
import os
import numpy as np
import torch
import wfdb
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from healthmonitor.dl_model import ArrhythmiaWarningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FS = 250
HISTORY_SEC, WINDOW_SEC, STRIDE_SEC = 600, 30, 15
N_WINDOWS = 39
PTS_PER_WIN = WINDOW_SEC * TARGET_FS
STRIDE_PTS = STRIDE_SEC * TARGET_FS
HISTORY_PTS = (N_WINDOWS - 1) * STRIDE_PTS + PTS_PER_WIN


def clean(sig, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype="band")
    return filtfilt(b, a, sig)


def extract_rr(win, fs=TARGET_FS):
    from scipy.signal import find_peaks
    seg = np.asarray(win, dtype=np.float32)
    th = 0.5 * np.std(seg)
    if th < 0.02:
        return np.zeros(4, dtype=np.float32)
    peaks, _ = find_peaks(seg, height=th, distance=int(fs * 0.25))
    if len(peaks) < 3:
        return np.zeros(4, dtype=np.float32)
    rr = np.diff(peaks) / fs * 1000.0
    return np.array([np.mean(rr) / 1000, np.std(rr) / 300,
                     np.sqrt(np.mean(np.diff(rr) ** 2)) / 100 if len(rr) > 1 else 0,
                     np.mean(np.abs(np.diff(rr)) > 50) if len(rr) > 1 else 0], dtype=np.float32)


def build_history(ecg, end_pt):
    rel = ecg[end_pt - HISTORY_PTS:end_pt]
    wins, rr4 = [], []
    for wi in range(N_WINDOWS):
        s = wi * STRIDE_PTS
        x = rel[s:s + PTS_PER_WIN]
        wins.append(((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32))
        rr4.append(extract_rr(x))
    rr_full = np.array([np.concatenate([rr4[i], np.zeros(5, np.float32)]) for i in range(N_WINDOWS)], np.float32)
    return np.stack(wins)[:, None, :], rr_full


def pvc_windows_from_ann(ann, end_pt, src_fs):
    """Mark which of the 39 history windows contain >=1 annotated V/E beat."""
    scale = TARGET_FS / float(src_fs)
    vbeats = [int(s * scale) for s, sym in zip(ann.sample, ann.symbol) if sym in ("V", "E")]
    start = end_pt - HISTORY_PTS
    has_v = np.zeros(N_WINDOWS, dtype=bool)
    for wi in range(N_WINDOWS):
        ws = start + wi * STRIDE_PTS
        we = ws + PTS_PER_WIN
        has_v[wi] = any(ws <= vb < we for vb in vbeats)
    return has_v


def main(rec="119", db="mitdb", target_class=1):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", db, rec)
    record = wfdb.rdrecord(path, sampto=int(20 * 60 * 360) if db == "mitdb" else None)
    ann = wfdb.rdann(path, "atr", pn_dir=None)
    src_fs = record.fs
    raw = record.p_signal[:, 0]
    ecg = signal.resample_poly(clean(raw, src_fs), TARGET_FS, int(src_fs)).astype(np.float32) \
        if src_fs != TARGET_FS else clean(raw, src_fs).astype(np.float32)

    end_pt = HISTORY_PTS + 60 * TARGET_FS   # 1 min into the record's usable span
    if end_pt > len(ecg):
        end_pt = len(ecg) - 1
    ws, rr = build_history(ecg, end_pt)
    bx = torch.from_numpy(ws).unsqueeze(0).to(device)
    bx_rr = torch.from_numpy(rr).unsqueeze(0).to(device)

    model = ArrhythmiaWarningNet().to(device)
    ck = torch.load(os.path.join(base, "models", "arrhythmia_warning_best.pth"),
                    map_location=device, weights_only=True)
    model.load_state_dict(ck.get("ema", ck.get("model", {})), strict=False)
    model.eval()

    cam = model.grad_cam(bx, bx_rr, target_class=target_class, head="future")[0].cpu().numpy()
    has_v = pvc_windows_from_ann(ann, end_pt, src_fs)

    # Alignment: mean CAM on windows WITH ectopic beats vs WITHOUT
    cam_v = cam[has_v].mean() if has_v.any() else float("nan")
    cam_n = cam[~has_v].mean() if (~has_v).any() else float("nan")
    print(f"Record {rec} (db={db}), target_class={target_class}")
    print(f"  history windows with annotated V/E beats: {int(has_v.sum())}/{N_WINDOWS}")
    print(f"  mean Grad-CAM on ECTOPIC windows : {cam_v:.3f}")
    print(f"  mean Grad-CAM on NON-ectopic wins: {cam_n:.3f}")
    if has_v.any() and (~has_v).any():
        print(f"  -> saliency is {'HIGHER' if cam_v > cam_n else 'NOT higher'} on true ectopic windows "
              f"(ratio {cam_v / (cam_n + 1e-6):.2f}x)")

    # plot
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(np.arange(N_WINDOWS), cam, color=["#EF4444" if v else "#3B82F6" for v in has_v])
    ax.set_xlabel("History window (red = contains annotated V/E beat)")
    ax.set_ylabel("Grad-CAM")
    ax.set_title(f"Class-conditional Grad-CAM vs annotated ectopy — {db}/{rec}")
    fig.tight_layout()
    os.makedirs(os.path.join(base, "results"), exist_ok=True)
    out = os.path.join(base, "results", f"gradcam_{db}_{rec}.png")
    fig.savefig(out, dpi=150)
    print(f"  saved {out}")


if __name__ == "__main__":
    # 201 = held-out DEMO record with sparse, localized PVC — verify PVC-class (1)
    # Grad-CAM saliency concentrates on the history windows that actually contain
    # annotated ventricular ectopic beats (a patient never seen in training).
    main(rec="201", db="mitdb", target_class=1)
