"""
main.py — FastAPI inference service (V6 dual-head)

Serves the V6 ArrhythmiaWarningNet:
  - probs_cur: current rhythm state (6-class)
  - probs_fut: future 2-5min tendency distribution (6-class) — the warning output
  - cam:       1D temporal saliency over the 39 history windows
"""
import os
import sys
import logging
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import find_peaks

# Ensure UTF-8 stdout so emoji/Chinese logs never crash uvicorn on a GBK console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from constants import (
    TARGET_FS, PTS_PER_WIN, STRIDE_PTS, N_WINDOWS,
    HISTORY_PTS, API_BUFFER_SIZE, CLASS_NAMES, MODEL_DIR,
)
from dl_model import ArrhythmiaWarningNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecg-api")

app = FastAPI(title="ECG arrhythmia early-warning inference engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- load model ----
model = ArrhythmiaWarningNet().to(device)
ckpt_path = os.path.join(MODEL_DIR, "arrhythmia_warning_best.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("ema", ckpt.get("model", {}))
    if state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Model weights loaded (%d missing, %d unexpected keys)",
                    len(missing), len(unexpected))
    else:
        logger.warning("Checkpoint has no ema/model state_dict; using random init")
else:
    logger.warning("No checkpoint at %s; using random init", ckpt_path)
model.eval()
logger.info("Inference engine ready, device=%s", device)


# ---- RR feature extraction (must match build_dataset_factory.py) ----
def _extract_rr(win, fs=TARGET_FS):
    try:
        seg = np.asarray(win, dtype=np.float32)
        th = 0.5 * np.std(seg)
        if th < 0.02:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        peaks, _ = find_peaks(seg, height=th, distance=int(fs * 0.25))
        if len(peaks) < 3:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        rr = np.diff(peaks) / fs * 1000.0
        return (np.array([np.mean(rr) / 1000.0, np.std(rr) / 300.0,
                          np.sqrt(np.mean(np.diff(rr) ** 2)) / 100.0 if len(rr) > 1 else 0.0,
                          np.mean(np.abs(np.diff(rr)) > 50) if len(rr) > 1 else 0.0], dtype=np.float32),
                rr.astype(np.float32))
    except Exception:
        return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)


def _sample_entropy(rr, m=2, r_f=0.2):
    rr = np.asarray(rr, dtype=np.float64); N = len(rr)
    if N < m + 2: return 0.0
    rv = r_f * np.std(rr)
    if rv < 1e-10: return 0.0
    def _phi(tl):
        t = np.array([rr[i:i + tl] for i in range(N - tl)])
        nt = len(t)
        if nt < 2: return 0
        return max(sum(np.sum(np.max(np.abs(t - t[i]), axis=1) < rv) - 1 for i in range(nt)), 0)
    A, B = _phi(m + 1), _phi(m)
    return float(-np.log(A / B) / 2.0) if B >= 1 and A > 0 else 0.0


def _poincare(rr):
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2: return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    d = rr[1:] - rr[:-1]; vd = np.var(d); vr = np.var(rr)
    sd1 = np.sqrt(0.5 * max(vd, 0)); sd2 = np.sqrt(max(2 * vr - 0.5 * vd, 0))
    return np.array([sd1 / 100.0, sd2 / 200.0, sd1 / max(sd2, 1e-10)], dtype=np.float32)


def _traj_rr(all_rr):
    if len(all_rr) < 10: return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    p = _poincare(rr)
    return np.array([_sample_entropy(rr), p[0], p[1], p[2],
                     float(np.std(rr) / max(np.mean(rr), 1e-8))], dtype=np.float32)


def build_window_sequence(ecg_array):
    rel = ecg_array[-HISTORY_PTS:]
    wins, f4s, all_rr = [], [], []
    for wi in range(N_WINDOWS):
        s, e = wi * STRIDE_PTS, wi * STRIDE_PTS + PTS_PER_WIN
        x = rel[s:e]
        x_n = (x - np.mean(x)) / (np.std(x) + 1e-8)
        wins.append(x_n.astype(np.float32))
        f4, iv = _extract_rr(x)
        f4s.append(f4)
        if len(iv) > 0: all_rr.append(iv)
    tf = _traj_rr(np.concatenate(all_rr)).astype(np.float32) if all_rr else np.zeros(5, dtype=np.float32)
    rr_f = np.array([np.concatenate([f4s[wi], tf]) for wi in range(N_WINDOWS)], dtype=np.float32)
    return np.stack(wins)[:, np.newaxis, :], rr_f


# ---- API ----
class ECGPayload(BaseModel):
    ecg: list[float]


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/api/predict")
def predict(payload: ECGPayload):
    try:
        ecg = payload.ecg
        if len(ecg) >= API_BUFFER_SIZE:
            arr = np.array(ecg[-API_BUFFER_SIZE:], dtype=np.float32)
        else:
            arr = np.zeros(API_BUFFER_SIZE, dtype=np.float32)
            if len(ecg) > 0:
                arr[-len(ecg):] = np.array(ecg, dtype=np.float32)

        ws, rs = build_window_sequence(arr)
        bx = torch.from_numpy(ws).unsqueeze(0).to(device)
        bx_rr = torch.from_numpy(rs).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(bx, x_rr=bx_rr)

        probs_fut = out["probs_fut"][0].cpu().float().tolist()
        probs_cur = out["probs_cur"][0].cpu().float().tolist()
        pred_fut = int(np.argmax(probs_fut))
        pred_cur = int(np.argmax(probs_cur))

        return {
            # Warning product = FUTURE tendency head
            "pred_class": pred_fut,
            "class_name": CLASS_NAMES[pred_fut],
            "probabilities": probs_fut,        # future tendency distribution
            # Current rhythm state (second head)
            "current_class": pred_cur,
            "current_name": CLASS_NAMES[pred_cur],
            "probabilities_cur": probs_cur,
            "cam": out["cam"][0].cpu().float().tolist(),
        }
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=traceback.format_exc()[-800:])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
