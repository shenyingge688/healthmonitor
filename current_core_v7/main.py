"""
main.py - FastAPI inference service for the accepted Round5 ensemble.

Serves the seed0/seed1/seed4 ArrhythmiaWarningNet logits ensemble:
  - probs_cur: current rhythm state (6-class)
  - probs_fut: future 2-5min tendency distribution (6-class)
  - cam:       1D temporal saliency over the 39 history windows
  - risk_std:  member disagreement for descriptive confidence display
"""
import os
import sys
import json
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
    HISTORY_PTS, API_BUFFER_SIZE, CLASS_NAMES, BASE_DIR,
)
from dl_model import ArrhythmiaWarningNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecg-api")

app = FastAPI(title="ECG arrhythmia early-warning inference engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- load accepted Round5 ensemble ----
DEFAULT_ENSEMBLE_CHECKPOINTS = [
    os.path.join(
        BASE_DIR,
        "models_v7_stage1",
        "seeds",
        f"seed{seed}",
        "arrhythmia_warning_best.pth",
    )
    for seed in (0, 1, 4)
]
UNCERTAINTY_CONFIG_PATH = os.environ.get(
    "ECG_UNCERTAINTY_CONFIG",
    os.path.join(
        BASE_DIR,
        "results_v7_round6",
        "ensemble_uncertainty_config.json",
    ),
)
CONFIDENCE_SEMANTICS = (
    "Descriptive agreement across ensemble members; not a clinical confidence "
    "interval or probability of correctness."
)


def configured_checkpoint_paths():
    override = os.environ.get("ECG_ENSEMBLE_CHECKPOINTS", "").strip()
    if not override:
        return DEFAULT_ENSEMBLE_CHECKPOINTS
    return [path.strip() for path in override.split(";") if path.strip()]


def load_uncertainty_config(path):
    if not os.path.exists(path):
        logger.warning(
            "Uncertainty config missing at %s; using validation-derived fallbacks",
            path,
        )
        return {
            "high": {"max_risk_std": 0.09, "required_vote_count": 3},
            "medium": {"max_risk_std": 0.22, "required_vote_count": 2},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ensemble(paths):
    loaded = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required ensemble checkpoint is missing: {path}")
        member = ArrhythmiaWarningNet().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt.get("ema", ckpt.get("model", {}))
        if not state:
            raise RuntimeError(f"Checkpoint has no ema/model state_dict: {path}")
        missing, unexpected = member.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible checkpoint {path}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )
        member.eval()
        loaded.append(member)
        logger.info("Loaded ensemble member: %s", path)
    if len(loaded) < 2:
        raise RuntimeError("Ensemble inference requires at least two checkpoints")
    return loaded


checkpoint_paths = configured_checkpoint_paths()
models = load_ensemble(checkpoint_paths)
model = models[0]  # Backward-compatible handle; serving uses run_ensemble().
ckpt_path = checkpoint_paths[0]
uncertainty_config = load_uncertainty_config(UNCERTAINTY_CONFIG_PATH)
logger.info(
    "Inference engine ready, device=%s, ensemble_size=%d",
    device,
    len(models),
)


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


@torch.inference_mode()
def run_ensemble(bx, bx_rr):
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(autocast_device):
        member_outputs = [member(bx, x_rr=bx_rr) for member in models]
    member_logits_cur = torch.stack(
        [out["logits_cur"].float() for out in member_outputs],
        dim=1,
    )
    member_logits_fut = torch.stack(
        [out["logits_fut"].float() for out in member_outputs],
        dim=1,
    )
    logits_cur = member_logits_cur.mean(dim=1)
    logits_fut = member_logits_fut.mean(dim=1)
    probs_cur = F.softmax(logits_cur, dim=-1)
    probs_fut = F.softmax(logits_fut, dim=-1)

    member_probs_fut = torch.stack(
        [out["probs_fut"].float() for out in member_outputs],
        dim=1,
    )
    member_risks = 1.0 - member_probs_fut[:, :, 0]
    risk_mean = member_risks.mean(dim=1)
    risk_std = member_risks.std(dim=1, unbiased=False)
    probabilities_std = member_probs_fut.std(dim=1, unbiased=False)
    member_classes = member_probs_fut.argmax(dim=-1)

    vote_counts = []
    confidence = []
    high_cfg = uncertainty_config["high"]
    medium_cfg = uncertainty_config["medium"]
    for batch_i in range(member_classes.shape[0]):
        counts = torch.bincount(
            member_classes[batch_i],
            minlength=len(CLASS_NAMES),
        )
        vote_count = int(counts.max().item())
        vote_counts.append(vote_count)
        std_value = float(risk_std[batch_i].item())
        if (
            std_value <= float(high_cfg["max_risk_std"])
            and vote_count >= int(high_cfg["required_vote_count"])
        ):
            confidence.append("high")
        elif (
            std_value <= float(medium_cfg["max_risk_std"])
            and vote_count >= int(medium_cfg["required_vote_count"])
        ):
            confidence.append("medium")
        else:
            confidence.append("low")

    return {
        "logits_cur": logits_cur,
        "probs_cur": probs_cur,
        "logits_fut": logits_fut,
        "probs_fut": probs_fut,
        "cam": torch.stack(
            [out["cam"].float() for out in member_outputs],
            dim=1,
        ).mean(dim=1),
        "member_probs_fut": member_probs_fut,
        "member_risks": member_risks,
        "risk_mean": risk_mean,
        "risk_std": risk_std,
        "probabilities_std": probabilities_std,
        "future_vote_count": vote_counts,
        "confidence": confidence,
    }


# ---- API ----
class ECGPayload(BaseModel):
    ecg: list[float]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "ensemble_size": len(models),
        "ensemble_members": [
            os.path.basename(os.path.dirname(path)) for path in checkpoint_paths
        ],
        "checkpoints": checkpoint_paths,
        "uncertainty_config": UNCERTAINTY_CONFIG_PATH,
    }


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
        out = run_ensemble(bx, bx_rr)

        probs_fut = out["probs_fut"][0].cpu().float().tolist()
        probs_cur = out["probs_cur"][0].cpu().float().tolist()
        pred_fut = int(np.argmax(probs_fut))
        pred_cur = int(np.argmax(probs_cur))
        member_risks = out["member_risks"][0].cpu().float().tolist()
        risk_mean = float(out["risk_mean"][0].cpu())
        risk_std = float(out["risk_std"][0].cpu())

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
            # Descriptive ensemble disagreement, not clinical confidence.
            "ensemble_size": len(models),
            "risk_score": float(1.0 - probs_fut[0]),
            "risk_mean": risk_mean,
            "risk_std": risk_std,
            "member_risks": member_risks,
            "probabilities_std": (
                out["probabilities_std"][0].cpu().float().tolist()
            ),
            "future_vote_count": int(out["future_vote_count"][0]),
            "confidence": out["confidence"][0],
            "confidence_note": CONFIDENCE_SEMANTICS,
        }
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=traceback.format_exc()[-800:])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
