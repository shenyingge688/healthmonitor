"""
main.py - FastAPI inference service for the frozen official ensemble.

Serves the three-member ArrhythmiaWarningNet logits ensemble:
  - probs_cur: current rhythm state (6-class)
  - probs_fut: future 5min tendency distribution (6-class)
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

from .constants import (
    TARGET_FS, PTS_PER_WIN, STRIDE_PTS, N_WINDOWS,
    HISTORY_PTS, API_BUFFER_SIZE, CLASS_NAMES,
)
from .dl_model import ArrhythmiaWarningNet
from .monitoring_policy import (
    DEFAULT_POLICY,
    POLICY_CONFIG_VERSION,
    policy_config_dict,
)
from .paths import OFFICIAL_ENSEMBLE_DIR, POLICY_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ecg-api")

app = FastAPI(title="ECG arrhythmia early-warning inference engine")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- load frozen official ensemble ----
DEFAULT_ENSEMBLE_CHECKPOINTS = [
    str(OFFICIAL_ENSEMBLE_DIR / f"seed{seed}" / "arrhythmia_warning_best.pth")
    for seed in (0, 1, 4)
]
UNCERTAINTY_CONFIG_PATH = os.environ.get(
    "ECG_UNCERTAINTY_CONFIG",
    str(POLICY_DIR / "ensemble_uncertainty_config.json"),
)
CONFIDENCE_SEMANTICS = (
    "Descriptive agreement across ensemble members; not clinical confidence, "
    "a confidence interval, or a probability of correctness."
)
AGREEMENT_NOTES = {
    "high": "三个模型输出较一致；该标签仅描述模型间一致性。",
    "medium": "模型输出存在一定差异，建议结合信号质量复核。",
    "low": "模型输出分歧较大，仅作风险参考。",
}


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


def assess_signal_quality(ecg_array, fs=TARGET_FS, recent_sec=10):
    """Return a conservative, transparent signal-quality summary."""
    arr = np.asarray(ecg_array, dtype=np.float32)
    recent_n = max(int(fs * recent_sec), 1)
    recent = arr[-recent_n:] if len(arr) else arr
    finite_fraction = float(np.isfinite(recent).mean()) if len(recent) else 0.0
    finite = recent[np.isfinite(recent)]
    reasons = []
    if len(finite) < recent_n * 0.95:
        reasons.append("存在缺失或非有限采样点")

    if len(finite):
        std = float(np.std(finite))
        peak_to_peak = float(np.ptp(finite))
        diffs = np.abs(np.diff(finite))
        flat_fraction = float(np.mean(diffs < 1e-5)) if len(diffs) else 1.0
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust_limit = max(12.0 * mad, 2.0)
        clipping_fraction = float(
            np.mean(np.abs(finite - median) > robust_limit)
        )
    else:
        std = peak_to_peak = 0.0
        flat_fraction = clipping_fraction = 1.0

    score = 1.0
    if std < 0.015 or peak_to_peak < 0.05:
        score -= 0.65
        reasons.append("信号幅度过低或接近直线")
    if flat_fraction > 0.20:
        score -= 0.35
        reasons.append("连续平坦采样比例偏高")
    if clipping_fraction > 0.02:
        score -= 0.25
        reasons.append("存在疑似削顶或异常尖峰")
    if finite_fraction < 0.95:
        score -= 0.50
    score = float(np.clip(score, 0.0, 1.0))
    level = "good" if score >= 0.75 else "limited" if score >= 0.40 else "poor"
    return {
        "level": level,
        "score": score,
        "reasons": reasons or ["近期信号幅度与连续性未见明显异常"],
        "metrics": {
            "finite_fraction": finite_fraction,
            "std": std,
            "peak_to_peak": peak_to_peak,
            "flat_fraction": flat_fraction,
            "clipping_fraction": clipping_fraction,
        },
        "window_sec": int(recent_sec),
    }


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
        "policy_config": policy_config_dict(),
        "policy_config_version": POLICY_CONFIG_VERSION,
    }


@app.post("/api/predict")
def predict(payload: ECGPayload):
    try:
        ecg = payload.ecg
        if len(ecg) == 0:
            raise HTTPException(
                status_code=422,
                detail="ECG payload is empty; no prediction was produced.",
            )
        received_samples = len(ecg)
        if received_samples < HISTORY_PTS:
            raise HTTPException(
                status_code=422,
                detail=(
                    "ECG payload is shorter than the required 10-minute "
                    "history; no prediction was produced."
                ),
            )
        history_complete = received_samples >= HISTORY_PTS
        if len(ecg) >= API_BUFFER_SIZE:
            arr = np.array(ecg[-API_BUFFER_SIZE:], dtype=np.float32)
        else:
            arr = np.zeros(API_BUFFER_SIZE, dtype=np.float32)
            if len(ecg) > 0:
                arr[-len(ecg):] = np.array(ecg, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise HTTPException(
                status_code=422,
                detail="ECG payload contains non-finite values; no prediction was produced.",
            )
        signal_quality = assess_signal_quality(arr)
        if signal_quality["level"] == "poor":
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "ECG signal quality is too low; no prediction was produced.",
                    "signal_quality": signal_quality,
                },
            )

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
        overall_risk_raw = float(1.0 - probs_fut[0])
        afib_denominator = max(float(probs_fut[2] + probs_fut[0]), 1e-8)
        afib_direction_score = float(probs_fut[2] / afib_denominator)
        agreement_level = out["confidence"][0]
        if signal_quality["level"] == "limited":
            input_status = "limited_signal"
        else:
            input_status = "ready"

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
            "risk_score": overall_risk_raw,
            "risk_mean": risk_mean,
            "risk_std": risk_std,
            "member_risks": member_risks,
            "probabilities_std": (
                out["probabilities_std"][0].cpu().float().tolist()
            ),
            "future_vote_count": int(out["future_vote_count"][0]),
            "confidence": agreement_level,
            "confidence_note": CONFIDENCE_SEMANTICS,
            # Public additions; legacy fields above remain unchanged for compatibility.
            "overall_risk_raw": overall_risk_raw,
            "afib_direction_score": afib_direction_score,
            "signal_quality": signal_quality,
            "agreement_level": agreement_level,
            "agreement_note": AGREEMENT_NOTES.get(
                agreement_level,
                CONFIDENCE_SEMANTICS,
            ),
            "policy_config_version": POLICY_CONFIG_VERSION,
            "policy_config": policy_config_dict(DEFAULT_POLICY),
            "input_status": input_status,
            "history_complete": history_complete,
            "received_samples": received_samples,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=traceback.format_exc()[-800:])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
