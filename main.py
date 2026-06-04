"""
Script: backend.py
心电智能监护推理服务 — 基于因果 TCN + 能量包络 + RR 间期特征的节律分类与风险预测
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import traceback
from scipy.signal import find_peaks

from dl_model import HierarchicalHazardNet

# —— 窗口参数，必须与 build_dataset_factory.py 保持一致 ——
TARGET_FS = 250
WINDOW_SEC = 30
OVERLAP_STRIDE_SEC = 15
PTS_PER_WIN = WINDOW_SEC * TARGET_FS          # 7500
STRIDE_PTS = OVERLAP_STRIDE_SEC * TARGET_FS    # 3750
N_WINDOWS = 19                                  # (300-30)//15 + 1
HISTORY_PTS = (N_WINDOWS - 1) * STRIDE_PTS + PTS_PER_WIN  # 75000
# API 接收 150000 点的缓冲区，但模型实际窗口仅使用最后 75000 点
API_BUFFER_SIZE = 150000

app = FastAPI(title="心电智能监护推理引擎")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalHazardNet().to(device)

def _load_checkpoint(model, ckpt_path):
    """容错加载：缺失/新增层 → 警告；尺寸不匹配 → 警告（需重训）"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("⚠️  checkpoint 维度不兼容（旧架构）— 模型将以随机权重启动")
            print("   请先运行 python train_trajectory.py 训练模型")
            return False
        raise

    if missing:
        critical = [k for k in missing if not any(x in k for x in (
            "heads.", "log_vars", "attention_query", "rhythm_head", "rhythm_envelope",
            "criticality_head", "hazard_head", "tcn.", "temperature", "input_proj",
            "rr_encoder",
        ))]
        if critical:
            print(f"❌ 模型结构不兼容，缺失关键层: {critical}")
            sys.exit(1)
        else:
            print(f"⚠️ {len(missing)} keys 缺失（新架构），将从零初始化")
    if unexpected:
        print(f"⚠️ checkpoint 含未知键（旧版遗留）: {len(unexpected)} 个")
    return True

ckpt_path = "models/v20_master_best.pth"
if os.path.exists(ckpt_path):
    ok = _load_checkpoint(model, ckpt_path)
    if not ok:
        print("🟡 使用未训练权重启动（仅供开发测试）")
else:
    print("⚠️ 未找到 checkpoint — 使用随机权重")
model.eval()
print(f"✅ 推理引擎启动成功。| Rhythm={model.heads.rhythm_head.out_features}cls | 等待推理请求...")


@app.get("/api/health")
def health():
    return {"status": "ok", "rhythm_classes": 3}


class ECGPayload(BaseModel):
    ecg: list[float]


def extract_rr_features(ecg_window, fs=250):
    """Extract RR interval features from a 30s ECG window.
    Returns (features_4, rr_intervals_ms) tuple."""
    try:
        seg = np.asarray(ecg_window, dtype=np.float32)
        threshold = 0.5 * np.std(seg)
        if threshold < 0.02:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        peaks, _ = find_peaks(seg, height=threshold, distance=int(fs * 0.25))
        if len(peaks) < 3:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        rr = np.diff(peaks) / fs * 1000.0
        mean_rr = np.mean(rr) / 1000.0
        sdnn = np.std(rr) / 300.0
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) / 100.0 if len(rr) > 1 else 0.0
        pnn50 = np.mean(np.abs(np.diff(rr)) > 50) if len(rr) > 1 else 0.0
        return (np.array([mean_rr, sdnn, rmssd, pnn50], dtype=np.float32),
                rr.astype(np.float32))
    except Exception:
        return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)


# ---------- Trajectory-level RR features (mirrors build_dataset_factory.py) ----------
def _sample_entropy_main(rr, m=2, r_factor=0.2):
    """Sample entropy of RR interval series (inference version)."""
    rr = np.asarray(rr, dtype=np.float64)
    N = len(rr)
    if N < m + 2: return 0.0
    r = r_factor * np.std(rr)
    if r < 1e-10: return 0.0
    def _phi(tl):
        templates = np.array([rr[i:i+tl] for i in range(N-tl)])
        nt = len(templates)
        if nt < 2: return 0
        count = sum(np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1 for i in range(nt))
        return max(count, 0)
    A, B = _phi(m+1), _phi(m)
    return float(-np.log(A / B) / 2.0) if B >= 1 and A > 0 else 0.0

def _poincare_features_main(rr):
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2: return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    diff = rr[1:] - rr[:-1]
    sd1 = np.sqrt(0.5 * max(np.var(diff), 0))
    sd2 = np.sqrt(max(2*np.var(rr) - 0.5*np.var(diff), 0))
    return np.array([sd1/100.0, sd2/200.0, sd1/max(sd2,1e-10)], dtype=np.float32)

def compute_traj_rr_features_main(all_rr):
    if len(all_rr) < 10: return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    poincare = _poincare_features_main(rr)
    return np.array([_sample_entropy_main(rr),
                     poincare[0], poincare[1], poincare[2],
                     float(np.std(rr)/max(np.mean(rr),1e-8))], dtype=np.float32)


def build_window_sequence(ecg_array: np.ndarray):
    """
    将长度为 API_BUFFER_SIZE 的原始 ECG 缓冲区，按训练逻辑切分为
    19 个重叠窗口 [N_WINDOWS, 1, PTS_PER_WIN] + RR 特征 [N_WINDOWS, 9]。
    """
    relevant = ecg_array[-HISTORY_PTS:]

    windows, rr_feat4_list, all_rr = [], [], []
    for w_i in range(N_WINDOWS):
        w_start = w_i * STRIDE_PTS
        w_end = w_start + PTS_PER_WIN
        x_raw = relevant[w_start:w_end]
        x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
        windows.append(x_norm.astype(np.float32))
        feat4, intervals = extract_rr_features(x_raw)
        rr_feat4_list.append(feat4)
        if len(intervals) > 0:
            all_rr.append(intervals)

    # Compute trajectory-level features and broadcast to all 19 windows
    if len(all_rr) > 0:
        traj_feat = compute_traj_rr_features_main(np.concatenate(all_rr)).astype(np.float32)
    else:
        traj_feat = np.zeros(5, dtype=np.float32)
    rr_full = np.array([np.concatenate([rr_feat4_list[w_i], traj_feat])
                        for w_i in range(N_WINDOWS)], dtype=np.float32)  # [19, 9]

    return (
        np.stack(windows)[:, np.newaxis, :],          # [N_WINDOWS, 1, PTS_PER_WIN]
        rr_full,                                        # [N_WINDOWS, 9]
    )


@app.post("/api/predict")
@torch.inference_mode()
def predict(payload: ECGPayload):
    try:
        # 1. 补齐 / 截断至 API_BUFFER_SIZE
        if len(payload.ecg) < API_BUFFER_SIZE:
            ecg_array = np.pad(payload.ecg, (API_BUFFER_SIZE - len(payload.ecg), 0), 'constant')
        else:
            ecg_array = np.array(payload.ecg[-API_BUFFER_SIZE:])

        # 2. 按训练逻辑构建 19 窗口序列 + RR 特征
        win_seq, rr_seq = build_window_sequence(ecg_array)  # [19, 1, 7500], [19, 9]

        # 3. 组装为模型期望的张量
        bx = torch.from_numpy(win_seq).unsqueeze(0).to(device)        # [1, 19, 1, 7500]
        bx_rr = torch.from_numpy(rr_seq).unsqueeze(0).to(device)      # [1, 19, 9]

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(bx, x_rr=bx_rr)["preds"]

            rhythm_logits = out["rhythm_logits"][0, -1]
            criticality_logits = out["criticality_logits"][0, -1]
            hazard_probs = out["hazard_probs"][0, -1]  # [3] → 30s/1m/5m
            # 返回当前 5 分钟风险值，dashboard 自行维护历史轨迹
            hazard_5m = hazard_probs[2]

            # 防线：切断任何潜在的 NaN/Inf，保护 FastAPI JSON 序列化器
            rhythm_logits = torch.nan_to_num(rhythm_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            criticality_logits = torch.nan_to_num(criticality_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            hazard_probs = torch.nan_to_num(hazard_probs, nan=0.0, posinf=1.0, neginf=0.0)
            hazard_5m = torch.nan_to_num(hazard_5m, nan=0.0, posinf=1.0, neginf=0.0)

        rhy_probs = F.softmax(rhythm_logits, dim=-1).cpu().tolist()
        cri_probs = F.softmax(criticality_logits, dim=-1).cpu().tolist()
        haz_probs_list = hazard_probs.cpu().tolist()

        return {
            "rhythm": rhy_probs,
            "criticality": cri_probs,
            "hazard": haz_probs_list,
            "risk_trajectory": float(hazard_5m.cpu().item()),
        }

    except Exception as e:
        err_detail = traceback.format_exc()
        print("\n" + "=" * 50)
        print("❌ 后端发生致命错误:")
        print(err_detail)
        print("=" * 50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
