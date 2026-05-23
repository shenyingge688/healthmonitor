"""
Script: backend.py
Version: V13 (Dual Attention Inference)
"""
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import traceback

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

app = FastAPI(title="HealthMonitor V10 Clinical Engine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalHazardNet().to(device)

try:
    checkpoint = torch.load("models/v10_master_best.pth", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["ema"], strict=False)
    if missing:
        # 只允许 hazard_temperature 缺失（旧 checkpoint 兼容）
        critical = [k for k in missing if "temperature" not in k]
        if critical:
            print(f"❌ 模型结构不兼容，缺失关键层: {critical}")
            sys.exit(1)
    if unexpected:
        print(f"⚠️ checkpoint 含未知键（旧版遗留）: {unexpected}")
    model.eval()
    print("✅ V10 引擎挂载成功，终极防弹模式已激活。")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("🛑 安全策略：模型加载失败，拒绝启动服务以防止随机预测输出。")
    sys.exit(1)


class ECGPayload(BaseModel):
    ecg: list[float]


def build_window_sequence(ecg_array: np.ndarray) -> np.ndarray:
    """
    将长度为 API_BUFFER_SIZE 的原始 ECG 缓冲区，按训练逻辑切分为
    19 个重叠窗口 [N_WINDOWS, 1, PTS_PER_WIN]，并对每个窗口独立 Z-score。
    """
    # 只取最后 HISTORY_PTS 个样本（与训练时的窗口覆盖范围一致）
    relevant = ecg_array[-HISTORY_PTS:]

    windows = []
    for w_i in range(N_WINDOWS):
        w_start = w_i * STRIDE_PTS
        w_end = w_start + PTS_PER_WIN
        x_raw = relevant[w_start:w_end]
        # 窗口级独立归一化 —— 与 build_dataset_factory.py:145 完全一致
        x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
        windows.append(x_norm.astype(np.float32))

    # [N_WINDOWS, 1, PTS_PER_WIN]
    return np.stack(windows)[:, np.newaxis, :]


@app.post("/api/predict")
@torch.inference_mode()
def predict(payload: ECGPayload):
    try:
        # 1. 补齐 / 截断至 API_BUFFER_SIZE
        if len(payload.ecg) < API_BUFFER_SIZE:
            ecg_array = np.pad(payload.ecg, (API_BUFFER_SIZE - len(payload.ecg), 0), 'constant')
        else:
            ecg_array = np.array(payload.ecg[-API_BUFFER_SIZE:])

        # 2. 按训练逻辑构建 19 窗口序列，每窗口独立归一化
        win_seq = build_window_sequence(ecg_array)  # [19, 1, 7500]

        # 3. 组装为模型期望的 4D 张量 [B=1, S=19, C=1, L=7500]
        bx = torch.from_numpy(win_seq).unsqueeze(0).to(device)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(bx)["preds"]

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
