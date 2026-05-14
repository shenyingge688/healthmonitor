"""
Script: backend.py (或 main.py / api.py)
Version: V10.0 (Master Clinical API) - 修复 500 Error 版
功能: 修复了输入张量形状不匹配的问题，严格按照 19 个时间窗重组张量
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import asynccontextmanager

from dl_model import HierarchicalHazardNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# 使用 Lifespan 管理模型，避免报错
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"🚀 正在初始化 V10 临床推理引擎 | Device = {device}")
    model = HierarchicalHazardNet().to(device)
    ckpt_path = "models/v10_master_best.pth"
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["ema"])
        model.eval()
        print("✅ V10 模型权重加载成功，服务准备就绪。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    yield
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="HealthMonitor V10 Clinical Engine", lifespan=lifespan)

class ECGPayload(BaseModel):
    ecg: list[float]

@app.post("/api/predict")
@torch.inference_mode()
def predict(payload: ECGPayload):
    # 1. 将数据转为 Numpy，并清洗可能由前端空载传来的 NaN 值
    ecg_raw = np.array(payload.ecg, dtype=np.float32)
    ecg_raw = np.nan_to_num(ecg_raw, nan=0.0)

    # 2. 我们只需要过去 300 秒的数据 (250Hz * 300s = 75000 点) 来构建 19 个窗口
    required_pts = 75000
    if len(ecg_raw) < required_pts:
        ecg_array = np.zeros(required_pts, dtype=np.float32)
        ecg_array[-len(ecg_raw):] = ecg_raw
    else:
        ecg_array = ecg_raw[-required_pts:]

    # 3. 🚀 核心修复：复刻数据工厂的切片逻辑，组装形状 [1, 19, 1, 7500]
    N_WINDOWS = 19
    PTS_PER_WIN = 7500        # 30秒窗口
    OVERLAP_STRIDE = 3750     # 15秒步长
    
    seq_x = np.zeros((1, N_WINDOWS, 1, PTS_PER_WIN), dtype=np.float32)

    for w_i in range(N_WINDOWS):
        w_start = w_i * OVERLAP_STRIDE
        w_end = w_start + PTS_PER_WIN
        x_win = ecg_array[w_start:w_end]
        
        # 必须执行对应的 Z-score 归一化
        std_val = np.std(x_win)
        if std_val < 1e-8:
            std_val = 1e-8
        x_norm = (x_win - np.mean(x_win)) / std_val
        
        seq_x[0, w_i, 0, :] = x_norm

    # 载入 GPU
    bx = torch.from_numpy(seq_x).to(device)

    with torch.amp.autocast("cuda"):
        out = model(bx)["preds"]

        # 提取最后一个时间窗口（当下时刻）进行实时临床诊断
        rhythm_logits = out["rhythm_logits"][0, -1]
        criticality_logits = out["criticality_logits"][0, -1]
        hazard_probs = out["hazard_probs"][0, -1]

        trajectory = out["hazard_probs"][0, -10:, 2].cpu().tolist()

    rhy_probs = F.softmax(rhythm_logits, dim=-1).cpu().tolist()
    cri_probs = F.softmax(criticality_logits, dim=-1).cpu().tolist()
    haz_probs_list = hazard_probs.cpu().tolist()

    return {
        "rhythm": rhy_probs,
        "criticality": cri_probs,
        "hazard": haz_probs_list,
        "risk_trajectory": trajectory
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)