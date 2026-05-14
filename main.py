"""
Script: main.py
Version: V8.3 Final (Clinical Inference Gateway)
功能描述:
为 Dashboard 提供实时 API 服务。
将 10 分钟的心电序列流切分为滚动窗口，送入冻结的单向因果模型。
计算 Log-Survival 累积生存风险、多任务并发概率，并返回 Time-to-Alarm 轨迹。
"""
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

from dl_model import LatentDynamicsForecastingNet

app = FastAPI(title="PTFN V8.3 Clinical Inference Gateway")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LatentDynamicsForecastingNet().to(device)

# 加载 V8.3 冻结的核心动力学权重
weight_path = 'models/ptfn_v83_core.pth'
if os.path.exists(weight_path):
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print("✅ [Gateway] V8.3 核心动力学权重加载成功！")
    except Exception as e:
        print(f"⚠️ [Gateway] 权重加载异常: {e}")
else:
    print("⚠️ [Gateway] 未找到权重文件，将使用随机初始化参数 (用于代码联调)")

model.eval()

class InferenceRequest(BaseModel):
    ecg: list

@app.post("/api/predict")
def predict(data: InferenceRequest):
    ecg_clean = np.array(data.ecg)
    TARGET_FS = 250
    
    # 动态适配长度：Dashboard 发送的是 150000 个点 (10分钟)
    if len(ecg_clean) < 75000: # 至少需要5分钟的数据来形成基础视野
        raise HTTPException(status_code=400, detail="数据长度不足")

    # V8.3 滚动视界参数：30秒窗口，15秒步长
    WINDOW_SEC, STRIDE_SEC = 30, 15
    pts_per_win = WINDOW_SEC * TARGET_FS
    stride_pts = STRIDE_SEC * TARGET_FS
    n_windows = (len(ecg_clean) - pts_per_win) // stride_pts + 1
    
    # 逐窗口局部标准化 (与 build_dataset_factory.py 严格对齐)
    seq_data = []
    for i in range(n_windows):
        x_raw = ecg_clean[i*stride_pts : i*stride_pts + pts_per_win]
        x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
        seq_data.append(x_norm.astype(np.float32))

    # 构建模型输入张量 Shape: [Batch=1, SeqLen, Channels=1, SignalLen]
    input_tensor = torch.tensor(np.array(seq_data).reshape(1, n_windows, 1, pts_per_win), dtype=torch.float32).to(device)

    with torch.no_grad():
        # 纯粹前向传播 (无 Attention, 单向推演)
        out = model(input_tensor)
        logits_vt = out["preds"]["vt_hazard_logits"][0] # Shape: [SeqLen, 3]
        logits_pvc = out["preds"]["pvc_logits"][0]      # Shape: [SeqLen]
        logits_afib = out["preds"]["afib_logits"][0]    # Shape: [SeqLen]
        
        # ==========================================
        # 计算 10-Min Risk Trajectory (Time-to-Alarm 曲线)
        # ==========================================
        risk_trajectory = []
        for t in range(n_windows):
            # V8.3 Log-Survival 稳定推导法则
            hazards_t = 0.5 * torch.sigmoid(logits_vt[t])
            log_surv_t = torch.sum(torch.log(1.0 - hazards_t + 1e-6))
            risk_t = 1.0 - torch.exp(log_surv_t)
            risk_trajectory.append(risk_t.item())

        # ==========================================
        # 提取当前时刻 (t = -1) 的多任务并发状态
        # ==========================================
        final_hazards = 0.5 * torch.sigmoid(logits_vt[-1])
        
        vt_risk_30s = final_hazards[0].item() # 极短期崩溃概率
        # 2分钟崩溃概率 = 1 - exp(log(1-h1) + log(1-h2))
        vt_risk_2m = 1.0 - torch.exp(torch.sum(torch.log(1.0 - final_hazards[:2] + 1e-6))).item()
        vt_risk_5m = risk_trajectory[-1]      # 综合恶化指数
        
        pvc_prob = torch.sigmoid(logits_pvc[-1]).item()
        afib_prob = torch.sigmoid(logits_afib[-1]).item()

    return {
        "vt_risk_5m": float(vt_risk_5m),
        "risk_trajectory": risk_trajectory,
        "pvc_prob": float(pvc_prob),
        "afib_prob": float(afib_prob),
        "vt_risk_30s": float(vt_risk_30s),
        "vt_risk_2m": float(vt_risk_2m)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)