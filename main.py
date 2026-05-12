"""
临床推断网关
包含: 异构联合推断 | MC Dropout 认知不确定性评估
"""
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from dl_model import LatentDynamicsForecastingNet

app = FastAPI(title="PTFN Clinical Inference Gateway")
model = LatentDynamicsForecastingNet()
model.eval()

def enable_dropout(m):
    if type(m) == torch.nn.Dropout: m.train()

@app.post("/api/predict_trajectory")
def predict_trajectory(data: dict):
    ecg_clean = np.array(data.get('ecg', []))
    TARGET_FS = 250
    HISTORY_SEC = 300 
    TARGET_LEN = HISTORY_SEC * TARGET_FS 

    if len(ecg_clean) != TARGET_LEN:
        raise HTTPException(status_code=400, detail="需完整的 5 分钟基准切片")

    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    
    # 动态构建重叠窗口
    WINDOW_SEC, STRIDE_SEC = 30, 15
    pts_per_win = WINDOW_SEC * TARGET_FS
    stride_pts = STRIDE_SEC * TARGET_FS
    n_windows = (TARGET_LEN - pts_per_win) // stride_pts + 1
    
    seq_data = [ecg_norm[i*stride_pts : i*stride_pts + pts_per_win] for i in range(n_windows)]
    input_tensor = torch.tensor(np.array(seq_data).reshape(n_windows, 1, pts_per_win), dtype=torch.float32).unsqueeze(0)

    # 预测意图：查询 5 分钟视界
    h_idx_5m = torch.tensor([2], dtype=torch.long)

    # 启用 MC Dropout 估计不确定性
    model.apply(enable_dropout) 
    mc_vt_hazards = []
    
    with torch.no_grad():
        for _ in range(20):
            out = model(input_tensor, h_idx_5m)
            # 使用 Sigmoid 激活 Hazard Logits
            mc_vt_hazards.append(torch.sigmoid(out["preds"]["vt_hazard_logits"])[0].item())
            
    mean_hazard = np.mean(mc_vt_hazards)
    uncertainty = np.std(mc_vt_hazards)
    
    # 提取无 Dropout 的归因权重
    model.eval()
    with torch.no_grad():
        clean_out = model(input_tensor, h_idx_5m)
        attn_weights = clean_out["attention"][0].tolist()

    return {
        "future_5m_prediction": {
            "vt_hazard_prob": float(mean_hazard),
            "epistemic_uncertainty": float(uncertainty),
            "is_reliable": uncertainty < 0.15 
        },
        "clinical_attribution_signals": attn_weights
    }