"""
Script: main.py
Version: V10.0 (Hierarchical Hazard API)
"""
import torch
import numpy as np
import uvicorn
import os
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from dl_model import HierarchicalHazardNet

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalHazardNet().to(device)

if os.path.exists('models/ptfn_v10_core.pth'): 
    model.load_state_dict(torch.load('models/ptfn_v10_core.pth', map_location=device, weights_only=True))
model.eval()

class InferenceRequest(BaseModel): 
    ecg: list

@app.post("/api/predict")
def predict(data: InferenceRequest):
    ecg_clean = np.array(data.ecg)
    TARGET_FS, WINDOW_SEC, STRIDE_SEC = 250, 30, 15
    pts_per_win, stride_pts = WINDOW_SEC * TARGET_FS, STRIDE_SEC * TARGET_FS
    n_windows = (len(ecg_clean) - pts_per_win) // stride_pts + 1
    
    if n_windows <= 0: return {"error": "数据长度不足"}
        
    seq_data = [((ecg_clean[i*stride_pts:i*stride_pts+pts_per_win] - np.mean(ecg_clean[i*stride_pts:i*stride_pts+pts_per_win])) / (np.std(ecg_clean[i*stride_pts:i*stride_pts+pts_per_win]) + 1e-8)).astype(np.float32) for i in range(n_windows)]
    input_tensor = torch.tensor(np.array(seq_data).reshape(1, n_windows, 1, pts_per_win), dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(input_tensor)["preds"]
        
        # 获取最后一个时间步的概率分布
        rhythm_probs = F.softmax(out["rhythm_logits"][0, -1], dim=-1).cpu().numpy().tolist()
        crit_probs = F.softmax(out["criticality_logits"][0, -1], dim=-1).cpu().numpy().tolist()
        hazard_probs = out["hazard_probs"][0, -1].cpu().numpy().tolist()

    return {
        "rhythm_probs": rhythm_probs,
        "crit_probs": crit_probs,
        "hazard_probs": hazard_probs
    }

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)