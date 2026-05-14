"""
Script: backend.py 
Version: V10.3 (Ultimate Master API - Dimension & Overflow Secured)
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import traceback

from dl_model import HierarchicalHazardNet

app = FastAPI(title="HealthMonitor V10 Clinical Engine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalHazardNet().to(device)

try:
    checkpoint = torch.load("models/v10_master_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    print("✅ V10 引擎挂载成功，终极防弹模式已激活。")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")

class ECGPayload(BaseModel):
    ecg: list[float]

@app.post("/api/predict")
@torch.inference_mode()
def predict(payload: ECGPayload):
    try:
        if len(payload.ecg) < 150000:
            ecg_array = np.pad(payload.ecg, (150000 - len(payload.ecg), 0), 'constant')
        else:
            ecg_array = np.array(payload.ecg[-150000:])

        # 🛡️ 防线 1：Z-Score 强制归一化，防止 FP16 精度计算时数值溢出变为 NaN
        std_val = np.std(ecg_array)
        if std_val < 1e-6:
            std_val = 1.0
        ecg_norm = (ecg_array - np.mean(ecg_array)) / std_val

        # 🚀 防线 2：绝对维度锁定！
        # 使用 view(1, 1, -1) 强行铸造 [Batch=1, Channel=1, Length=150000] 的 3D 张量
        bx = torch.tensor(ecg_norm, dtype=torch.float32).view(1, 1, -1).to(device)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            
            rhythm_logits = out["rhythm_logits"][0, -1]
            criticality_logits = out["criticality_logits"][0, -1]
            hazard_probs = out["hazard_probs"][0, -1]
            trajectory = out["hazard_probs"][0, -10:, 2]

            # 🛡️ 防线 3：物理切断任何潜在的 NaN/Inf，保护 FastAPI 的 JSON 序列化器
            rhythm_logits = torch.nan_to_num(rhythm_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            criticality_logits = torch.nan_to_num(criticality_logits, nan=0.0, posinf=5.0, neginf=-5.0)
            hazard_probs = torch.nan_to_num(hazard_probs, nan=0.0, posinf=1.0, neginf=0.0)
            trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=1.0, neginf=0.0)

        rhy_probs = F.softmax(rhythm_logits, dim=-1).cpu().tolist()
        cri_probs = F.softmax(criticality_logits, dim=-1).cpu().tolist()
        haz_probs_list = hazard_probs.cpu().tolist()
        traj_list = trajectory.cpu().tolist()

        return {
            "rhythm": rhy_probs,
            "criticality": cri_probs,
            "hazard": haz_probs_list,
            "risk_trajectory": traj_list
        }

    except Exception as e:
        err_detail = traceback.format_exc()
        print("\n" + "="*50)
        print("❌ 后端发生致命错误:")
        print(err_detail)
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)