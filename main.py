"""
HealthMonitor V3.0 - FastAPI 推断网关
"""
import sys
import torch
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from dl_model import HybridWarningNet
from collections import deque
from cachetools import TTLCache # 【修复】导入 TTL 缓存防内存泄漏

app = FastAPI(title="HealthMonitor Inference Engine")

class MultiClassRiskManager:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(
            [np.array([1.0/6.0]*6) for _ in range(window_size)], 
            maxlen=window_size
        )

    def reset(self):
        self.history.clear()
        for _ in range(self.window_size):
            self.history.append(np.array([1.0/6.0]*6))

    def smooth_probabilities(self, current_probs):
        self.history.append(np.array(current_probs))
        avg_probs = np.mean(list(self.history), axis=0)
        smoothed_top1_class = int(np.argmax(avg_probs))
        smoothed_top1_prob = float(avg_probs[smoothed_top1_class])
        return smoothed_top1_class, smoothed_top1_prob, avg_probs.tolist()

# 【修复】使用 TTLCache：最大追踪1000个设备，若设备10分钟（600秒）未发送请求则自动销毁状态
risk_managers: TTLCache = TTLCache(maxsize=1000, ttl=600)

def get_risk_manager(device_id: str) -> MultiClassRiskManager:
    if device_id not in risk_managers:
        risk_managers[device_id] = MultiClassRiskManager(window_size=5)
    return risk_managers[device_id]

model = HybridWarningNet()

try:
    model.load_state_dict(
        torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True)
    )
    model.eval()
    print("✅ 多分类推断引擎装载完毕，正在监听...")
except Exception as e:
    print(f"❌ 致命错误：核心权重装载失败，已切断服务。原因: {e}")
    sys.exit(1)

@app.post("/api/predict")
def predict_future(data: dict, device_id: str = Header("default-device", alias="X-Device-ID")):
    ecg_clean = np.array(data.get('ecg', []))
    TARGET_LEN = 150000

    if len(ecg_clean) == 0 or np.isnan(ecg_clean).any() or np.isinf(ecg_clean).any():
        raise HTTPException(status_code=400, detail="检测到无效的心电数据")

    if len(ecg_clean) != TARGET_LEN:
        if len(ecg_clean) < TARGET_LEN:
            ecg_clean = np.pad(ecg_clean, (0, TARGET_LEN - len(ecg_clean)), 'constant')
        else:
            ecg_clean = ecg_clean[-TARGET_LEN:]

    if np.std(ecg_clean) < 1e-8:
        raise HTTPException(status_code=400, detail="信号方差过低，疑似导联脱落")

    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)

    probs = outputs["prob"][0].cpu().numpy()
    cam_array = outputs["cam"].cpu().numpy().flatten().tolist()
    
    local_manager = get_risk_manager(device_id)
    final_class, final_prob, final_all_probs = local_manager.smooth_probabilities(probs)

    return {
        "pred_class": final_class,
        "future_risk_prob": final_prob,
        "all_probs": final_all_probs,
        "cam_heatmap": cam_array
    }

@app.post("/api/reset/{device_id}")
def reset_risk_manager(device_id: str):
    if device_id in risk_managers:
        risk_managers[device_id].reset()
        return {"status": "ok", "message": f"Session {device_id} reset."}
    return {"status": "ok", "message": f"No active session for {device_id}."}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "active_sessions": len(risk_managers)}