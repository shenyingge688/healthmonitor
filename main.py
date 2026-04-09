from fastapi import FastAPI
import torch
import numpy as np
from collections import deque
from dl_model import HybridWarningNet

app = FastAPI()

class RiskManager:
    # 🎯 临床刻度重置：
    # 0.00 ~ 0.60 (Normal) - 允许 50% 的摇摆概率存在于静默区
    # 0.60 ~ 0.85 (Warning) - 中度形态偏移
    # > 0.85 (High) - 确定性极高的高危畸变
    def __init__(self, window_size=5, warning_th=0.60, high_th=0.85):
        self.history = deque(maxlen=window_size)
        self.warning_th = warning_th
        self.high_th = high_th
        
    def update_and_get_level(self, raw_prob):
        self.history.append(raw_prob)
        avg_prob = sum(self.history) / len(self.history)
        
        if avg_prob >= self.high_th: 
            return "High", avg_prob
        elif avg_prob >= self.warning_th: 
            return "Warning", avg_prob
        return "Normal", avg_prob

risk_manager = RiskManager()
model = HybridWarningNet()

try:
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True))
    model.eval()
    print("✅ 超前预警引擎装载完毕：监听端口 8001。")
except Exception as e:
    print(f"❌ 引擎加载失败，请确保已重新训练模型。错误: {e}")

@app.post("/api/predict")
async def predict_future(data: dict):
    ecg_clean = np.array(data['ecg'])
    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)
    
    with torch.no_grad():
        _, risk_prob = model(input_tensor) 
        
    level, smoothed = risk_manager.update_and_get_level(risk_prob.item())
    
    return {
        "future_risk_prob": smoothed, 
        "alert_level": level
    }