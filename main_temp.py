"""
-> (FastAPI - ->)
"""
from fastapi import FastAPI
import torch
import numpy as np
from dl_model import HybridWarningNet

app = FastAPI()

import numpy as np
from collections import deque

# ==========================================
# -> 
# ==========================================
class MultiClassRiskManager:
    """
    ->
    """
    def __init__(self, window_size=5):
        # -> N -> 6 ->
        self.history = deque(maxlen=window_size)
        
    def smooth_probabilities(self, current_probs):
        """-> 6 -> 6 ->"""
        self.history.append(current_probs)
        
        # ->
        avg_probs = np.mean(list(self.history), axis=0)
        
        # -> Top-1 ->
        smoothed_top1_class = int(np.argmax(avg_probs))
        smoothed_top1_prob = float(avg_probs[smoothed_top1_class])
        
        return smoothed_top1_class, smoothed_top1_prob, avg_probs.tolist()

# -> (-> 5 ->)
risk_manager = MultiClassRiskManager(window_size=5)

model = HybridWarningNet()

# ==========================================
# ->
# ==========================================
try:
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True))
    model.eval()
    print("-> -> 8000 ->...")
except Exception as e:
    print(f"-> ->: {e}")

# ==========================================
# API ->
# ==========================================
@app.post("/api/predict")
async def predict_future(data: dict):
    # 1. ->
    ecg_clean = np.array(data['ecg'])
    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    
    # 2. ->
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        
    # 3. -> CAM
    probs = outputs["prob"][0].cpu().numpy()
    cam_array = outputs["cam"].cpu().numpy().flatten().tolist()
    
    # 4. ->
    final_class, final_prob, final_all_probs = risk_manager.smooth_probabilities(probs)
    
    # 5. ->
    return {
        "pred_class": final_class,
        "future_risk_prob": final_prob,
        "all_probs": final_all_probs,
        "cam_heatmap": cam_array
    }