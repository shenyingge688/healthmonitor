"""
模块名称：后端推理服务接口 (FastAPI - 多分类重构版)
"""
from fastapi import FastAPI
import torch
import numpy as np
from dl_model import HybridWarningNet

app = FastAPI()

import numpy as np
from collections import deque

# ==========================================
# 多分类临床状态防抖管理器 
# ==========================================
class MultiClassRiskManager:
    """
    功能：对多分类的概率数组进行滑动时间窗平滑，消除瞬态伪影导致的警报反复跳变。
    """
    def __init__(self, window_size=5):
        # 维护一个队列，存储最近 N 帧的 6 分类概率数组
        self.history = deque(maxlen=window_size)
        
    def smooth_probabilities(self, current_probs):
        """输入当前帧的 6 个概率，输出平滑后的 6 个概率及最终定性"""
        self.history.append(current_probs)
        
        # 计算时间窗内各个类别的平均概率
        avg_probs = np.mean(list(self.history), axis=0)
        
        # 基于平滑后的均值重新进行 Top-1 判定
        smoothed_top1_class = int(np.argmax(avg_probs))
        smoothed_top1_prob = float(avg_probs[smoothed_top1_class])
        
        return smoothed_top1_class, smoothed_top1_prob, avg_probs.tolist()

# 实例化管理器 (保留最近 5 次预测进行平滑，约等于平滑过去数秒的决策)
risk_manager = MultiClassRiskManager(window_size=5)

model = HybridWarningNet()

# ==========================================
# 模型引擎挂载
# ==========================================
try:
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True))
    model.eval()
    print("✅ 深度学习多分类诊断引擎装载完毕，正在监听 8000 端口...")
except Exception as e:
    print(f"❌ 引擎装载失败！系统抛出错误: {e}")

# ==========================================
# API 路由与推理流程
# ==========================================
@app.post("/api/predict")
async def predict_future(data: dict):
    # 1. 提取信号并执行归一化
    ecg_clean = np.array(data['ecg'])
    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    
    # 2. 格式化送入网络
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        
    # 3. 提取瞬时结果与 CAM
    probs = outputs["prob"][0].cpu().numpy()
    cam_array = outputs["cam"].cpu().numpy().flatten().tolist()
    
    # 4. 经过平滑状态机处理，拦截瞬时毛刺
    final_class, final_prob, final_all_probs = risk_manager.smooth_probabilities(probs)
    
    # 5. 返回稳定的平滑数据给前端
    return {
        "pred_class": final_class,
        "future_risk_prob": final_prob,
        "all_probs": final_all_probs,
        "cam_heatmap": cam_array
    }