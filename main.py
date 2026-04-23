"""
模块名称：后端推理服务接口 (FastAPI Inference Service)
模块功能：
    1. 实例化 AI 模型网络，并装载经过训练的最佳权重参数 (.pth)。
    2. 提供基于 HTTP 协议的 /api/predict 预测端点。
    3. 接收前端实时推流的心电数组，进行物理标准化预处理。
    4. 集成 RiskManager 平滑队列，抹除瞬态预测毛刺，返回稳定的临床警报级别和可解释热力图。
"""

from fastapi import FastAPI
import torch
import numpy as np
from collections import deque
from dl_model import HybridWarningNet

app = FastAPI()

# ==========================================
# 临床状态防抖管理器
# ==========================================
class RiskManager:
    """
    功能：利用滑动窗口平均法对模型的瞬时输出概率进行平滑处理，防止前端警报频繁闪烁。
    阈值定义：
        0.00 ~ 0.60 -> Normal (处于安全静默区)
        0.60 ~ 0.85 -> Warning (出现中度异常波动)
        > 0.85      -> High (存在高度失稳畸变风险)
    """
    def __init__(self, window_size=5, warning_th=0.60, high_th=0.85):
        self.history = deque(maxlen=window_size) 
        self.warning_th = warning_th
        self.high_th = high_th

    def update_and_get_level(self, raw_prob):
        """输入当前帧概率，输出平滑去噪后的综合警报等级与均值概率"""
        self.history.append(raw_prob)
        avg_prob = sum(self.history) / len(self.history)
        
        if avg_prob >= self.high_th:
            return "High", avg_prob
        elif avg_prob >= self.warning_th:
            return "Warning", avg_prob
        return "Normal", avg_prob

# 初始化平滑管理器与空模型
risk_manager = RiskManager()
model = HybridWarningNet()

# ==========================================
# 模型引擎挂载
# ==========================================
try:
    # 启用 weights_only=True 保障安全加载
    model.load_state_dict(torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True))
    model.eval()
    print("✅ 深度学习超前预警引擎装载完毕，正在监听 8001 端口...")
except Exception as e:
    print(f"❌ 引擎装载失败！请确认模型训练是否完成。系统抛出错误: {e}")

# ==========================================
# API 路由与推理流程
# ==========================================
@app.post("/api/predict")
async def predict_future(data: dict):
    """
    接收并处理前端心电推流，返回分析结果字典。
    """
    # 1. 提取信号并执行临床标准的均值方差归一化
    ecg_clean = np.array(data['ecg'])
    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    
    # 2. 格式化张量并送入网络结构: [Batch(1), SeqLen(300), Channel(1), Points(500)]
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(input_tensor)

    # 3. 提取结果：瞬时预测概率与对应的时序热力图
    raw_prob = outputs["prob"].item()
    cam_array = outputs["cam"].cpu().numpy().flatten().tolist() 

    # 4. 状态机平滑评估，拦截毛刺假阳性
    level, smoothed_prob = risk_manager.update_and_get_level(raw_prob)

    # 5. 构建响应字典返回给前端业务系统
    return {
        "future_risk_prob": smoothed_prob, 
        "alert_level": level,              
        "cam_heatmap": cam_array           
    }