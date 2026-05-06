"""
HealthMonitor V3.0 - FastAPI 推断网关
功能描述：提供接收实时心电序列的高频推断接口。
        具备会话隔离功能、防抖平滑处理、异常容错及 PyTorch 静态推理图挂载功能。
"""

import sys
import torch
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from dl_model import HybridWarningNet
from collections import deque, defaultdict

app = FastAPI(title="HealthMonitor Inference Engine")

class MultiClassRiskManager:
    """
    时序防抖管理器：
    在连续帧推理时，通过维护一个固定长度 (window_size) 的滑动窗口，
    对分类概率进行均值计算，以此消除单帧内由干扰产生的概率跳变。
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        # 预填充均匀分布，防止初次请求时缓冲区为空导致报错
        self.history = deque(
            [np.array([1.0/6.0]*6) for _ in range(window_size)], 
            maxlen=window_size
        )

    def reset(self):
        """用于切换病床或设备连接断开时清空旧数据"""
        self.history.clear()
        for _ in range(self.window_size):
            self.history.append(np.array([1.0/6.0]*6))

    def smooth_probabilities(self, current_probs):
        """传入当前帧的预测概率，返回平滑后的最终决策"""
        self.history.append(np.array(current_probs))
        avg_probs = np.mean(list(self.history), axis=0)
        smoothed_top1_class = int(np.argmax(avg_probs))
        smoothed_top1_prob = float(avg_probs[smoothed_top1_class])
        return smoothed_top1_class, smoothed_top1_prob, avg_probs.tolist()

# 使用 defaultdict 为不同的 device_id 分配独立的防抖队列，避免并发污染
risk_managers: dict[str, MultiClassRiskManager] = {}

def get_risk_manager(device_id: str) -> MultiClassRiskManager:
    if device_id not in risk_managers:
        risk_managers[device_id] = MultiClassRiskManager(window_size=5)
    return risk_managers[device_id]

# ==========================================
# [权重装载] 模型初始化与常驻内存
# ==========================================
model = HybridWarningNet()

try:
    model.load_state_dict(
        torch.load('models/hybrid_v5_massive_best.pth', map_location='cpu', weights_only=True)
    )
    model.eval()
    print("✅ 多分类推断引擎装载完毕，正在监听...")
except Exception as e:
    # 捕获异常后立即阻断进程启动，避免空转
    print(f"❌ 致命错误：核心权重装载失败，已切断服务。原因: {e}")
    sys.exit(1)

# ==========================================
# [API网关] 推理端点
# ==========================================
@app.post("/api/predict")
def predict_future(data: dict, device_id: str = Header("default-device", alias="X-Device-ID")):
    """
    接收来自前端/硬件的长序列数组，返回 6 分类置信度与 CAM 特征数组。
    注意：没有使用 async def，使得高耗时同步推断自动分配到后台线程池。
    """
    ecg_clean = np.array(data.get('ecg', []))
    TARGET_LEN = 150000

    # 输入有效性验证
    if len(ecg_clean) == 0 or np.isnan(ecg_clean).any() or np.isinf(ecg_clean).any():
        raise HTTPException(status_code=400, detail="检测到无效的心电数据 (Empty / NaN / Inf)")

    # 序列维度对齐操作 (裁剪或填充)
    if len(ecg_clean) != TARGET_LEN:
        if len(ecg_clean) < TARGET_LEN:
            ecg_clean = np.pad(ecg_clean, (0, TARGET_LEN - len(ecg_clean)), 'constant')
        else:
            ecg_clean = ecg_clean[-TARGET_LEN:]

    # 导联脱落检测 (基于极低方差)
    if np.std(ecg_clean) < 1e-8:
        raise HTTPException(status_code=400, detail="信号方差过低，疑似导联脱落")

    # 数据归一化及张量重构
    ecg_norm = (ecg_clean - np.mean(ecg_clean)) / (np.std(ecg_clean) + 1e-8)
    input_tensor = torch.tensor(ecg_norm.reshape(1, 300, 1, 500), dtype=torch.float32)

    # 隔离梯度的纯粹推断
    with torch.no_grad():
        outputs = model(input_tensor)

    probs = outputs["prob"][0].cpu().numpy()
    cam_array = outputs["cam"].cpu().numpy().flatten().tolist()
    
    # 获取专属状态机，平滑当前帧预测
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
    """供外部调用的清除指定设备状态的端点"""
    if device_id in risk_managers:
        risk_managers[device_id].reset()
        return {"status": "ok", "message": f"Session {device_id} reset."}
    return {"status": "ok", "message": f"No active session for {device_id}."}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "active_sessions": len(risk_managers)}