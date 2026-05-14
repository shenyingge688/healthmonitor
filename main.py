"""
Script: backend.py (或 main.py / api.py)
Version: V10.0 (Master Clinical API)
功能: 加载 V10 最佳模型，提供三维层级推理服务
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np

# 导入你刚刚定型的 V10 网络架构
from dl_model import HierarchicalHazardNet

app = FastAPI(title="HealthMonitor V10 Clinical Engine")

# ==========================================
# 1. 初始化与挂载 V10 最佳模型
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在初始化 V10 临床推理引擎 | Device = {device}")

model = HierarchicalHazardNet().to(device)

# 🚨 挂载刚刚训练出炉的最佳权重 (关闭 weights_only 以允许读取 Numpy 指标)
ckpt_path = "models/v10_master_best.pth"
try:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"]) # 必须加载 EMA 的影子权重，最稳定！
    model.eval()
    print("✅ V10 模型权重加载成功，服务准备就绪。")
except Exception as e:
    print(f"❌ 模型加载失败，请确保 models/v10_master_best.pth 存在！错误: {e}")

# ==========================================
# 2. 定义请求体数据结构
# ==========================================
class ECGPayload(BaseModel):
    ecg: list[float]  # 前端传入的 150000 长度 (10分钟@250Hz) 的心电数据

# ==========================================
# 3. 核心推理接口
# ==========================================
@app.post("/api/predict")
@torch.inference_mode() # 极速推理模式
def predict(payload: ECGPayload):
    # 长度校验
    if len(payload.ecg) < 150000:
        # 如果长度不够，左侧补零补齐 10 分钟窗口
        ecg_array = np.pad(payload.ecg, (150000 - len(payload.ecg), 0), 'constant')
    else:
        ecg_array = np.array(payload.ecg[-150000:])

    # 转换张量并搬运到 GPU
    bx = torch.tensor(ecg_array, dtype=torch.float32).unsqueeze(0).to(device)

    # ⚡ AMP 混合精度推理
    with torch.amp.autocast("cuda"):
        out = model(bx)["preds"]

        # V10 模型输出形状均为 [Batch, Window=19, Classes]
        # 我们只取最后一个时间窗口（当下时刻）进行实时临床诊断
        rhythm_logits = out["rhythm_logits"][0, -1]
        criticality_logits = out["criticality_logits"][0, -1]
        hazard_probs = out["hazard_probs"][0, -1]

        # 提取恶化轨迹 (取过去 10 个窗口的 5分钟 Hazard 概率画趋势图)
        trajectory = out["hazard_probs"][0, -10:, 2].cpu().tolist()

    # 将 Logits 转换为概率 (Hazard 本身已经是概率了)
    rhy_probs = F.softmax(rhythm_logits, dim=-1).cpu().tolist()
    cri_probs = F.softmax(criticality_logits, dim=-1).cpu().tolist()
    haz_probs_list = hazard_probs.cpu().tolist()

    # 构建严格匹配 V10 前端 JSON 格式的响应
    return {
        "rhythm": rhy_probs,
        "criticality": cri_probs,
        "hazard": haz_probs_list,
        "risk_trajectory": trajectory
    }

# ==========================================
# 4. 启动服务
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)