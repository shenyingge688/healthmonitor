# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from signal_processor import SignalProcessor
from model_inference import HealthPredictor
import config
import numpy as np
app = FastAPI(title="多模态健康预警系统 API")

processor = SignalProcessor(sampling_rate=config.SAMPLING_RATE)
predictor = HealthPredictor()

class SignalPayload(BaseModel):
    ecg: List[float]
    ppg: List[float]
    eda: List[float]

@app.post("/api/predict")
async def analyze_window(payload: SignalPayload):
    """
    接收前端或 ESP32 传入的 3 秒滑动窗口数据 [cite: 139]
    执行 12 维特征提取与疾病分类
    """
    # 验证数据长度
    if len(payload.ecg) < config.WINDOW_POINTS:
        return {"error": "Insufficient data points for 3s window"}
        
    # 1. 提取 12 维特征
    features = processor.extract_features(
        np.array(payload.ecg), 
        np.array(payload.ppg), 
        np.array(payload.eda)
    )
    
    # 2. 模型推理与规则修正
    result = predictor.predict(features)
    
    # 3. 此处应将 result 写入 SQLite 数据库的 predictions 表 [cite: 142]
    # TODO: Insert into SQLite
    
    return {
        "status": "success",
        "features_extracted": len(features),
        "prediction": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)



