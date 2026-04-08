# model_inference.py
import joblib
import numpy as np
from config import MODEL_PATH, SCALER_PATH, DISEASE_LABELS

class HealthPredictor:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.is_ready = True
        except FileNotFoundError:
            print("模型文件未找到，将返回模拟推理结果。")
            self.is_ready = False

    def predict(self, feature_vector):
        """执行模型推理与规则修正 [cite: 152, 153]"""
        # 1. 提取基础判别特征用于规则
        hr = feature_vector[0]      # mean_hr
        eda_std = feature_vector[10] # skin_conductance_std
        
        # 2. 基础模型预测
        if self.is_ready:
            scaled_feat = self.scaler.transform([feature_vector])
            probs = self.model.predict_proba(scaled_feat)[0]
            pred_class_idx = np.argmax(probs)
            pred_disease = DISEASE_LABELS[pred_class_idx]
            confidence = probs[pred_class_idx]
        else:
            # 若无真实模型，模拟一个正常状态输出
            pred_disease = "Normal"
            confidence = 0.85
            
        # 3. 规则修正引擎 (硬规则覆盖) [cite: 153]
        # 规则 1: 心率 > 130 -> 心律失常 [cite: 154]
        if hr > 130:
            pred_disease = "Arrhythmia"
            confidence = 0.90
        # 规则 2: 心率 < 60 -> 睡眠呼吸暂停 [cite: 155]
        elif hr < 60:
            pred_disease = "Sleep Apnea"
            confidence = 0.85
        # 规则 3: 90 < HR < 115 且 EDA std > 0.25 -> 压力过载 [cite: 156]
        elif 90 < hr < 115 and eda_std > 0.25:
            pred_disease = "Stress Overload"
            confidence = 0.90
        # 规则 4: 90 < HR < 115 且 EDA std <= 0.25 -> 自主神经紊乱 [cite: 157]
        elif 90 < hr < 115 and eda_std <= 0.25:
            pred_disease = "Autonomic Disorder"
            confidence = 0.70
            
        # 4. 预警分级判定 [cite: 117]
        alert_level = "Normal"
        if confidence >= 0.85 and pred_disease != "Normal":
            alert_level = "High"
        elif confidence >= 0.70 and pred_disease != "Normal":
            alert_level = "Warning"
            
        return {
            "disease": pred_disease,
            "confidence": round(confidence, 4),
            "alert_level": alert_level,
            "hr": round(hr, 1)
        }