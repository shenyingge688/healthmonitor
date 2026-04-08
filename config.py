# config.py
import os

# 信号采集配置
SAMPLING_RATE = 250  # 采样率 (Hz) [cite: 118]
WINDOW_SIZE_SEC = 3  # 滑动窗口时长 (秒) [cite: 139]
WINDOW_POINTS = SAMPLING_RATE * WINDOW_SIZE_SEC  # 窗口内的数据点数 (750点)

# 疾病标签定义 
DISEASE_LABELS = {
    0: "Normal",
    1: "Arrhythmia",
    2: "Sleep Apnea",
    3: "Stress Overload",
    4: "Autonomic Disorder"
}

# 数据库与模型路径
DB_URL = "sqlite:///./health_monitor.db"
MODEL_PATH = os.path.join("models", "rf_model_5class.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")