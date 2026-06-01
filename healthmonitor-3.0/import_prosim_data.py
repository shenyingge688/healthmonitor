import pandas as pd
import numpy as np
import requests
from scipy import signal
from scipy.signal import butter, filtfilt

# ==========================================
# 1. 读取示波器导出的 CSV 文件
# ==========================================
# 假设您的 CSV 文件有两列：'Time' 和 'Voltage_mV'
csv_path = "prosim_vt_test.csv" 
df = pd.read_csv(csv_path)

# 提取电压幅值（如果示波器导出的是 V，需要乘以 1000 转为 mV）
raw_voltage = df['Voltage_mV'].values 

# ==========================================
# 2. 信号预处理 (严格对齐 HybridWarningNet V3.0 规范)
# ==========================================
ORIGINAL_FS = 1000  # 替换为您的示波器实际导出采样率
TARGET_FS = 250     # 模型强制要求 250Hz

# (1) 降采样/重采样至 250Hz
if ORIGINAL_FS != TARGET_FS:
    ecg_resampled = signal.resample_poly(raw_voltage, TARGET_FS, ORIGINAL_FS)
else:
    ecg_resampled = raw_voltage

# (2) 临床级带通滤波 (0.5 - 45Hz)
def clean_ecg_signal(data, fs=250):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

ecg_filtered = clean_ecg_signal(ecg_resampled, fs=TARGET_FS)

# (3) 全局标准化
ecg_norm = (ecg_filtered - np.mean(ecg_filtered)) / (np.std(ecg_filtered) + 1e-8)

print(f"✅ 示波器数据解析成功！总长度: {len(ecg_norm)} 个数据点 (约 {len(ecg_norm)/250:.1f} 秒)")

# ==========================================
# 3. 对接与测试方式
# ==========================================
# 方式 A：直接将数据发送给本地运行的 FastAPI 后端进行测试
def test_via_api(ecg_array):
    # 截取最后 150000 个点 (10分钟) 模拟实时推流，如果数据不够长，则发送全部
    payload_data = ecg_array[-150000:] if len(ecg_array) > 150000 else ecg_array
    
    try:
        resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": payload_data.tolist()})
        if resp.status_code == 200:
            result = resp.json()
            print("🏥 后端 API 推断结果:")
            print(f"- 最可能分类: Class {result['pred_class']}")
            print(f"- 核心置信度: {result['future_risk_prob']*100:.2f}%")
            print(f"- 6分类全景概率: {np.round(np.array(result['all_probs'])*100, 2)}")
    except Exception as e:
        print(f"API 请求失败: {e}")

test_via_api(ecg_norm)

# 方式 B：将处理好的数据保存为 .npy，以便前端 dashboard.py 读取
# np.save("prosim_custom_signal.npy", ecg_norm)