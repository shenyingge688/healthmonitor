"""
HealthMonitor V3.0 - 硬件直通接口验证脚本
功能描述：从本地采集的示波器 csv 文档读入测试点，清洗后转发至后端接口以效验物理通信回路。
"""

import pandas as pd
import numpy as np
import requests
from scipy import signal
from scipy.signal import butter, filtfilt

csv_path = "prosim_vt_test.csv"
try:
    df = pd.read_csv(csv_path)
    raw_voltage = df['Voltage_mV'].values
except Exception as e:
    print(f"❌ 物理日志读取阻断: {e}")
    exit(1)

ORIGINAL_FS = 1000
TARGET_FS = 250

# 下行硬件采样率与系统内定采样率对齐重置
ecg_resampled = signal.resample_poly(raw_voltage, TARGET_FS, ORIGINAL_FS) if ORIGINAL_FS != TARGET_FS else raw_voltage

def clean_ecg_signal(data, fs=250):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

ecg_filtered = clean_ecg_signal(ecg_resampled, fs=TARGET_FS)
ecg_norm = (ecg_filtered - np.mean(ecg_filtered)) / (np.std(ecg_filtered) + 1e-8)
print(f"✅ 示波器信号整理挂载完成，总点阵规模: {len(ecg_norm)} 点")

def test_via_api(ecg_array):
    # 抽取尾端时序送入接口检验
    payload_data = ecg_array[-150000:] if len(ecg_array) > 150000 else ecg_array
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/api/predict", 
            json={"ecg": payload_data.tolist()},
            headers={"X-Device-ID": "PROSIM_01"}
        )
        if resp.status_code == 200:
            result = resp.json()
            print("🔗 网关回执结果呈现:")
            print(f"- 最优指向分类: Class {result['pred_class']}")
            print(f"- 置信阀值: {result['future_risk_prob']*100:.2f}%")
            print(f"- 分布光谱图: {np.round(np.array(result['all_probs'])*100, 2)}")
        else:
            print(f"⚠️ 状态回执响应非预期: {resp.status_code}")
    except Exception as e:
        print(f"❌ 建立通信渠道失败: {e}")

test_via_api(ecg_norm)