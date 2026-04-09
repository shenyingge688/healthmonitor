import wfdb
import numpy as np
import torch
import random
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os

# 保持 250Hz 采样
target_fs = 250    
HISTORY_PTS = 600 * target_fs
PREDICT_PTS = 300 * target_fs
STRIDE_SEC = 60

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

def build_balanced_dataset(record_list, name, risk_threshold=20):
    X_normal, X_risk = [], []
    print(f"\n📂 正在构建数据集: {name} (目标：平衡类别)")
    
    for rec in tqdm(record_list, desc="读取记录", unit="rec", ncols=100):
        try:
            record = wfdb.rdrecord(rec, pn_dir='mitdb')
            annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
            raw = record.p_signal[:, 0]
            clean_raw = clean_ecg_signal(raw, fs=360) 
            ecg = signal.resample(clean_raw, int(len(clean_raw) * (250 / 360)))
            
            anno_idx = np.round(annotation.sample * (250 / 360)).astype(int)
            risk_idx = anno_idx[~np.isin(np.array(annotation.symbol), ['N', '.', '/'])]

            for start in range(0, len(ecg) - HISTORY_PTS - PREDICT_PTS, STRIDE_SEC * target_fs):
                predict_start = start + HISTORY_PTS
                predict_end = predict_start + PREDICT_PTS
                
                # 计算异常点数
                risk_hits = np.sum((risk_idx >= predict_start) & (risk_idx < predict_end))
                x_win = ecg[start : predict_start]
                x_norm = (x_win - np.mean(x_win)) / (np.std(x_win) + 1e-8)
                sample = x_norm.reshape(300, 1, 500)

                # 分类存放
                if risk_hits >= risk_threshold:
                    X_risk.append(sample)
                else:
                    X_normal.append(sample)
        except Exception: continue
    
    # 🎯 核心平衡逻辑：欠采样
    num_samples = min(len(X_normal), len(X_risk))
    print(f"原始统计 -> 正常: {len(X_normal)}, 风险: {len(X_risk)}")
    
    # 随机抽取相等数量的样本，强制 1:1
    X_final = random.sample(X_normal, num_samples) + random.sample(X_risk, num_samples)
    Y_final = [0.0] * num_samples + [1.0] * num_samples
    
    # 打乱顺序
    combined = list(zip(X_final, Y_final))
    random.shuffle(combined)
    X_final, Y_final = zip(*combined)

    if X_final:
        os.makedirs('dataset', exist_ok=True)
        torch.save({'X': torch.tensor(np.array(X_final), dtype=torch.float32), 
                    'Y': torch.tensor(np.array(Y_final), dtype=torch.float32).unsqueeze(1)}, f'dataset/{name}.pt')
        print(f"📊 {name} 平衡完成。最终样本数: {len(X_final)}, 风险占比: 50.0%")

# 执行构建
demo_cases = ['100', '119', '201', '208', '233']
all_recs = wfdb.get_record_list('mitdb')
train_recs = [r for r in all_recs if r not in (demo_cases + ['102', '104', '107', '217'])]

# 建议将 threshold 设为 20，确保 100 号记录彻底被判定为正常
build_balanced_dataset(train_recs, 'train_massive_v5', risk_threshold=20)
build_balanced_dataset(demo_cases, 'val_demo_v5', risk_threshold=20)