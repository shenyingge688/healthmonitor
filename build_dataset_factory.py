"""
HealthMonitor V3.0 - 数据获取与标签制作工厂
功能描述：从 PhysioNet 开源库解析临床序列与标注信息，依据设定滑动窗口提取输入特征，
        同时利用状态机裁决每个10分钟片段归属哪种主要心律失常特征，并执行软标签平滑处理。
"""

import wfdb
import numpy as np
import torch
import random
import os
import sys
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'mitdb')
SAVE_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_FS = 250
HISTORY_PTS = 600 * TARGET_FS   # 模型输入观测窗口 (10 分钟)
PREDICT_PTS = 300 * TARGET_FS   # 提取目标标签的未来窗口 (5 分钟)
STRIDE_SEC = 60                 # 滑动步长

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

def build_balanced_dataset(record_list, name, is_train=True):
    X_all, Y_all = [], []
    pbar = tqdm(record_list, desc=f"构建数据集 [{name}]", unit="rec", file=sys.stdout)
    
    for rec in pbar:
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            clean_raw = clean_ecg_signal(record.p_signal[:, 0], fs=360)
            ecg = signal.resample_poly(clean_raw, TARGET_FS, 360)
            anno_idx = np.round(annotation.sample * (TARGET_FS / 360)).astype(int)
            
            # --- 状态机机制解析长节律标注 ---
            current_rhythm = 0
            anno_classes_list = []
            
            for sym, aux in zip(annotation.symbol, annotation.aux_note):
                # 如果发生长期的宏观节律变化，更新基础状态
                if isinstance(aux, str) and aux.startswith('('):
                    rhythm_str = aux.upper()
                    if 'VF' in rhythm_str or 'VFIB' in rhythm_str: current_rhythm = 3
                    elif 'VT' in rhythm_str or 'VFL' in rhythm_str: current_rhythm = 4
                    elif 'AFIB' in rhythm_str: current_rhythm = 2
                    elif 'AFL' in rhythm_str or 'AT' in rhythm_str or 'SVT' in rhythm_str: current_rhythm = 5
                    elif 'B' in rhythm_str or 'T' in rhythm_str: current_rhythm = 1
                    elif 'N' in rhythm_str or 'NSR' in rhythm_str: current_rhythm = 0
                
                # 为该标注点打上具体类别，优先遵循宏观状态
                if current_rhythm in [2, 3, 4, 5]:
                    beat_class = current_rhythm
                else:
                    if sym in ['V', 'E', 'r']: beat_class = 1
                    elif sym in ['A', 'a', 'S', 'J']: beat_class = 5
                    elif sym in ['N', '.', '/', 'L', 'R']: beat_class = 0
                    else: beat_class = -1
                anno_classes_list.append(beat_class)
                
            anno_classes = np.array(anno_classes_list)
            valid_mask = anno_classes != -1
            valid_anno_idx = anno_idx[valid_mask]
            valid_anno_classes = anno_classes[valid_mask]
            
            # --- 滑动截取窗口并计算对应裁决标签 ---
            for start in range(0, len(ecg) - HISTORY_PTS - PREDICT_PTS, STRIDE_SEC * TARGET_FS):
                predict_start = start + HISTORY_PTS
                predict_end = predict_start + PREDICT_PTS
                
                # 获取未来五分钟内出现的所有心律活动标签
                window_mask = (valid_anno_idx >= predict_start) & (valid_anno_idx < predict_end)
                classes_in_window = valid_anno_classes[window_mask]
                
                x_raw = ecg[start : predict_start]
                x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                sample = x_norm.reshape(300, 1, 500)
                
                # 根据不同类别的优先级确定整个时段的主要标签
                if len(classes_in_window) == 0:
                    final_label = 0
                else:
                    unique_classes, counts = np.unique(classes_in_window, return_counts=True)
                    class_counts = dict(zip(unique_classes, counts))
                    
                    # 首先满足严重致死类指标一票裁决
                    for extreme_class in [3, 4, 2]:
                        if extreme_class in class_counts:
                            final_label = extreme_class
                            break
                    else:
                        # 对于早搏等则遵循密度频发原则裁定
                        pvc_count = class_counts.get(1, 0)
                        at_count = class_counts.get(5, 0)
                        if at_count >= 2: final_label = 5
                        elif pvc_count >= 3: final_label = 1
                        else: final_label = 0
                
                X_all.append(sample)
                Y_all.append(final_label)
                
        except Exception:
            continue
            
    # --- 构建分布缓冲，使用部分欠采样压制多数类 ---
    class_samples = {i: [] for i in range(6)}
    for x, y in zip(X_all, Y_all):
        class_samples[y].append(x)
        
    severe_abnormal_count = sum([len(class_samples[i]) for i in range(2, 6)])
    
    max_pvc_samples = max(200, int(severe_abnormal_count * 2.5))
    if len(class_samples[1]) > max_pvc_samples:
        class_samples[1] = random.sample(class_samples[1], max_pvc_samples)
        
    total_abnormal_now = sum([len(class_samples[i]) for i in range(1, 6)])
    max_normal_samples = max(200, int(total_abnormal_now * 3.0))
    if len(class_samples[0]) > max_normal_samples:
        class_samples[0] = random.sample(class_samples[0], max_normal_samples)
        
    X_final, Y_final = [], []
    for c, samples in class_samples.items():
        X_final.extend(samples)
        Y_final.extend([c] * len(samples))
        
    # 执行 Label Smoothing 软标签生成策略
    Y_smoothed = []
    for y in Y_final:
        smoothed_label = [0.02] * 6
        smoothed_label[y] = 0.90
        Y_smoothed.append(smoothed_label)
        
    combined = list(zip(X_final, Y_smoothed))
    random.shuffle(combined)
    
    if not combined:
        return
        
    X_final, Y_smoothed = zip(*combined)
    
    X_tensor = torch.empty((len(X_final), 300, 1, 500), dtype=torch.float32)
    for i, x_arr in enumerate(X_final):
        X_tensor[i] = torch.from_numpy(x_arr)
        
    Y_tensor = torch.tensor(list(Y_smoothed), dtype=torch.float32)
    
    # 内存释放动作
    del X_final, Y_smoothed, combined
    gc.collect()
    
    save_path = os.path.join(SAVE_DIR, f'{name}.pt')
    torch.save({'X': X_tensor, 'Y': Y_tensor}, save_path)
    pbar.write(f"✅ 生成 {name}.pt 成功 (总数: {len(X_tensor)})")

if __name__ == '__main__':
    print("🚀 正在全局扫描并加载数据库...")
    db_names = ['mitdb', 'cudb', 'vfdb', 'afdb']
    all_recs_paths = []
    for db in db_names:
        db_dir = os.path.join(BASE_DIR, 'data', db)
        if os.path.exists(db_dir):
            recs = [os.path.join(db_dir, f.split('.')[0]) for f in os.listdir(db_dir) if f.endswith('.dat')]
            all_recs_paths.extend(recs)
            
    demo_cases = ['100', '119', '201', '207', '209']
    leak_cases = demo_cases + ['102', '104', '107', '217']
    safe_recs = [r for r in all_recs_paths if os.path.basename(r) not in leak_cases]
    
    # 执行基于病人身份的物理验证切分策略
    random.seed(42)
    random.shuffle(safe_recs)
    split_idx = int(0.8 * len(safe_recs))
    train_recs = safe_recs[:split_idx]
    test_recs = safe_recs[split_idx:]
    
    build_balanced_dataset(train_recs, 'train_massive_v5', is_train=True)
    build_balanced_dataset(test_recs, 'test_pure_v5', is_train=False)