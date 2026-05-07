"""
HealthMonitor V4.0 - 数据获取与多任务标签制作工厂
功能描述：从 PhysioNet 开源库解析临床序列与标注信息。
        动态匹配高质量导联，同时提取“当下10分钟”与“未来5分钟”的双重标签，
        并生成 HRV 预留特征，服务于 Micro-Meso-Macro 架构及多任务学习。
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

def get_interval_label(classes_in_window, default_rhythm):
    """根据窗口内包含的心搏标注，裁决该窗口的最终分类"""
    if len(classes_in_window) == 0:
        return default_rhythm
        
    unique_classes, counts = np.unique(classes_in_window, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    
    for extreme_class in [3, 4, 2]:
        if extreme_class in class_counts:
            return extreme_class
    else:
        pvc_count = class_counts.get(1, 0)
        at_count = class_counts.get(5, 0)
        if at_count >= 2: return 5
        elif pvc_count >= 3: return 1
        else: return default_rhythm

def build_balanced_dataset(record_list, name, is_train=True):
    X_all, Y_future_all, Y_current_all, HRV_all = [], [], [], []
    pbar = tqdm(record_list, desc=f"构建数据集 [{name}]", unit="rec", file=sys.stdout)
    
    for rec in pbar:
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            
            # 【优化 1】动态导联匹配，避免 MLII 和 V1 混用导致模型认知错乱
            sig_names = record.sig_name
            target_idx = 0
            for idx, s_name in enumerate(sig_names):
                if s_name.upper() in ['MLII', 'II', 'LEAD II', 'V5']:
                    target_idx = idx
                    break
                    
            clean_raw = clean_ecg_signal(record.p_signal[:, target_idx], fs=360)
            ecg = signal.resample_poly(clean_raw, TARGET_FS, 360)
            anno_idx = np.round(annotation.sample * (TARGET_FS / 360)).astype(int)
            
            # --- 状态机机制解析长节律标注 ---
            current_rhythm = 0
            anno_classes_list = []
            
            for sym, aux in zip(annotation.symbol, annotation.aux_note):
                if isinstance(aux, str) and aux.startswith('('):
                    rhythm_str = aux.upper()
                    if 'VF' in rhythm_str or 'VFIB' in rhythm_str: current_rhythm = 3
                    elif 'VT' in rhythm_str or 'VFL' in rhythm_str: current_rhythm = 4
                    elif 'AFIB' in rhythm_str: current_rhythm = 2
                    elif 'AFL' in rhythm_str or 'AT' in rhythm_str or 'SVT' in rhythm_str: current_rhythm = 5
                    elif 'B' in rhythm_str or 'T' in rhythm_str: current_rhythm = 1
                    elif 'N' in rhythm_str or 'NSR' in rhythm_str: current_rhythm = 0
                
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
                current_start = start
                current_end = start + HISTORY_PTS
                predict_start = current_end
                predict_end = predict_start + PREDICT_PTS
                
                # 获取当下10分钟与未来5分钟的独立标签掩码
                current_mask = (valid_anno_idx >= current_start) & (valid_anno_idx < current_end)
                future_mask = (valid_anno_idx >= predict_start) & (valid_anno_idx < predict_end)
                
                final_current_label = get_interval_label(valid_anno_classes[current_mask], current_rhythm)
                final_future_label = get_interval_label(valid_anno_classes[future_mask], current_rhythm)
                
                x_raw = ecg[current_start : current_end]
                x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                sample = x_norm.astype(np.float32).reshape(300, 1, 500)
                
                # 【优化 2】伪 HRV 特征提取 (占位10维)，供后续 Transformer 宏观层使用
                dummy_hrv = np.random.normal(0, 1, 10).astype(np.float32)
                
                X_all.append(sample)
                HRV_all.append(dummy_hrv)
                Y_future_all.append(final_future_label)
                Y_current_all.append(final_current_label)
                
        except Exception:
            continue
            
    # --- 构建分布缓冲，使用欠采样压制多数类 (基于主要预测任务：Future Label) ---
    indices_by_class = {i: [] for i in range(6)}
    for idx, y_fut in enumerate(Y_future_all):
        indices_by_class[y_fut].append(idx)
        
    severe_count = sum(len(indices_by_class[i]) for i in range(2, 6))
    max_pvc = max(200, int(severe_count * 2.5))
    if len(indices_by_class[1]) > max_pvc:
        indices_by_class[1] = random.sample(indices_by_class[1], max_pvc)
        
    total_abn = sum(len(indices_by_class[i]) for i in range(1, 6))
    max_normal = max(200, int(total_abn * 3.0))
    if len(indices_by_class[0]) > max_normal:
        indices_by_class[0] = random.sample(indices_by_class[0], max_normal)
        
    final_indices = []
    for c in range(6):
        final_indices.extend(indices_by_class[c])
    random.shuffle(final_indices)
    
    if not final_indices:
        return
        
    # 执行 Label Smoothing
    def smooth_label(y):
        sl = [0.02] * 6
        sl[y] = 0.90
        return sl
        
    X_tensor = torch.empty((len(final_indices), 300, 1, 500), dtype=torch.float32)
    HRV_tensor = torch.empty((len(final_indices), 10), dtype=torch.float32)
    Y_future_list, Y_current_list = [], []
    
    for i, orig_idx in enumerate(final_indices):
        X_tensor[i] = torch.from_numpy(X_all[orig_idx])
        HRV_tensor[i] = torch.from_numpy(HRV_all[orig_idx])
        Y_future_list.append(smooth_label(Y_future_all[orig_idx]))
        Y_current_list.append(smooth_label(Y_current_all[orig_idx]))
        
    Y_future_tensor = torch.tensor(Y_future_list, dtype=torch.float32)
    Y_current_tensor = torch.tensor(Y_current_list, dtype=torch.float32)
    
    # 内存释放动作
    del X_all, HRV_all, Y_future_all, Y_current_all, final_indices
    gc.collect()
    
    save_path = os.path.join(SAVE_DIR, f'{name}.pt')
    torch.save({
        'X': X_tensor, 
        'HRV': HRV_tensor, 
        'Y_future': Y_future_tensor, 
        'Y_current': Y_current_tensor
    }, save_path)
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
    
    random.seed(42)
    random.shuffle(safe_recs)
    split_idx = int(0.8 * len(safe_recs))
    train_recs = safe_recs[:split_idx]
    test_recs = safe_recs[split_idx:]
    
    build_balanced_dataset(train_recs, 'train_massive_v5', is_train=True)
    build_balanced_dataset(test_recs, 'test_pure_v5', is_train=False)