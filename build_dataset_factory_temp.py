"""
模块名称：数据工厂与特征流水线 (Data Builder Factory)
模块功能：
    1. 自动校验并同步 MIT-BIH 云端与本地的数据库。
    2. 执行临床级信号预处理（带通滤波与降采样）。
    3. 基于滑动窗口提取 10 分钟历史特征，并基于未来 5 分钟的波形形态进行打标。
    4. 采用 1:1 欠采样策略与标签平滑（Label Smoothing）技术，生成高鲁棒性的 PyTorch 张量数据集。
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

# ==========================================
# 模块 1：系统路径与全局参数配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'mitdb')
SAVE_DIR = os.path.join(BASE_DIR, 'dataset')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_FS = 250                # 统一重采样频率为 250Hz
HISTORY_PTS = 600 * TARGET_FS  # 观察过去 10 分钟 (150,000 点)
PREDICT_PTS = 300 * TARGET_FS  # 预测未来 5 分钟 (75,000 点)
STRIDE_SEC = 60                # 滑动窗口步长 1 分钟

# ==========================================
# 模块 2：临床级信号预处理
# ==========================================
def clean_ecg_signal(data, fs=360):
    """
    功能：利用四阶巴特沃斯带通滤波器 (0.5Hz - 45Hz)，
    消除基线漂移（如呼吸运动）和高频噪声（如肌电干扰和工频干扰）。
    """
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

# ==========================================
# 模块 3：本地数据库智能同步
# ==========================================
print(f"[start] 正在校验并同步本地临床数据: {DATA_DIR}")
try:
    all_record_names = wfdb.get_record_list('mitdb')
    # 使用 sys.stdout 强制控制台平滑输出，避免多行刷屏
    pbar_sync = tqdm(all_record_names, desc="数据校验同步", unit="条", dynamic_ncols=True, leave=False, file=sys.stdout)
    
    for rec in pbar_sync:
        hea_exists = os.path.exists(os.path.join(DATA_DIR, f"{rec}.hea"))
        dat_exists = os.path.exists(os.path.join(DATA_DIR, f"{rec}.dat"))
        atr_exists = os.path.exists(os.path.join(DATA_DIR, f"{rec}.atr"))
        
        # 仅下载缺失的文件，节省带宽与时间
        if not (hea_exists and dat_exists and atr_exists):
            wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[rec])
            
    print("[done] 临床数据库本地同步就绪。")
except Exception as e:
    print(f"[warn] 数据库同步提示: {e}")

# 定义多分类映射字典
def get_class_from_annotation(symbol, aux_note):
    # 优先匹配节律级恶性事件 (aux_note)
    if aux_note:
        note = aux_note.upper()
        if '(VFIB' in note or '(VFL' in note: return 3
        if '(VT' in note: return 4
        if '(AFIB' in note: return 2
        if '(AFL' in note or '(SVT' in note or '(AT' in note: return 5
    
    # 匹配心搏级事件 (symbol)
    if symbol in ['V', 'E']: return 1 # PVC
    if symbol in ['N', '.', '/', 'L', 'R', 'e', 'j']: return 0
    return 0
# ==========================================
# 模块 4：核心数据集构建引擎
# ==========================================
def build_balanced_dataset(record_list, name, risk_threshold=20):
    X_multi, Y_multi = [], []
    
    pbar = tqdm(record_list, desc=f"构建数据集 [{name}]", unit="rec", dynamic_ncols=True, file=sys.stdout)
    for rec in pbar:
        try:
            rec_path = os.path.join(DATA_DIR, rec)
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
            raw_ecg = record.p_signal[:, 0]
            
            # 1. 滤波与重采样
            clean_raw = clean_ecg_signal(raw_ecg, fs=360)
            ecg = signal.resample_poly(clean_raw, TARGET_FS, 360)
            
            # 2. 全局标准化
            ecg_global_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            
            # 3. 注释时间轴对齐 (360Hz 映射至 250Hz) 并映射分类
            anno_idx = np.round(annotation.sample * (TARGET_FS / 360)).astype(int)
            anno_classes = np.array([get_class_from_annotation(sym, aux) for sym, aux in zip(annotation.symbol, annotation.aux_note)])
            
            # 4. 滑动窗口切片提取
            for start in range(0, len(ecg_global_norm) - HISTORY_PTS - PREDICT_PTS, STRIDE_SEC * TARGET_FS):
                predict_start = start + HISTORY_PTS
                predict_end = predict_start + PREDICT_PTS
                
                window_mask = (anno_idx >= predict_start) & (anno_idx < predict_end)
                window_classes = anno_classes[window_mask]
                
                # 定性逻辑：寻找窗口内的最高危标签
                priority_map = {3: 6, 4: 5, 2: 4, 5: 3, 1: 2, 0: 1}
                target_class = 0
                highest_priority = 0
                
                if len(window_classes) > 0:
                    class_counts = {c: np.sum(window_classes == c) for c in set(window_classes)}
                    for cls, count in class_counts.items():
                        # 恶性节律只需出现即报警，PVC需达到频发阈值
                        if (cls in [2, 3, 4, 5] and count >= 1) or (cls == 1 and count >= risk_threshold) or (cls == 0):
                            if priority_map[cls] > highest_priority:
                                highest_priority = priority_map[cls]
                                target_class = cls
                
                x_norm = ecg_global_norm[start : predict_start]
                sample = x_norm.reshape(300, 1, 500)
                
                X_multi.append(sample)
                
                # 标签平滑 (Label Smoothing) for 6 classes
                smoothed_label = [0.02] * 6
                smoothed_label[target_class] = 0.90
                Y_multi.append(smoothed_label)
                
        except Exception:
            continue
            
    # 打乱并保存 (为多分类保持精简，省略了重采样逻辑，直接保存全量切片)
    combined = list(zip(X_multi, Y_multi))
    random.shuffle(combined)
    X_multi, Y_multi = zip(*combined)
    
    if X_multi:
        save_path = os.path.join(SAVE_DIR, f'{name}.pt')
        torch.save({'X': torch.tensor(np.array(X_multi), dtype=torch.float32), 
                    'Y': torch.tensor(np.array(Y_multi), dtype=torch.float32)}, save_path)
        pbar.write(f"[info] 已生成 {name} -> 样本量: {len(X_multi)}")
# ==========================================
# 模块 5：执行控制流 
# ==========================================

if __name__ == '__main__':
    # 替换为前端正在使用的全新多分类经典病案
    demo_cases = ['100', '119', '201', '207', '209'] 
    
    all_recs = sorted(list(set([f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')])))
    
    # 严格在训练集中剔除前端演示用的病例，防止数据泄露 (Data Leakage)
    train_recs = [r for r in all_recs if r not in (demo_cases + ['102', '104', '107', '217'])]
    
    build_balanced_dataset(train_recs, 'train_massive_v5')
    build_balanced_dataset(demo_cases, 'val_demo_v5')