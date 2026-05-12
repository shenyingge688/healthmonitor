"""
核心机制: Time-Block Split | WFDB Rhythm Occupancy 解析 | Prediction Gap
"""
import wfdb
import numpy as np
import torch
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os
import random

TARGET_FS = 250
HISTORY_SEC = 300                # 观测历史缩短为 5 分钟 (抓取更多抢救数据)
WINDOW_SEC = 30                  # 单窗口 30 秒
OVERLAP_STRIDE_SEC = 15          # 滑动步长 15 秒 (50% 重叠)
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1

GAP_SEC = 30                     # 临床预测盲区 (Prediction Gap)
PTS_PER_WIN = WINDOW_SEC * TARGET_FS

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

def parse_wfdb_semantics(annotation, target_fs, total_pts):
    """解析点过程与连续状态"""
    rhythm_timeline = np.zeros(total_pts, dtype=int) 
    beat_annotations = []
    
    current_state = 0
    last_idx = 0
    
    for idx, sym, aux in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        sample_idx = int(idx * (target_fs / 360.0))
        if sample_idx >= total_pts: break
            
        # 1. 状态转移解析 (AFib/VT Occupancy)
        if isinstance(aux, str) and aux.startswith('('):
            rhythm_timeline[last_idx:sample_idx] = current_state
            rhythm_str = aux.upper()
            if 'AFIB' in rhythm_str: current_state = 2
            elif 'VT' in rhythm_str or 'VFIB' in rhythm_str or 'VFL' in rhythm_str: current_state = 4
            elif 'N' in rhythm_str: current_state = 0
            else: current_state = 0
            last_idx = sample_idx
            
        # 2. 离散节律点解析 (PVC Beats)
        if sym in ['V', 'E']:
            beat_annotations.append((sample_idx, 1)) # 1 for PVC
            
    rhythm_timeline[last_idx:] = current_state
    
    beat_idx = np.array([b[0] for b in beat_annotations]) if beat_annotations else np.array([])
    return rhythm_timeline, beat_idx

def extract_structured_labels(rhythm_timeline, beat_idx, window_start, window_end):
    """提取结构化预测目标"""
    # 1. PVC Count (Burden)
    if len(beat_idx) > 0:
        pvc_count = np.sum((beat_idx >= window_start) & (beat_idx < window_end))
    else:
        pvc_count = 0
        
    # 2. AFib Occupancy
    window_pts = window_end - window_start
    afib_pts = np.sum(rhythm_timeline[window_start:window_end] == 2)
    afib_occupancy = afib_pts / max(1, window_pts)
    
    # 3. VT/VF Hazard Onset
    vt_vf_onset = 1.0 if np.any(rhythm_timeline[window_start:window_end] == 4) else 0.0
    
    return [float(pvc_count), float(afib_occupancy), float(vt_vf_onset)]

def build_clinical_trajectory_dataset(record_list, name, timeline_mode='train', save_dir='./dataset'):
    os.makedirs(save_dir, exist_ok=True)
    X_seq_all = []
    Y_pvc_all, Y_afib_all, Y_vt_all = [], [], []
    
    for rec in tqdm(record_list, desc=f"构建临床演化集 [{name} - {timeline_mode}]"):
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            target_idx = next((i for i, n in enumerate(record.sig_name) if n.upper() in ['MLII', 'II', 'LEAD II', 'V5']), 0)
            ecg = signal.resample_poly(clean_ecg_signal(record.p_signal[:, target_idx], fs=360), TARGET_FS, 360)
            
            total_len = len(ecg)
            rhythm_timeline, beat_idx = parse_wfdb_semantics(annotation, TARGET_FS, total_len)
            
            if timeline_mode == 'train': 
                search_start, search_end = 0, total_len
            elif timeline_mode == 'val': 
                search_start, search_end = 0, total_len
            else: 
                search_start, search_end = 0, total_len
                
            max_start = search_end - (HISTORY_SEC + GAP_SEC + 300) * TARGET_FS
            if max_start <= search_start: continue
            
            # 步长 15 秒取样，极大增加稀有事件捕获率
            for start in range(search_start, max_start, 15 * TARGET_FS): 
                history_end_idx = start + HISTORY_SEC * TARGET_FS
                gap_end_idx = history_end_idx + GAP_SEC * TARGET_FS
                
                seq_data = []
                for w_i in range(N_WINDOWS):
                    win_start = start + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                    win_end = win_start + PTS_PER_WIN
                    x_raw = ecg[win_start:win_end]
                    x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                    seq_data.append(x_norm.astype(np.float32))
                
                # 针对不同视界抽取标签 (此处仅以 5m 为例演示数据保存)
                # 实际中你可以生成三组 labels
                mask_5m_end = gap_end_idx + 300 * TARGET_FS
                labels_5m = extract_structured_labels(rhythm_timeline, beat_idx, gap_end_idx, mask_5m_end)
                
                X_seq_all.append(np.array(seq_data).reshape(N_WINDOWS, 1, PTS_PER_WIN))
                Y_pvc_all.append(labels_5m[0])
                Y_afib_all.append(labels_5m[1])
                Y_vt_all.append(labels_5m[2])
                
        except Exception:
            continue
            
    if not X_seq_all: return
    
    torch.save({
        'X': torch.tensor(np.array(X_seq_all), dtype=torch.float32),
        'Y_pvc': torch.tensor(Y_pvc_all, dtype=torch.float32),
        'Y_afib': torch.tensor(Y_afib_all, dtype=torch.float32),
        'Y_vt': torch.tensor(Y_vt_all, dtype=torch.float32)
    }, os.path.join(save_dir, f'{name}_{timeline_mode}.pt'))
    print(f"✅ 生成完毕 (总数: {len(X_seq_all)})")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_recs_paths = []
    for db in ['mitdb', 'cudb', 'vfdb']:
        db_dir = os.path.join(BASE_DIR, 'data', db)
        if os.path.exists(db_dir):
            all_recs_paths.extend([os.path.join(db_dir, f.split('.')[0]) for f in os.listdir(db_dir) if f.endswith('.dat')])
            
    safe_recs = [r for r in all_recs_paths if os.path.basename(r) not in ['100', '102', '104', '107', '119', '201', '207', '209', '217']]
    random.seed(42)
    random.shuffle(safe_recs)
    
    split_idx = int(0.8 * len(safe_recs))
    build_clinical_trajectory_dataset(safe_recs[:split_idx], 'ptfn', timeline_mode='train')
    build_clinical_trajectory_dataset(safe_recs[split_idx:], 'ptfn', timeline_mode='val')