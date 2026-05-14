"""
Script: build_dataset_factory.py
Version: V8.3 Final (Rolling Horizon Protocol - Debugged)
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
HISTORY_SEC = 300                # 观测历史 5 分钟
WINDOW_SEC = 30                  # 单窗口 30 秒
OVERLAP_STRIDE_SEC = 15          # 滑动步长 15 秒 (50% 重叠)
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1
PTS_PER_WIN = WINDOW_SEC * TARGET_FS

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

def parse_wfdb_semantics(annotation, target_fs, total_pts):
    rhythm_timeline = np.zeros(total_pts, dtype=int) 
    beat_annotations = []
    current_state = 0
    last_idx = 0
    
    for idx, sym, aux in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        sample_idx = int(idx * (target_fs / 360.0))
        if sample_idx >= total_pts: break
            
        if isinstance(aux, str) and aux.startswith('('):
            rhythm_timeline[last_idx:sample_idx] = current_state
            rhythm_str = aux.upper()
            if 'AFIB' in rhythm_str: current_state = 2
            elif 'VT' in rhythm_str or 'VFIB' in rhythm_str or 'VFL' in rhythm_str: current_state = 4
            else: current_state = 0
            last_idx = sample_idx
            
        if sym in ['V', 'E']:
            beat_annotations.append((sample_idx, 1))
            
    rhythm_timeline[last_idx:] = current_state
    beat_idx = np.array([b[0] for b in beat_annotations]) if beat_annotations else np.array([])
    return rhythm_timeline, beat_idx

def extract_labels(rhythm_timeline, beat_idx, win_start, win_end, future_start, future_end, target_fs):
    # 1. 提取【当前窗口】的辅助形态学特征 (用于稳定 Latent Space)
    pvc_count = np.sum((beat_idx >= win_start) & (beat_idx < win_end)) if len(beat_idx) > 0 else 0
    pvc_prob = 1.0 if pvc_count > 0 else 0.0  # 转为二值概率
    
    win_pts = win_end - win_start
    afib_pts = np.sum(rhythm_timeline[win_start:win_end] == 2)
    afib_occupancy = float(afib_pts / max(1, win_pts))
    
    # 2. 提取【未来窗口】的 VT 生存标签 (消除 Gap 盲区，直接从 win_end 监控)
    future_rhythm = rhythm_timeline[future_start:future_end]
    vt_indices = np.where(future_rhythm == 4)[0]
    
    if len(vt_indices) == 0:
        vt_survival = [0.0, 0.0, 0.0]
    else:
        time_to_vt = vt_indices[0] / target_fs
        if time_to_vt <= 30.0:
            vt_survival = [1.0, -1.0, -1.0] 
        elif time_to_vt <= 120.0:
            vt_survival = [0.0, 1.0, -1.0]  
        else:
            vt_survival = [0.0, 0.0, 1.0]   
    
    return pvc_prob, afib_occupancy, vt_survival

def build_clinical_trajectory_dataset(record_list, name, timeline_mode='train', save_dir='./dataset'):
    os.makedirs(save_dir, exist_ok=True)
    X_seq_all, Y_pvc_all, Y_afib_all, Y_vt_all = [], [], [], []
    
    for rec in tqdm(record_list, desc=f"构建演化集 [{name} - {timeline_mode}]"):
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            target_idx = next((i for i, n in enumerate(record.sig_name) if n.upper() in ['MLII', 'II', 'LEAD II', 'V5']), 0)
            ecg = signal.resample_poly(clean_ecg_signal(record.p_signal[:, target_idx], fs=360), TARGET_FS, 360)
            
            total_len = len(ecg)
            rhythm_timeline, beat_idx = parse_wfdb_semantics(annotation, TARGET_FS, total_len)
            
            # 严格截断：确保最后一个窗口有完整的未来 5 分钟
            max_start = total_len - (HISTORY_SEC + 300) * TARGET_FS
            if max_start <= 0: continue
            
            for start in range(0, max_start, 15 * TARGET_FS): 
                seq_data, seq_pvc, seq_afib, seq_vt = [], [], [], []
                
                for w_i in range(N_WINDOWS):
                    win_start = start + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                    win_end = win_start + PTS_PER_WIN
                    
                    x_raw = ecg[win_start:win_end]
                    x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                    seq_data.append(x_norm.astype(np.float32))
                    
                    # 锚定相对未来 (紧接窗口之后 5 分钟)
                    future_start = win_end
                    future_end = future_start + 300 * TARGET_FS
                    
                    pvc, afib, vt = extract_labels(rhythm_timeline, beat_idx, win_start, win_end, future_start, future_end, TARGET_FS)
                    seq_pvc.append(pvc)
                    seq_afib.append(afib)
                    seq_vt.append(vt)
                
                X_seq_all.append(np.array(seq_data).reshape(N_WINDOWS, 1, PTS_PER_WIN))
                Y_pvc_all.append(seq_pvc)
                Y_afib_all.append(seq_afib)
                Y_vt_all.append(seq_vt)
                
        except Exception as e:
            continue
            
    if not X_seq_all: return
    
    torch.save({
        'X': torch.tensor(np.array(X_seq_all), dtype=torch.float32),
        'Y_pvc': torch.tensor(np.array(Y_pvc_all), dtype=torch.float32),
        'Y_afib': torch.tensor(np.array(Y_afib_all), dtype=torch.float32),
        'Y_vt': torch.tensor(np.array(Y_vt_all), dtype=torch.float32) 
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