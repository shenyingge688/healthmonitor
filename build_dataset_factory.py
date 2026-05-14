"""
Script: build_dataset_factory.py
Version: V10.1 (Sharding + Float16 + Adaptive Stride + UCSO)
"""
import wfdb
import numpy as np
import torch
import os
import random
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_FS = 250
HISTORY_SEC = 300
WINDOW_SEC = 30
OVERLAP_STRIDE_SEC = 15  # 窗口内部特征提取步长
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1
PTS_PER_WIN = WINDOW_SEC * TARGET_FS
HAZARD_WINDOWS_SEC = [30, 60, 300] 

SHARD_SIZE = 2500  # 🚀 内存防爆：每个 Shard 最多 2500 条轨迹

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

def parse_ucso_annotations(annotation, target_fs, total_pts):
    rhythm_timeline = np.zeros(total_pts, dtype=int)  
    crit_timeline = np.zeros(total_pts, dtype=int)    
    
    current_rhythm = 0
    current_crit = 0
    last_idx = 0
    
    for idx, sym, aux in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        sample_idx = int(idx * (target_fs / 360.0))
        if sample_idx >= total_pts: break
            
        if isinstance(aux, str) and aux.startswith('('):
            rhythm_timeline[last_idx:sample_idx] = current_rhythm
            crit_timeline[last_idx:sample_idx] = current_crit
            note = aux.upper()
            
            if 'AFIB' in note: current_rhythm = 2
            elif 'SVTA' in note or 'AT' in note or 'AFL' in note: current_rhythm = 3
            else: current_rhythm = 0
            
            if 'VFIB' in note or 'VFL' in note: current_crit = 3
            elif 'VT' in note: current_crit = 2
            else: current_crit = 0
            
            last_idx = sample_idx
            
        if sym in ['V', 'E']:
            rhythm_timeline[max(0, sample_idx-125):min(total_pts, sample_idx+125)] = 1

    rhythm_timeline[last_idx:] = current_rhythm
    crit_timeline[last_idx:] = current_crit
    return rhythm_timeline, crit_timeline

def extract_hierarchical_labels(rhythm_timeline, crit_timeline, win_end, db_source):
    current_rhythm = rhythm_timeline[win_end - 1]
    current_crit = crit_timeline[win_end - 1]
    
    if current_crit == 0:
        win_start = max(0, win_end - TARGET_FS * 10)
        pvc_ratio = np.mean(rhythm_timeline[win_start:win_end] == 1)
        if pvc_ratio > 0.3:
            current_crit = 1
            
    hazard_labels = []
    for horizon in HAZARD_WINDOWS_SEC:
        future_end = min(len(crit_timeline), win_end + horizon * TARGET_FS)
        future_crit_region = crit_timeline[win_end:future_end]
        is_collapse = 1.0 if np.any(future_crit_region >= 2) else 0.0
        hazard_labels.append(is_collapse)
        
    mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 1.0
    db = db_source.lower()
    if 'mitdb' in db: mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 0.0
    elif 'afdb' in db: mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 0.0
    elif 'vfdb' in db or 'cudb' in db: mask_rhythm, mask_crit, mask_hazard = 0.0, 1.0, 1.0
    elif 'ptbxl' in db: mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 0.0
        
    return current_rhythm, current_crit, hazard_labels, mask_rhythm, mask_crit, mask_hazard

def init_buffer():
    return {'X': [], 'Y_rhythm': [], 'Y_criticality': [], 'Y_hazard': [], 'M_rhythm': [], 'M_criticality': [], 'M_hazard': []}

def save_shard(buffer, shard_idx, save_dir, split_name):
    """🚀 流式落盘：释放内存，转换为 Float16"""
    shard_path = os.path.join(save_dir, f'{split_name}_shard_{shard_idx:03d}.pt')
    torch.save({
        'X': torch.tensor(np.array(buffer['X']), dtype=torch.float16), # 减半内存
        'Y_rhythm': torch.tensor(buffer['Y_rhythm'], dtype=torch.long),
        'Y_criticality': torch.tensor(buffer['Y_criticality'], dtype=torch.long),
        'Y_hazard': torch.tensor(buffer['Y_hazard'], dtype=torch.float32),
        'M_rhythm': torch.tensor(buffer['M_rhythm'], dtype=torch.float32),
        'M_criticality': torch.tensor(buffer['M_criticality'], dtype=torch.float32),
        'M_hazard': torch.tensor(buffer['M_hazard'], dtype=torch.float32)
    }, shard_path)
    print(f'💾 [流式落盘] 成功写入 Shard: {os.path.basename(shard_path)} (容量: {len(buffer["X"])} 条)')

def build_v10_dataset(record_list, name, db_source):
    buffer = init_buffer()
    shard_idx = 0
    total_extracted = 0
    
    for rec in tqdm(record_list, desc=f"构建演化集 [{name}]"):
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            ecg = signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), TARGET_FS, 360)
            
            rhy_timeline, cri_timeline = parse_ucso_annotations(annotation, TARGET_FS, len(ecg))
            
            current_pt = 0
            max_pt = len(ecg) - HISTORY_SEC * TARGET_FS
            
            while current_pt < max_pt:
                win_end = current_pt + HISTORY_SEC * TARGET_FS
                
                # 1. 获取当前时刻标签
                r_lbl, c_lbl, h_lbl, m_r, m_c, m_h = extract_hierarchical_labels(rhy_timeline, cri_timeline, win_end, db_source)
                
                # 🚀 2. Adaptive Temporal Sampling (自适应步长)
                if c_lbl >= 2: stride_sec = 1                 # VT/VF: 1s 逐帧高密扫描
                elif sum(h_lbl) > 0: stride_sec = 2           # Pre-collapse: 2s 捕捉突变
                elif c_lbl == 1 or r_lbl == 1: stride_sec = 5 # 异位激惹: 5s
                else: stride_sec = 15                         # 稳态: 15s 大步长跨越，防止冗余
                
                # 3. 轨迹切片组装
                seq_x = []
                for w_i in range(N_WINDOWS):
                    w_start = current_pt + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                    w_end = w_start + PTS_PER_WIN
                    x_raw = ecg[w_start:w_end]
                    x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                    seq_x.append(x_norm.astype(np.float16)) # Early downcast
                
                buffer['X'].append(np.array(seq_x).reshape(N_WINDOWS, 1, PTS_PER_WIN))
                buffer['Y_rhythm'].append(r_lbl)
                buffer['Y_criticality'].append(c_lbl)
                buffer['Y_hazard'].append(h_lbl)
                buffer['M_rhythm'].append(m_r)
                buffer['M_criticality'].append(m_c)
                buffer['M_hazard'].append(m_h)
                
                total_extracted += 1
                
                # 🚀 4. Shard 落盘检测
                if len(buffer['X']) >= SHARD_SIZE:
                    save_shard(buffer, shard_idx, SAVE_DIR, name)
                    shard_idx += 1
                    buffer = init_buffer()
                    
                current_pt += int(stride_sec * TARGET_FS)
                
        except Exception: 
            continue
            
    # 收尾最后一个不完整的 shard
    if len(buffer['X']) > 0:
        save_shard(buffer, shard_idx, SAVE_DIR, name)
        
    print(f"✅ {name} 构建完毕，共计提取 {total_extracted} 条轨迹。")

if __name__ == '__main__':
    print("🚀 启动 V10.1 ICU-Scale 数据流管线...")
    target_databases = ['mitdb', 'afdb', 'vfdb', 'cudb']
    
    for db_name in target_databases:
        db_dir = os.path.join(DATA_DIR, db_name)
        if os.path.exists(db_dir):
            print(f"\n🔍 扫描到本地数据库: [{db_name}]")
            recs = [os.path.join(db_dir, f.split('.')[0]) for f in os.listdir(db_dir) if f.endswith('.dat')]
            if not recs: continue
                
            random.seed(42)
            random.shuffle(recs)
            split = int(0.8 * len(recs))
            
            build_v10_dataset(recs[:split], f'v10_{db_name}_train', db_source=db_name)
            build_v10_dataset(recs[split:], f'v10_{db_name}_val', db_source=db_name)