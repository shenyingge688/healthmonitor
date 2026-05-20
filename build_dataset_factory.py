"""
Script: build_dataset_factory.py
Version: V10.2 (Fixed Stride + Bounded Oversampling + Safe Augmentations)
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
OVERLAP_STRIDE_SEC = 15
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1
PTS_PER_WIN = WINDOW_SEC * TARGET_FS
HAZARD_WINDOWS_SEC = [30, 60, 300]

SHARD_SIZE = 2500

# Bounded oversampling -- each sample repeated up to max_repeats with safe augmentations
FIXED_STRIDE_SEC = 10
# 数据库特定步长 — afdb 样本过量 (10h records)，加大步长降采样
DB_STRIDE_OVERRIDE = {"afdb": 50}
MAX_REPEATS = {
    0: 1,   # Normal
    1: 2,   # PVC
    2: 4,   # AFIB / VT
    3: 4,   # SVT / VF
}


# ---------- safe augmentations ----------
def safe_baseline_wander(x, fs=250, max_amp=0.15):
    t = np.arange(len(x)) / fs
    freq = np.random.uniform(0.1, 0.5)
    phase = np.random.uniform(0, 2 * np.pi)
    return x + max_amp * np.sin(2 * np.pi * freq * t + phase)


def safe_gaussian_noise(x, sigma=0.02):
    return x + np.random.randn(len(x)).astype(np.float32) * sigma


def safe_gain_scaling(x, scale_range=(0.9, 1.1)):
    return x * np.random.uniform(*scale_range)


def augment_window(x_norm):
    aug_type = np.random.choice(["wander", "noise", "gain", "none"])
    x_aug = x_norm.copy()
    if aug_type == "wander":
        x_aug = safe_baseline_wander(x_aug)
    elif aug_type == "noise":
        x_aug = safe_gaussian_noise(x_aug)
    elif aug_type == "gain":
        x_aug = safe_gain_scaling(x_aug)
    return (x_aug - np.mean(x_aug)) / (np.std(x_aug) + 1e-8)


# ---------- core pipeline ----------
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
        if sample_idx >= total_pts:
            break

        if isinstance(aux, str) and aux.startswith('('):
            rhythm_timeline[last_idx:sample_idx] = current_rhythm
            crit_timeline[last_idx:sample_idx] = current_crit
            note = aux.upper()

            if 'AFIB' in note:
                current_rhythm = 2
            elif 'SVTA' in note or 'AT' in note or 'AFL' in note:
                current_rhythm = 3
            else:
                current_rhythm = 0

            if 'VFIB' in note or 'VFL' in note:
                current_crit = 3
            elif 'VT' in note:
                current_crit = 2
            else:
                current_crit = 0

            last_idx = sample_idx

        if sym in ['V', 'E']:
            # 扩大 PVC 标注窗口：±375 samples (1.5s @ 250Hz)，增加 30s 窗口捕获概率
            region = rhythm_timeline[max(0, sample_idx - 375):min(total_pts, sample_idx + 375)]
            region[region == 0] = 1

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
    if 'mitdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 0.3
    elif 'afdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 0.0
    elif 'vfdb' in db or 'cudb' in db:
        mask_rhythm, mask_crit, mask_hazard = 0.0, 1.0, 1.0
    elif 'svdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 0.3
    elif 'ptbxl' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 0.0

    return current_rhythm, current_crit, hazard_labels, mask_rhythm, mask_crit, mask_hazard


def init_buffer():
    return {
        'X': [], 'Y_rhythm': [], 'Y_criticality': [], 'Y_hazard': [],
        'M_rhythm': [], 'M_criticality': [], 'M_hazard': [],
    }


def save_shard(buffer, shard_idx, save_dir, split_name):
    shard_path = os.path.join(save_dir, f'{split_name}_shard_{shard_idx:03d}.pt')
    torch.save({
        'X': torch.tensor(np.array(buffer['X']), dtype=torch.float16),
        'Y_rhythm': torch.tensor(buffer['Y_rhythm'], dtype=torch.long),
        'Y_criticality': torch.tensor(buffer['Y_criticality'], dtype=torch.long),
        'Y_hazard': torch.tensor(buffer['Y_hazard'], dtype=torch.float32),
        'M_rhythm': torch.tensor(buffer['M_rhythm'], dtype=torch.float32),
        'M_criticality': torch.tensor(buffer['M_criticality'], dtype=torch.float32),
        'M_hazard': torch.tensor(buffer['M_hazard'], dtype=torch.float32)
    }, shard_path)
    print(f'[streaming] shard {os.path.basename(shard_path)}  ({len(buffer["X"])} samples)')


def build_v10_dataset(record_list, name, db_source):
    buffer = init_buffer()
    shard_idx = 0
    total_extracted = 0

    for rec in tqdm(record_list, desc=f"Building [{name}]"):
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            ecg = signal.resample_poly(
                clean_ecg_signal(record.p_signal[:, 0], fs=360), TARGET_FS, 360
            )

            rhy_timeline, cri_timeline = parse_ucso_annotations(
                annotation, TARGET_FS, len(ecg)
            )

            current_pt = 0
            max_pt = len(ecg) - HISTORY_SEC * TARGET_FS

            while current_pt < max_pt:
                win_end = current_pt + HISTORY_SEC * TARGET_FS

                r_lbl, c_lbl, h_lbl, m_r, m_c, m_h = extract_hierarchical_labels(
                    rhy_timeline, cri_timeline, win_end, db_source
                )

                # ---- bounded oversampling ----
                # Minority classes are repeated up to MAX_REPEATS with safe augmentations.
                # Original (aug_i=0) is always kept unfiltered.
                crit_class = max(c_lbl, 2) if c_lbl >= 2 else c_lbl
                rhy_class = max(r_lbl, 2) if r_lbl >= 2 else r_lbl
                n_repeats = max(
                    MAX_REPEATS.get(crit_class, 1),
                    MAX_REPEATS.get(rhy_class, 1),
                )
                if sum(h_lbl) > 0:
                    n_repeats = min(n_repeats + 1, 5)

                for aug_i in range(n_repeats):
                    seq_x = []
                    for w_i in range(N_WINDOWS):
                        w_start = current_pt + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                        w_end = w_start + PTS_PER_WIN
                        x_raw = ecg[w_start:w_end]
                        x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)

                        if aug_i > 0:
                            x_norm = augment_window(x_norm)

                        seq_x.append(x_norm.astype(np.float16))

                    buffer['X'].append(
                        np.array(seq_x).reshape(N_WINDOWS, 1, PTS_PER_WIN)
                    )
                    buffer['Y_rhythm'].append(r_lbl)
                    buffer['Y_criticality'].append(c_lbl)
                    buffer['Y_hazard'].append(h_lbl)
                    buffer['M_rhythm'].append(m_r)
                    buffer['M_criticality'].append(m_c)
                    buffer['M_hazard'].append(m_h)

                    total_extracted += 1

                # ---- shard checkpoint ----
                if len(buffer['X']) >= SHARD_SIZE:
                    save_shard(buffer, shard_idx, SAVE_DIR, name)
                    shard_idx += 1
                    buffer = init_buffer()

                # ---- fixed stride with per-db override (afdb downsampled 5x) ----
                stride = DB_STRIDE_OVERRIDE.get(db_source, FIXED_STRIDE_SEC)
                current_pt += int(stride * TARGET_FS)

        except Exception as e:
            print(f"[skip] {rec}: {type(e).__name__}: {e}")
            continue

    if len(buffer['X']) > 0:
        save_shard(buffer, shard_idx, SAVE_DIR, name)

    print(f"[done] {name}: {total_extracted} trajectories")


if __name__ == '__main__':
    print("V10.2 data pipeline (fixed stride + bounded oversampling)")
    target_databases = ['mitdb', 'afdb', 'vfdb', 'cudb', 'svdb']

    for db_name in target_databases:
        db_dir = os.path.join(DATA_DIR, db_name)
        if os.path.exists(db_dir):
            print(f"\nDatabase: [{db_name}]")
            recs = [
                os.path.join(db_dir, f.split('.')[0])
                for f in os.listdir(db_dir) if f.endswith('.dat')
            ]
            if not recs:
                continue

            random.seed(42)
            random.shuffle(recs)
            split = int(0.8 * len(recs))

            build_v10_dataset(recs[:split], f'v10_{db_name}_train', db_source=db_name)
            build_v10_dataset(recs[split:], f'v10_{db_name}_val', db_source=db_name)
