"""
Script: build_dataset_factory.py
Version: 3-class Rhythm (SVT/AT merged into AFIB) + Waveform Hazard
"""
import wfdb
import numpy as np
import torch
import os
import random
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
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
# 节律过采样倍数（3 类：Normal, PVC, 室上性心律失常）
MAX_REPEATS_RHY = {0: 1, 1: 2, 2: 4}
# 危急度过采样倍数（VF 8x，对抗严重样本稀疏）
MAX_REPEATS_CRI = {0: 1, 1: 2, 2: 4, 3: 8}


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


def cutmix_vf(x_norm, vf_pool):
    """VF CutMix: blend a random VF segment into current window.
    x_norm: [7500] ECG window. vf_pool: list of VF segments of similar length.
    Returns augmented window or original if pool is empty."""
    if not vf_pool:
        return x_norm
    vf_seg = vf_pool[np.random.randint(len(vf_pool))]
    # Resize VF segment to match window if needed
    if len(vf_seg) < len(x_norm):
        vf_seg = np.pad(vf_seg, (0, len(x_norm) - len(vf_seg)), 'edge')
    else:
        vf_seg = vf_seg[:len(x_norm)]
    # Blend: 0.3-0.7 alpha mix
    alpha = np.random.uniform(0.3, 0.7)
    x_mix = alpha * x_norm + (1 - alpha) * (vf_seg - np.mean(vf_seg)) / (np.std(vf_seg) + 1e-8)
    return (x_mix - np.mean(x_mix)) / (np.std(x_mix) + 1e-8)


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


# ---------- waveform stability hazard ----------
def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / max(norm, 1e-8)


def extract_waveform_features(seg):
    """Extract 6 waveform features for stability comparison."""
    if len(seg) < 100:
        return np.zeros(6, dtype=np.float32)
    seg = np.asarray(seg, dtype=np.float32)
    amp_std = np.std(seg)
    deriv_mean = np.mean(np.abs(np.diff(seg)))
    zcr = np.mean(np.diff(np.signbit(seg)).astype(np.float32))
    freqs = np.abs(np.fft.rfft(seg))
    n = len(freqs)
    lf_band = np.mean(freqs[:max(1, int(n * 0.02))])
    mf_band = np.mean(freqs[max(1, int(n * 0.02)):max(2, int(n * 0.06))])
    hf_band = np.mean(freqs[max(2, int(n * 0.06)):max(3, int(n * 0.18))])
    return np.array([amp_std, deriv_mean, zcr, lf_band, mf_band, hf_band], dtype=np.float32)


def compute_waveform_deviation(ecg, win_end):
    """Waveform deviation from baseline at 3 time scales → [dev_30s, dev_1m, dev_5m] in [0,1]."""
    win_start = max(0, win_end - PTS_PER_WIN)
    current_feat = extract_waveform_features(ecg[win_start:win_end])

    bl_short_start = max(0, win_end - 2 * 60 * TARGET_FS)
    bl_short_end = max(0, win_end - PTS_PER_WIN)
    bl_short = ecg[bl_short_start:bl_short_end]

    bl_med_start = max(0, win_end - 5 * 60 * TARGET_FS)
    bl_med = ecg[bl_med_start:bl_short_start]

    dev_short = float(np.clip(1.0 - _cosine_sim(current_feat, extract_waveform_features(bl_short)), 0, 1)) if len(bl_short) > TARGET_FS else 0.0
    dev_med   = float(np.clip(1.0 - _cosine_sim(current_feat, extract_waveform_features(bl_med)), 0, 1))   if len(bl_med) > TARGET_FS   else dev_short
    dev_long  = float(np.clip(max(dev_short, dev_med), 0, 1))
    return [dev_short, dev_med, dev_long]


# ---------- RR feature extraction (per-window 4 + trajectory-level 5 = 9) ----------
def extract_rr_features(ecg_window, fs=250):
    """Extract RR interval features from a 30s ECG window.
    Returns (features_4, rr_intervals_ms) tuple.
    features_4: [mean_rr, sdnn, rmssd, pNN50] — normalised per-window statistics.
    rr_intervals_ms: raw RR interval array in ms (empty if insufficient peaks)."""
    try:
        seg = np.asarray(ecg_window, dtype=np.float32)
        threshold = 0.5 * np.std(seg)
        if threshold < 0.02:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        peaks, _ = find_peaks(seg, height=threshold, distance=int(fs * 0.25))
        if len(peaks) < 3:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        rr = np.diff(peaks) / fs * 1000.0  # ms
        mean_rr = np.mean(rr) / 1000.0      # normalised
        sdnn = np.std(rr) / 300.0
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) / 100.0 if len(rr) > 1 else 0.0
        pnn50 = np.mean(np.abs(np.diff(rr)) > 50) if len(rr) > 1 else 0.0
        return (np.array([mean_rr, sdnn, rmssd, pnn50], dtype=np.float32),
                rr.astype(np.float32))
    except Exception:
        return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)


# ---------- Trajectory-level RR features (AFIB vs SVT/AT discrimination) ----------
def _sample_entropy(rr, m=2, r_factor=0.2):
    """Sample entropy of RR interval series.
    Quantifies irregularity: high for AFIB, low for SVT/AT (regular).
    Normalised by dividing by 2.0 to keep in [0, ~1.5] range."""
    rr = np.asarray(rr, dtype=np.float64)
    N = len(rr)
    if N < m + 2:
        return 0.0
    r = r_factor * np.std(rr)
    if r < 1e-10:
        return 0.0

    def _phi(template_len):
        count = 0
        templates = np.array([rr[i:i + template_len] for i in range(N - template_len)])
        n_templates = len(templates)
        if n_templates < 2:
            return 0
        for i in range(n_templates):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist < r) - 1
        return max(count, 0)

    A = _phi(m + 1)
    B = _phi(m)
    if B < 1:
        return 0.0
    ratio = A / B
    if ratio <= 0:
        return 0.0
    return float(-np.log(ratio) / 2.0)


def _poincare_features(rr):
    """Poincare plot features.
    SD1: short-term (beat-to-beat) variability, perpendicular to identity line.
    SD2: long-term variability, along identity line.
    ratio: SD1/SD2 — near 1.0 for AFIB (symmetric scatter); << 0.5 for regular rhythms.
    Returns [sd1, sd2, ratio] normalised."""
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rr_n = rr[1:]
    rr_n1 = rr[:-1]
    diff = rr_n - rr_n1
    var_diff = np.var(diff)
    var_rr = np.var(rr)
    sd1 = np.sqrt(0.5 * max(var_diff, 0))
    sd2 = np.sqrt(max(2 * var_rr - 0.5 * var_diff, 0))
    ratio = sd1 / max(sd2, 1e-10)
    return np.array([sd1 / 100.0, sd2 / 200.0, ratio], dtype=np.float32)


def compute_trajectory_rr_features(all_rr_intervals):
    """Compute 5 trajectory-level RR features from concatenated RR intervals in ms.
    Returns: [sample_entropy, poincare_sd1, poincare_sd2, poincare_ratio, cv_rr]
    Returns zeros if fewer than 10 total RR intervals."""
    if len(all_rr_intervals) < 10:
        return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr_intervals, dtype=np.float64)
    sampen = _sample_entropy(rr)
    sd1, sd2, sd_ratio = _poincare_features(rr)
    cv_rr = float(np.std(rr) / max(np.mean(rr), 1e-8))
    return np.array([sampen, sd1, sd2, sd_ratio, cv_rr], dtype=np.float32)


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
            # 仅在非 Normal 节律时填充间隙，保留搏动级 PVC 标记 (class 1)
            if current_rhythm != 0:
                rhythm_timeline[last_idx:sample_idx] = current_rhythm
            if current_crit != 0:
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
            # PVC 标注窗口：±1125 samples (4.5s @ 250Hz)，覆盖完整 30s 窗口
            region = rhythm_timeline[max(0, sample_idx - 1125):min(total_pts, sample_idx + 1125)]
            region[region == 0] = 1

        # 心房起源搏动: A(房早), a(差传房早), S(室上早), J(交界早)
        # 标记为 rhythm class 1，指示心房激惹信号
        if sym in ['A', 'a', 'S', 'J']:
            region = rhythm_timeline[max(0, sample_idx - 1125):min(total_pts, sample_idx + 1125)]
            region[region == 0] = 1

    if current_rhythm != 0:
        rhythm_timeline[last_idx:] = current_rhythm
    if current_crit != 0:
        crit_timeline[last_idx:] = current_crit
    return rhythm_timeline, crit_timeline


def extract_hierarchical_labels(rhythm_timeline, crit_timeline, win_end, db_source, ecg=None, current_pt=None):
    current_rhythm = rhythm_timeline[win_end - 1]
    current_crit = crit_timeline[win_end - 1]

    if current_crit == 0:
        win_start = max(0, win_end - TARGET_FS * 30)
        pvc_ratio = np.mean(rhythm_timeline[win_start:win_end] == 1)
        if pvc_ratio > 0.15:
            current_crit = 1

    # ---- V14: waveform stability hazard (replaces VT/VF event prediction) ----
    if ecg is not None:
        hazard_labels = compute_waveform_deviation(ecg, win_end)
    else:
        hazard_labels = [0.0, 0.0, 0.0]

    # Database masks — all databases contribute to hazard with continuous labels
    mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 1.0
    db = db_source.lower()
    if 'mitdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 1.0
    elif 'afdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 1.0  # vfdb/cudb only for crit
    elif 'vfdb' in db or 'cudb' in db:
        mask_rhythm, mask_crit, mask_hazard = 0.0, 1.0, 1.0
    elif 'svdb' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 1.0, 1.0
    elif 'ptbxl' in db:
        mask_rhythm, mask_crit, mask_hazard = 1.0, 0.0, 0.0

    # 合并 SVT/AT → AFIB (室上性心律失常)
    if current_rhythm == 3:
        current_rhythm = 2

    return current_rhythm, current_crit, hazard_labels, mask_rhythm, mask_crit, mask_hazard


def init_buffer():
    return {
        'X': [], 'X_rr': [], 'Y_rhythm': [], 'Y_criticality': [], 'Y_hazard': [],
        'M_rhythm': [], 'M_criticality': [], 'M_hazard': [],
    }


def save_shard(buffer, shard_idx, save_dir, split_name):
    shard_path = os.path.join(save_dir, f'{split_name}_shard_{shard_idx:03d}.pt')
    torch.save({
        'X': torch.tensor(np.array(buffer['X']), dtype=torch.float16),
        'X_rr': torch.tensor(np.array(buffer['X_rr']), dtype=torch.float16),
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

            # SVT/AT 节律窗口扩展: aux_note 仅标记节律起始点，扩展至后续 60s
            is_svt = rhy_timeline == 3
            if is_svt.any():
                svt_ends = np.where(is_svt[:-1] & ~is_svt[1:])[0]
                for end_pos in svt_ends:
                    end_s = min(len(rhy_timeline), end_pos + 15000)  # 60s @ 250Hz
                    region = rhy_timeline[end_pos:end_s]
                    region[region == 0] = 2  # 合并到 AFIB (室上性心律失常)

            # ---- Collect VF waveform segments for CutMix augmentation ----
            vf_pool = []
            vf_mask = cri_timeline >= 3
            if vf_mask.any():
                # Extract contiguous VF regions
                vf_starts = np.where(vf_mask[1:] & ~vf_mask[:-1])[0]
                vf_ends = np.where(~vf_mask[1:] & vf_mask[:-1])[0]
                for s, e in zip(vf_starts[:20], vf_ends[:20]):  # max 20 segments
                    if e - s > TARGET_FS:  # at least 1 second
                        vf_pool.append(ecg[s:e].copy())


            current_pt = 0
            max_pt = len(ecg) - HISTORY_SEC * TARGET_FS

            while current_pt < max_pt:
                win_end = current_pt + HISTORY_SEC * TARGET_FS

                r_lbl, c_lbl, h_lbl, m_r, m_c, m_h = extract_hierarchical_labels(
                    rhy_timeline, cri_timeline, win_end, db_source,
                    ecg=ecg, current_pt=current_pt,
                )

                # ---- split oversampling (rhy vs cri, 独立控制) ----
                n_repeats = max(
                    MAX_REPEATS_RHY.get(r_lbl, 1),
                    MAX_REPEATS_CRI.get(c_lbl, 1),
                )

                for aug_i in range(n_repeats):
                    seq_x = []
                    if aug_i == 0:
                        seq_rr_4 = []
                        all_rr_intervals = []

                    for w_i in range(N_WINDOWS):
                        w_start = current_pt + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                        w_end = w_start + PTS_PER_WIN
                        x_raw = ecg[w_start:w_end]
                        x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)

                        if aug_i > 0:
                            # Apply VF CutMix with 40% probability for VF-class samples
                            if c_lbl >= 3 and vf_pool and np.random.random() < 0.10:
                                x_norm = cutmix_vf(x_norm, vf_pool)
                            else:
                                x_norm = augment_window(x_norm)

                        seq_x.append(x_norm.astype(np.float16))

                        if aug_i == 0:
                            feat4, intervals = extract_rr_features(x_raw)
                            seq_rr_4.append(feat4.astype(np.float16))
                            if len(intervals) > 0:
                                all_rr_intervals.append(intervals)

                    # ---- Trajectory-level RR features (5-dim, broadcast to all 19 windows) ----
                    if aug_i == 0:
                        if len(all_rr_intervals) > 0:
                            concat_rr = np.concatenate(all_rr_intervals)
                            traj_feat = compute_trajectory_rr_features(concat_rr).astype(np.float16)
                        else:
                            traj_feat = np.zeros(5, dtype=np.float16)
                        seq_rr_full = np.array([
                            np.concatenate([seq_rr_4[w_i], traj_feat])
                            for w_i in range(N_WINDOWS)
                        ], dtype=np.float16)  # shape: [19, 9]

                    buffer['X'].append(
                        np.array(seq_x).reshape(N_WINDOWS, 1, PTS_PER_WIN)
                    )
                    buffer['X_rr'].append(seq_rr_full.copy())

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
    print("心电智能监护数据管道 — 三分类节律 + 波形稳定性偏离指数 + RR 轨迹特征")
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

            build_v10_dataset(recs[:split], f'v15_{db_name}_train', db_source=db_name)
            build_v10_dataset(recs[split:], f'v15_{db_name}_val', db_source=db_name)
