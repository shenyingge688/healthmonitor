"""
Script: build_dataset_factory.py
ECG arrhythmia early-warning dataset builder V6

Key changes from V3:
  - Soft future distribution label (class proportions) instead of hard label
  - Current state label for dual-head supervision
  - Transition weight: high for state changes (PVC->VT, VT->VF), low for steady
  - Patient-isolated train/val/demo split maintained

Data format per sample:
  X:       [N, 39, 1, 7500]  float16  (ECG windows)
  X_rr:    [N, 39, 9]         float16  (RR features)
  Y_cur:   [N]                long     (current state: 0-5)
  Y_fut:   [N, 6]             float16  (future distribution)
  T_weight:[N]                float16  (transition weight 1.0~4.0)
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
HISTORY_SEC = 600
PREDICT_SEC = 300
WINDOW_SEC = 30
OVERLAP_STRIDE_SEC = 15
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1  # 39
PTS_PER_WIN = WINDOW_SEC * TARGET_FS  # 7500

SHARD_SIZE = 2500
FIXED_STRIDE_SEC = 10
DB_STRIDE_OVERRIDE = {"afdb": 50}

DEMO_CASES = {'mitdb': ['100', '119', '209', '201', '207']}

# 验证集: 精选含 VT/VF/AFib 的记录，覆盖全部 6 类
VAL_MITDB = ['214', '221', '233', '217']

# 补充库全量使用 (train/val 按记录分割)
SUPPLEMENT_RECORDS = {
    'afdb': ['04015', '04043', '04048', '04936', '08219'],
    'vfdb': ['418', '419', '420', '421', '428', '429', '430', '607'],
    'cudb': ['cu01', 'cu02'],
    'svdb': ['800', '801'],
}

# 验证集固定选: 每个补充库的末尾 1-2 条固定给 val
VAL_SUPPLEMENTS = {
    'afdb': ['08219'],
    'vfdb': ['428', '430'],
    'cudb': ['cu02'],
    'svdb': ['801'],
}

# Transition weight: (from_majority_class, to_majority_class) -> weight
TRANSITION_WEIGHTS = {
    (1, 4): 3.0, (1, 3): 3.5,   # PVC -> VT/VF
    (4, 3): 4.0,                  # VT -> VF
    (0, 1): 1.5, (0, 2): 2.5, (0, 4): 3.0, (0, 3): 4.0,
    (4, 4): 1.5,                  # VT continuation
}


# =========================================================
# Data augmentation
# =========================================================

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


# =========================================================
# Signal preprocessing
# =========================================================

def clean_ecg_signal(data, fs):
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)


# =========================================================
# RR feature extraction
# =========================================================

def extract_rr_features(ecg_window, fs=250):
    try:
        seg = np.asarray(ecg_window, dtype=np.float32)
        threshold = 0.5 * np.std(seg)
        if threshold < 0.02:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        peaks, _ = find_peaks(seg, height=threshold, distance=int(fs * 0.25))
        if len(peaks) < 3:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        rr = np.diff(peaks) / fs * 1000.0
        mean_rr = np.mean(rr) / 1000.0
        sdnn = np.std(rr) / 300.0
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) / 100.0 if len(rr) > 1 else 0.0
        pnn50 = np.mean(np.abs(np.diff(rr)) > 50) if len(rr) > 1 else 0.0
        return (np.array([mean_rr, sdnn, rmssd, pnn50], dtype=np.float32),
                rr.astype(np.float32))
    except Exception:
        return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)


def _sample_entropy(rr, m=2, r_factor=0.2):
    rr = np.asarray(rr, dtype=np.float64)
    N = len(rr)
    if N < m + 2: return 0.0
    r = r_factor * np.std(rr)
    if r < 1e-10: return 0.0

    def _phi(tl):
        templates = np.array([rr[i:i+tl] for i in range(N-tl)])
        nt = len(templates)
        if nt < 2: return 0
        count = sum(np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1
                    for i in range(nt))
        return max(count, 0)

    A, B = _phi(m+1), _phi(m)
    return float(-np.log(A / B) / 2.0) if B >= 1 and A > 0 else 0.0


def _poincare_features(rr):
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2: return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    diff = rr[1:] - rr[:-1]
    var_diff = np.var(diff)
    var_rr = np.var(rr)
    sd1 = np.sqrt(0.5 * max(var_diff, 0))
    sd2 = np.sqrt(max(2*var_rr - 0.5*var_diff, 0))
    return np.array([sd1/100.0, sd2/200.0, sd1/max(sd2,1e-10)], dtype=np.float32)


def compute_traj_rr_features(all_rr):
    if len(all_rr) < 10: return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    s_en = _sample_entropy(rr)
    sd1, sd2, sd_ratio = _poincare_features(rr)
    cv_rr = float(np.std(rr)/max(np.mean(rr), 1e-8))
    return np.array([s_en, sd1, sd2, sd_ratio, cv_rr], dtype=np.float32)


# =========================================================
# Annotation parsing
# =========================================================

def parse_annotations(annotation, target_fs, src_fs, total_pts):
    """Build per-sample rhythm-class timeline.

    annotation.sample indices are in the SOURCE sampling rate (src_fs); the
    signal has been resampled to target_fs. Beat/rhythm markers must therefore
    be rescaled by target_fs/src_fs, NOT a hardcoded /360 (which corrupted the
    time axis of every non-360Hz database: afdb/vfdb/cudb=250Hz, svdb=128Hz).
    """
    labels = np.zeros(total_pts, dtype=int)
    current_rhythm = 0
    last_idx = 0
    scale = target_fs / float(src_fs)

    for idx, sym, aux in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        sample_idx = int(idx * scale)
        if sample_idx >= total_pts:
            break

        if isinstance(aux, str) and aux.startswith('('):
            if current_rhythm != 0:
                labels[last_idx:sample_idx] = current_rhythm

            note = aux.upper()
            if 'VFIB' in note or 'VFL' in note:
                current_rhythm = 3
            elif 'VT' in note:
                current_rhythm = 4
            elif 'AFIB' in note:
                current_rhythm = 2
            elif 'AFL' in note or 'SVT' in note or 'AT' in note:
                current_rhythm = 5
            else:
                current_rhythm = 0
            last_idx = sample_idx

        # Ventricular ectopic beats define PVC class (±4.5s island).
        if sym in ['V', 'E']:
            region = labels[max(0, sample_idx-1125):min(total_pts, sample_idx+1125)]
            region[region == 0] = 1

        # NOTE: isolated supraventricular ectopic beats (A/a/S/J) are NOT PVC
        # and are NOT sustained AT/SVT — previously mis-merged into class 1,
        # poisoning both PVC and SVT. They are left as background (class 0);
        # sustained atrial rhythms come from the aux_note markers above.

    if current_rhythm != 0:
        labels[last_idx:] = current_rhythm
    return labels


# =========================================================
# V6: Soft future distribution + current label + transition weight
# =========================================================

def get_distribution(labels, start_pt, end_pt):
    window = labels[start_pt:end_pt]
    total = len(window)
    if total == 0:
        return np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32)
    dist = np.zeros(6, dtype=np.float32)
    for cls in range(6):
        dist[cls] = np.sum(window == cls) / total
    return dist


def get_current_label(labels, history_end):
    window = labels[max(0, history_end - 30 * TARGET_FS):history_end]
    if len(window) == 0:
        return 0
    priority = {3: 6, 4: 5, 2: 4, 5: 3, 1: 2, 0: 1}
    best_cls, best_pri = 0, 0
    for cls in range(6):
        count = np.sum(window == cls)
        if count > 0 and priority[cls] > best_pri:
            best_pri = priority[cls]
            best_cls = cls
    return best_cls


def get_transition_weight(cur_maj, fut_dist):
    """Up-weight samples whose future majority class differs from the current
    state (cur_maj), or whose future contains malignant (VF/VT) mass.

    cur_maj is the SAME current-state label used for Y_cur supervision (30s
    priority-argmax), so the transition signal is aligned with what the
    current head is trained to produce.
    """
    fut_maj = int(np.argmax(fut_dist))
    if cur_maj == fut_maj:
        weight = 1.0
    else:
        weight = TRANSITION_WEIGHTS.get((cur_maj, fut_maj), 2.0)
    # Boost on ANY malignant future mass (not just argmax change), so frequent
    # but non-dominant VT/VF still receives elevated weight.
    danger = float(fut_dist[3] + fut_dist[4])
    if danger > 0.05:
        weight = max(weight, 1.5)
    if danger > 0.2:
        weight = max(weight, 2.5)
    if danger > 0.4:
        weight = max(weight, 3.5)
    return weight


# =========================================================
# Dataset construction
# =========================================================

def init_buffer():
    return {'X': [], 'X_rr': [], 'Y_cur': [], 'Y_fut': [], 'T_weight': []}


def save_shard(buffer, shard_idx, split_name):
    path = os.path.join(SAVE_DIR, f'{split_name}_shard_{shard_idx:03d}.pt')
    torch.save({
        'X': torch.tensor(np.array(buffer['X']), dtype=torch.float16),
        'X_rr': torch.tensor(np.array(buffer['X_rr']), dtype=torch.float16),
        'Y_cur': torch.tensor(buffer['Y_cur'], dtype=torch.long),
        'Y_fut': torch.tensor(np.array(buffer['Y_fut']), dtype=torch.float16),
        'T_weight': torch.tensor(buffer['T_weight'], dtype=torch.float16),
    }, path)
    print(f'  [shard] {os.path.basename(path)}  ({len(buffer["X"])} samples)')


def build_dataset(record_list, name, db_source):
    buffer = init_buffer()
    shard_idx, total_extracted = 0, 0
    tw_all, ycur_all = [], []   # coverage instrumentation
    import time as _time

    for rec_idx, rec in enumerate(tqdm(record_list, desc=f"Building [{name}]")):
        try:
            _t0 = _time.time()

            atr_path = rec + '.atr'
            if not os.path.exists(atr_path):
                print(f"  [skip] {rec}: no .atr file")
                continue

            record_obj = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr', pn_dir=None)

            signal_len = record_obj.sig_len if hasattr(record_obj, 'sig_len') else len(record_obj.p_signal)
            src_fs = record_obj.fs if hasattr(record_obj, 'fs') else 360
            MAX_SAMPLES = 2 * 3600 * int(src_fs)
            orig_len = signal_len
            if signal_len > MAX_SAMPLES:
                signal_len = MAX_SAMPLES
                print(f"  [info] {os.path.basename(rec)}: trunc {orig_len/src_fs/3600:.1f}h -> 2h")

            raw_sig = record_obj.p_signal[:signal_len, 0]
            if src_fs != TARGET_FS:
                ecg = signal.resample_poly(
                    clean_ecg_signal(raw_sig, fs=src_fs), TARGET_FS, int(src_fs))
            else:
                ecg = clean_ecg_signal(raw_sig, fs=src_fs)
            ecg = ecg.astype(np.float32)

            labels = parse_annotations(annotation, TARGET_FS, src_fs, len(ecg))

            svt_mask = labels == 5
            if svt_mask.any():
                svt_ends = np.where(svt_mask[:-1] & ~svt_mask[1:])[0]
                for end_pos in svt_ends:
                    end_s = min(len(labels), end_pos + 15000)
                    region = labels[end_pos:end_s]
                    region[region == 0] = 5

            current_pt = 0
            max_pt = len(ecg) - (HISTORY_SEC + PREDICT_SEC) * TARGET_FS
            if max_pt <= 0:
                continue

            while current_pt < max_pt:
                history_end = current_pt + HISTORY_SEC * TARGET_FS
                pred_start = history_end
                pred_end = pred_start + PREDICT_SEC * TARGET_FS

                cur_label = get_current_label(labels, history_end)
                fut_dist = get_distribution(labels, pred_start, pred_end)
                t_weight = get_transition_weight(cur_label, fut_dist)

                seq_x, seq_rr_4, all_rr_intervals = [], [], []

                for w_i in range(N_WINDOWS):
                    w_start = current_pt + w_i * OVERLAP_STRIDE_SEC * TARGET_FS
                    w_end = w_start + PTS_PER_WIN
                    x_raw = ecg[w_start:w_end]
                    x_norm = (x_raw - np.mean(x_raw)) / (np.std(x_raw) + 1e-8)
                    seq_x.append(x_norm.astype(np.float16))

                    feat4, intervals = extract_rr_features(x_raw)
                    seq_rr_4.append(feat4.astype(np.float16))
                    if len(intervals) > 0:
                        all_rr_intervals.append(intervals)

                if len(all_rr_intervals) > 0:
                    concat_rr = np.concatenate(all_rr_intervals)
                    traj_feat = compute_traj_rr_features(concat_rr).astype(np.float16)
                else:
                    traj_feat = np.zeros(5, dtype=np.float16)
                seq_rr_full = np.array([
                    np.concatenate([seq_rr_4[w_i], traj_feat])
                    for w_i in range(N_WINDOWS)
                ], dtype=np.float16)

                buffer['X'].append(np.array(seq_x).reshape(N_WINDOWS, 1, PTS_PER_WIN))
                buffer['X_rr'].append(seq_rr_full.copy())
                buffer['Y_cur'].append(cur_label)
                buffer['Y_fut'].append(fut_dist)
                buffer['T_weight'].append(t_weight)
                tw_all.append(t_weight)
                ycur_all.append(cur_label)
                total_extracted += 1

                if len(buffer['X']) >= SHARD_SIZE:
                    save_shard(buffer, shard_idx, name)
                    shard_idx += 1
                    buffer = init_buffer()

                stride = DB_STRIDE_OVERRIDE.get(db_source, FIXED_STRIDE_SEC)
                current_pt += int(stride * TARGET_FS)

            _elapsed = _time.time() - _t0
            if _elapsed > 30:
                print(f"  [slow] {rec}: {_elapsed:.1f}s")

        except Exception as e:
            print(f"  [skip] {rec}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(buffer['X']) > 0:
        save_shard(buffer, shard_idx, name)

    # ---- coverage instrumentation (data-5 / class balance) ----
    if tw_all:
        tw = np.array(tw_all)
        yc = np.array(ycur_all)
        names = ['Normal', 'PVC', 'AFib', 'VF', 'VT', 'AT/SVT']
        frac_gt1 = float((tw > 1.0).mean())
        print(f"  [coverage] {name}: T_weight>1 = {frac_gt1*100:.1f}% "
              f"(mean={tw.mean():.2f} max={tw.max():.1f})")
        cur_counts = {names[i]: int((yc == i).sum()) for i in range(6)}
        print(f"  [coverage] {name} Y_cur dist: {cur_counts}")

    print(f"  [done] {name}: {total_extracted} trajectories")
    return total_extracted


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':
    print("=" * 60)
    print("V6 Soft-Label Dataset Builder")
    print(f"Input: {HISTORY_SEC}s -> Output: {PREDICT_SEC}s")
    print(f"Windows: {N_WINDOWS} x {WINDOW_SEC}s (stride {OVERLAP_STRIDE_SEC}s)")
    print("=" * 60)

    mitdb_dir = os.path.join(DATA_DIR, 'mitdb')
    all_mitdb = sorted([
        f.split('.')[0] for f in os.listdir(mitdb_dir)
        if f.endswith('.dat')
    ]) if os.path.isdir(mitdb_dir) else []

    demo_mitdb = DEMO_CASES['mitdb']
    train_mitdb = [r for r in all_mitdb if r not in demo_mitdb and r not in VAL_MITDB]
    val_mitdb = [r for r in all_mitdb if r in VAL_MITDB]

    print(f"\nMIT-BIH: {len(all_mitdb)} records total")
    print(f"  Train: {len(train_mitdb)}  |  Val: {len(val_mitdb)}  |  Demo: {len(demo_mitdb)}")

    train_supplements = {}
    val_supplements = {}
    for db_name, recs in SUPPLEMENT_RECORDS.items():
        db_dir = os.path.join(DATA_DIR, db_name)
        if not os.path.isdir(db_dir):
            print(f"  [{db_name}] dir not found, skip")
            continue
        available = [f.split('.')[0] for f in os.listdir(db_dir)
                     if f.endswith('.dat') or f.endswith('.qrs')]
        available = list(set(available))
        found = [r for r in recs if r in available]
        val_fixed = VAL_SUPPLEMENTS.get(db_name, [])
        train_recs_db = [r for r in found if r not in val_fixed]
        val_recs_db = [r for r in found if r in val_fixed]
        if train_recs_db:
            train_supplements[db_name] = train_recs_db
        if val_recs_db:
            val_supplements[db_name] = val_recs_db
        print(f"  [{db_name}]: {len(found)}/{len(recs)} available (train={len(train_recs_db)} val={len(val_recs_db)})")

    print("\n" + "=" * 60)
    print("[train] Training set")
    print("=" * 60)

    train_recs = [os.path.join(mitdb_dir, r) for r in train_mitdb]
    for db_name, recs in train_supplements.items():
        db_dir = os.path.join(DATA_DIR, db_name)
        train_recs.extend([os.path.join(db_dir, r) for r in recs])
        print(f"  + [{db_name}]: {len(recs)} records")

    random.seed(42)
    random.shuffle(train_recs)
    build_dataset(train_recs, 'train', db_source='mitdb')

    print("\n" + "=" * 60)
    print("[val] Validation set (patient-isolated)")
    print("=" * 60)

    val_recs = [os.path.join(mitdb_dir, r) for r in val_mitdb]
    for db_name, recs in val_supplements.items():
        db_dir = os.path.join(DATA_DIR, db_name)
        val_recs.extend([os.path.join(db_dir, r) for r in recs])
        print(f"  + [{db_name}]: {len(recs)} records")

    build_dataset(val_recs, 'val', db_source='mitdb')

    print("\nDone.")
    print(f"  Demo (never used): {DEMO_CASES}")
    print(f"  Val patients: {VAL_MITDB}")
