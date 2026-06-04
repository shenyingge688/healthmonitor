"""
Script: build_dataset_factory.py
超前 5 分钟心律失常预测数据集构建引擎

预测范式: 输入过去 10 分钟 ECG → 预测未来 5 分钟最严重节律

数据策略:
  - MIT-BIH 全量为主力 (~39 条训练记录)
  - AFDB / VFDB / CUDB / SVDB 少量补充 (各 2~3 条，弥补长程 AF 和危重 VT/VF 样本)
  - 训练/验证/演示三集完全隔离，防止数据泄露
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
HISTORY_SEC = 600          # 输入: 过去 10 分钟
PREDICT_SEC = 300          # 预测窗口: 未来 5 分钟
WINDOW_SEC = 30            # 单窗口时长
OVERLAP_STRIDE_SEC = 15    # 窗口步长
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1  # 39
PTS_PER_WIN = WINDOW_SEC * TARGET_FS  # 7500

SHARD_SIZE = 2500
FIXED_STRIDE_SEC = 10
DB_STRIDE_OVERRIDE = {"afdb": 50}  # AFDB 长程记录降采样

# 过采样倍数 (6 类)
MAX_REPEATS = {0: 1, 1: 2, 2: 4, 3: 8, 4: 4, 5: 4}

# ---- 完全隔离的数据集划分 ----
DEMO_CASES = {'mitdb': ['100', '119', '209', '201', '207']}

VAL_MITDB = ['102', '104', '107', '217']

SUPPLEMENT_RECORDS = {
    'afdb': ['04043', '04936'],
    'vfdb': ['422', '423', '426'],
    'cudb': ['cu01'],
    'svdb': ['800'],
}


# =========================================================
# 数据增强
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
# 信号预处理
# =========================================================

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)


# =========================================================
# RR 特征提取
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
# 标注解析
# =========================================================

def parse_annotations(annotation, target_fs, total_pts):
    """解析 MIT-BIH 标注为 6 分类时间线。"""
    labels = np.zeros(total_pts, dtype=int)

    current_rhythm = 0
    last_idx = 0

    # 标注优先级: VF/VFL > VT > AFIB > AFL/AT/SVT > PVC > Normal
    for idx, sym, aux in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        sample_idx = int(idx * (target_fs / 360.0))
        if sample_idx >= total_pts:
            break

        if isinstance(aux, str) and aux.startswith('('):
            # 填充前一段间隙
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

        # 心搏级 PVC
        if sym in ['V', 'E']:
            region = labels[max(0, sample_idx-1125):min(total_pts, sample_idx+1125)]
            region[region == 0] = 1

        # 心房起源搏动
        if sym in ['A', 'a', 'S', 'J']:
            region = labels[max(0, sample_idx-1125):min(total_pts, sample_idx+1125)]
            region[region == 0] = 1

    # 收尾
    if current_rhythm != 0:
        labels[last_idx:] = current_rhythm

    return labels


def get_future_label(labels, pred_start, pred_end):
    """从未来 5 分钟窗口提取最高危标签。

    优先级: VF(3) > VT(4) > AFIB(2) > AT/AFL/SVT(5) > PVC(1) > Normal(0)
    直接用最大值即可(类编号越大越危重，AT/SVT=5 在 VT=4 之后是对的)。
    但 AFIB=2 优先级应高于 AT/SVT=5 — 实际上 AFIB 节律更不稳定。

    修正: 使用自定义优先级映射
    """
    window = labels[pred_start:pred_end]
    if len(window) == 0:
        return 0

    # 6-class priority: VF(3) > VT(4) > AFIB(2) > AT/SVT(5) > PVC(1) > Normal(0)
    # 重排: class → priority weight
    priority = {3: 6, 4: 5, 2: 4, 5: 3, 1: 2, 0: 1}
    best_cls, best_pri = 0, 0
    for cls in range(6):
        count = np.sum(window == cls)
        if count > 0 and priority[cls] > best_pri:
            best_pri = priority[cls]
            best_cls = cls
    return best_cls


# =========================================================
# 数据集构建
# =========================================================

def init_buffer():
    return {'X': [], 'X_rr': [], 'Y': []}


def save_shard(buffer, shard_idx, split_name):
    path = os.path.join(SAVE_DIR, f'{split_name}_shard_{shard_idx:03d}.pt')
    torch.save({
        'X': torch.tensor(np.array(buffer['X']), dtype=torch.float16),
        'X_rr': torch.tensor(np.array(buffer['X_rr']), dtype=torch.float16),
        'Y': torch.tensor(buffer['Y'], dtype=torch.long),
    }, path)
    print(f'  [shard] {os.path.basename(path)}  ({len(buffer["X"])} samples)')


def build_dataset(record_list, name, db_source):
    buffer = init_buffer()
    shard_idx, total_extracted = 0, 0
    import time as _time

    for rec_idx, rec in enumerate(tqdm(record_list, desc=f"Building [{name}]")):
        try:
            _t0 = _time.time()

            # 检查本地标注文件是否存在，避免触发 PhysioNet 下载
            atr_path = rec + '.atr'
            if not os.path.exists(atr_path):
                print(f"  [skip] {rec}: no local .atr file, skipping")
                continue

            record = wfdb.rdrecord(rec)
            # 不指定 pn_dir 以避免 wfdb 自动联网下载
            annotation = wfdb.rdann(rec, 'atr', pn_dir=None)

            # AFDB 长程记录 (10h+) 截断至前 2 小时以加速处理
            signal_len = record.sig_len if hasattr(record, 'sig_len') else len(record.p_signal)
            MAX_SAMPLES = 2 * 3600 * 360  # 2 hours @ 360Hz
            orig_len = signal_len
            if signal_len > MAX_SAMPLES:
                signal_len = MAX_SAMPLES
                print(f"  [info] {rec}: truncating {orig_len/360/3600:.1f}h → 2h")

            raw_sig = record.p_signal[:signal_len, 0]
            ecg = signal.resample_poly(
                clean_ecg_signal(raw_sig, fs=360),
                TARGET_FS, 360)
            ecg = ecg.astype(np.float32)

            labels = parse_annotations(annotation, TARGET_FS, len(ecg))

            # 扩展 SVT/AT 节律窗口: aux_note 仅标记起始点，延长至 60s
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

                # 从未来 5 分钟提取标签
                label = get_future_label(labels, pred_start, pred_end)

                n_repeats = MAX_REPEATS.get(label, 1)

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
                            x_norm = augment_window(x_norm)

                        seq_x.append(x_norm.astype(np.float16))

                        if aug_i == 0:
                            feat4, intervals = extract_rr_features(x_raw)
                            seq_rr_4.append(feat4.astype(np.float16))
                            if len(intervals) > 0:
                                all_rr_intervals.append(intervals)

                    if aug_i == 0:
                        if len(all_rr_intervals) > 0:
                            concat_rr = np.concatenate(all_rr_intervals)
                            traj_feat = compute_traj_rr_features(concat_rr).astype(np.float16)
                        else:
                            traj_feat = np.zeros(5, dtype=np.float16)
                        seq_rr_full = np.array([
                            np.concatenate([seq_rr_4[w_i], traj_feat])
                            for w_i in range(N_WINDOWS)
                        ], dtype=np.float16)

                    buffer['X'].append(
                        np.array(seq_x).reshape(N_WINDOWS, 1, PTS_PER_WIN))
                    buffer['X_rr'].append(seq_rr_full.copy())
                    buffer['Y'].append(label)
                    total_extracted += 1

                if len(buffer['X']) >= SHARD_SIZE:
                    save_shard(buffer, shard_idx, name)
                    shard_idx += 1
                    buffer = init_buffer()

                stride = DB_STRIDE_OVERRIDE.get(db_source, FIXED_STRIDE_SEC)
                current_pt += int(stride * TARGET_FS)

            _elapsed = _time.time() - _t0
            if _elapsed > 30:
                print(f"  [slow] {rec}: {_elapsed:.1f}s, {orig_len} samples → {total_extracted} trajectories")

        except Exception as e:
            print(f"  [skip] {rec}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(buffer['X']) > 0:
        save_shard(buffer, shard_idx, name)

    print(f"  [done] {name}: {total_extracted} trajectories")
    return total_extracted


# =========================================================
# 主流程
# =========================================================

if __name__ == '__main__':
    print("=" * 60)
    print("超前 5 分钟心律失常预测 — 数据集构建")
    print(f"输入: {HISTORY_SEC}s (过去 10 分钟) → 输出: {PREDICT_SEC}s (未来 5 分钟)")
    print(f"窗口划分: {N_WINDOWS} 窗口 × {WINDOW_SEC}s (步长 {OVERLAP_STRIDE_SEC}s)")
    print("=" * 60)

    # ---- 收集 MIT-BIH 记录 ----
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

    # ---- 收集补充数据库记录 ----
    train_supplements = {}
    val_supplements = {}
    for db_name, recs in SUPPLEMENT_RECORDS.items():
        db_dir = os.path.join(DATA_DIR, db_name)
        if not os.path.isdir(db_dir):
            print(f"  [{db_name}] 目录不存在，跳过")
            continue
        available = [f.split('.')[0] for f in os.listdir(db_dir) if f.endswith('.dat')]
        found = [r for r in recs if r in available]
        if len(found) > 1:
            train_supplements[db_name] = found[:-1]
            val_supplements[db_name] = [found[-1]]
        else:
            train_supplements[db_name] = found
        print(f"  [{db_name}]: {len(found)}/{len(recs)} available")

    # ---- 构建训练集 ----
    print("\n" + "=" * 60)
    print("📦 构建训练集 (MIT-BIH 纯库)")
    print("=" * 60)

    train_recs = [os.path.join(mitdb_dir, r) for r in train_mitdb]
    # 补充库暂不加入，先用纯 MIT-BIH 快速验证
    # for db_name, recs in train_supplements.items():
    #     db_dir = os.path.join(DATA_DIR, db_name)
    #     train_recs.extend([os.path.join(db_dir, r) for r in recs])

    random.seed(42)
    random.shuffle(train_recs)
    build_dataset(train_recs, 'train', db_source='mitdb')

    # ---- 构建验证集 ----
    print("\n" + "=" * 60)
    print("📦 构建验证集 (MIT-BIH 纯库)")
    print("=" * 60)

    val_recs = [os.path.join(mitdb_dir, r) for r in val_mitdb]
    # for db_name, recs in val_supplements.items():
    #     db_dir = os.path.join(DATA_DIR, db_name)
    #     val_recs.extend([os.path.join(db_dir, r) for r in recs])

    build_dataset(val_recs, 'val', db_source='mitdb')

    print("\n✅ 数据集构建完成。")
    print(f"  - 演示集 (绝不参与训练): {DEMO_CASES}")
    print(f"  - 训练验证 gap 记录: {VAL_MITDB}")
