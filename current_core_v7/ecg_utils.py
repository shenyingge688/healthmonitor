"""
ecg_utils.py — 共享的 ECG 信号处理与 RR 特征提取工具函数

本模块整合了项目中多个文件重复定义的:
  - 信号预处理 (带通滤波)
  - R 峰检测与 RR 间期特征
  - 样本熵 (Sample Entropy)
  - 庞加莱图特征 (Poincaré Plot)
  - 轨迹级特征汇总
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# =========================================================
# 信号预处理
# =========================================================

def clean_ecg_signal(data, fs=360):
    """4 阶巴特沃斯带通滤波 (0.5–45 Hz)，去除基线漂移与高频噪声。

    Args:
        data: 一维 ECG 信号数组
        fs: 原始采样率 (Hz)

    Returns:
        滤波后的信号
    """
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype="band")
    return filtfilt(b, a, data)


# =========================================================
# RR 间期特征提取 (窗口级)
# =========================================================

def extract_rr_features(ecg_window, fs=250):
    """从单窗口 ECG 信号提取 4 维 RR 间期特征。

    返回:
        rr_feat: [mean_rr, sdnn, rmssd, pnn50] — shape (4,)
        rr_intervals: 原始 RR 间期数组 (ms)
    """
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


# =========================================================
# 样本熵 (Sample Entropy)
# =========================================================

def sample_entropy(rr, m=2, r_factor=0.2):
    """计算 RR 间期序列的样本熵 — 衡量心律不规则的复杂度。

    Args:
        rr: RR 间期序列 (ms)
        m: 模板长度
        r_factor: 容差因子 (乘以序列标准差)

    Returns:
        样本熵值 (float); 序列过短或方差趋零返回 0.0
    """
    rr = np.asarray(rr, dtype=np.float64)
    N = len(rr)
    if N < m + 2:
        return 0.0
    r = r_factor * np.std(rr)
    if r < 1e-10:
        return 0.0

    def _phi(tl):
        templates = np.array([rr[i : i + tl] for i in range(N - tl)])
        nt = len(templates)
        if nt < 2:
            return 0
        count = sum(
            np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1
            for i in range(nt)
        )
        return max(count, 0)

    A, B = _phi(m + 1), _phi(m)
    return float(-np.log(A / B) / 2.0) if B >= 1 and A > 0 else 0.0


# =========================================================
# 庞加莱图特征 (Poincaré Plot)
# =========================================================

def poincare_features(rr):
    """从 RR 间期序列提取庞加莱图特征 (SD1, SD2, SD1/SD2 ratio)。

    Args:
        rr: RR 间期序列 (ms)

    Returns:
        [sd1/100, sd2/200, sd1/max(sd2, 1e-10)] — shape (3,)
    """
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    diff = rr[1:] - rr[:-1]
    var_diff = np.var(diff)
    var_rr = np.var(rr)
    sd1 = np.sqrt(0.5 * max(var_diff, 0))
    sd2 = np.sqrt(max(2 * var_rr - 0.5 * var_diff, 0))
    return np.array(
        [sd1 / 100.0, sd2 / 200.0, sd1 / max(sd2, 1e-10)], dtype=np.float32
    )


# =========================================================
# 轨迹级 RR 特征汇总
# =========================================================

def compute_traj_rr_features(all_rr):
    """从完整 10 分钟 RR 序列提取 5 维轨迹级特征。

    包含: 样本熵 + SD1 + SD2 + SD1/SD2 + CV(RR)

    Args:
        all_rr: 完整 RR 间期序列 (ms)

    Returns:
        5 维特征向量 (float32)
    """
    if len(all_rr) < 10:
        return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    s_en = sample_entropy(rr)
    p = poincare_features(rr)
    cv_rr = float(np.std(rr) / max(np.mean(rr), 1e-8))
    return np.array([s_en, p[0], p[1], p[2], cv_rr], dtype=np.float32)


# =========================================================
# 窗口序列构建 (供推理服务使用)
# =========================================================

def build_window_sequence(ecg_array, n_windows=39, history_pts=150000,
                          pts_per_win=7500, stride_pts=3750):
    """将 ECG 缓冲区切分为 N_WINDOWS 个重叠窗口 + RR 特征。

    用于 main.py 推理管线。

    Args:
        ecg_array: 完整 ECG 缓冲区
        n_windows: 窗口数量
        history_pts: 历史总点数
        pts_per_win: 每窗口点数
        stride_pts: 窗口步长

    Returns:
        wins: [N_WINDOWS, 1, PTS_PER_WIN] 窗口序列
        rr_f: [N_WINDOWS, 9] RR 特征 (4 局部 + 5 全局轨迹)
    """
    rel = ecg_array[-history_pts:]
    wins, f4s = [], []

    for wi in range(n_windows):
        s, e = wi * stride_pts, wi * stride_pts + pts_per_win
        x = rel[s:e]
        x_n = (x - np.mean(x)) / (np.std(x) + 1e-8)
        wins.append(x_n.astype(np.float32))
        f4, _ = extract_rr_features(x)
        f4s.append(f4)

    # 全局轨迹特征：对完整 10 分钟信号一次性 RR 提取
    _, global_rr = extract_rr_features(rel)
    tf = (
        compute_traj_rr_features(global_rr).astype(np.float32)
        if len(global_rr) > 0
        else np.zeros(5, dtype=np.float32)
    )

    rr_f = np.array(
        [np.concatenate([f4s[wi], tf]) for wi in range(n_windows)], dtype=np.float32
    )
    return np.stack(wins)[:, np.newaxis, :], rr_f
