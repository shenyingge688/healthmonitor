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
print(f"🚀 正在校验并同步本地临床数据: {DATA_DIR}")
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
            
    print("✅ 临床数据库本地同步就绪。")
except Exception as e:
    print(f"⚠️ 数据库同步提示: {e}")

# ==========================================
# 模块 4：核心数据集构建引擎
# ==========================================
def build_balanced_dataset(record_list, name, risk_threshold=20):
    """
    功能：遍历心电记录，提取时序切片并打标，最终强制 1:1 平衡并加入标签平滑。
    参数：
        risk_threshold: 未来 5 分钟内出现 20 个异常搏动即判定为高危（逼近频发早搏临床标准）。
    """
    X_normal, X_risk = [], []
    
    pbar = tqdm(record_list, desc=f"构建数据集 [{name}]", unit="rec", dynamic_ncols=True, file=sys.stdout)

    for rec in pbar:
        try:
            rec_path = os.path.join(DATA_DIR, rec)
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
            raw_ecg = record.p_signal[:, 0]
            
            # 1. 滤波与重采样 (提速方案)
            clean_raw = clean_ecg_signal(raw_ecg, fs=360)
            ecg = signal.resample_poly(clean_raw, TARGET_FS, 360)
            
            # 2. 全局标准化
            ecg_global_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
            
            # 3. 注释时间轴对齐 (360Hz 映射至 250Hz)
            anno_idx = np.round(annotation.sample * (TARGET_FS / 360)).astype(int)
            # 筛选出所有非正常心律的异常点索引
            risk_idx = anno_idx[~np.isin(np.array(annotation.symbol), ['N', '.', '/'])]

            # 4. 滑动窗口切片提取
            for start in range(0, len(ecg_global_norm) - HISTORY_PTS - PREDICT_PTS, STRIDE_SEC * TARGET_FS):
                predict_start = start + HISTORY_PTS
                predict_end = predict_start + PREDICT_PTS
                
                # 统计未来预测窗口内的异常点命中数
                risk_hits = np.sum((risk_idx >= predict_start) & (risk_idx < predict_end))
                
                # 提取过去 10 分钟的数据，重塑为网络所需的 [300步, 1通道, 500点]
                x_norm = ecg_global_norm[start : predict_start]
                sample = x_norm.reshape(300, 1, 500)
                
                if risk_hits >= risk_threshold:
                    X_risk.append(sample)
                else:
                    X_normal.append(sample)
        except Exception:
            continue

    # 5. 1:1 绝对平衡欠采样
    num_samples = min(len(X_normal), len(X_risk))
    X_final = random.sample(X_normal, num_samples) + random.sample(X_risk, num_samples)
    
    # 6. 标签平滑 (Label Smoothing) - 降低模型过度自信，提供容错死区
    Y_final = [0.05] * num_samples + [0.95] * num_samples 

    combined = list(zip(X_final, Y_final))
    random.shuffle(combined)
    X_final, Y_final = zip(*combined)

    if X_final:
        save_path = os.path.join(SAVE_DIR, f'{name}.pt')
        torch.save({'X': torch.tensor(np.array(X_final), dtype=torch.float32), 
                    'Y': torch.tensor(np.array(Y_final), dtype=torch.float32).unsqueeze(1)},
                   save_path)
        pbar.write(f"📊 已生成 {name} -> 样本量: {len(X_final)} (正负样本比 1:1)")

# ==========================================
# 模块 5：执行控制流
# ==========================================
if __name__ == '__main__':
    demo_cases = ['100', '119', '201', '208', '233']
    all_recs = sorted(list(set([f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')])))
    # 剔除演示集与已知质量极差的病例，构建纯净训练集
    train_recs = [r for r in all_recs if r not in (demo_cases + ['102', '104', '107', '217'])]

    build_balanced_dataset(train_recs, 'train_massive_v5')
    build_balanced_dataset(demo_cases, 'val_demo_v5')