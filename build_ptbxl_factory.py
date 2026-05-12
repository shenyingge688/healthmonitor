"""
Phase 1 - PTB-XL 形态学预训练数据工厂
提取 500Hz Lead II，降采样至 250Hz，补零至 30 秒，生成 5 大超级类多标签
"""
import pandas as pd
import numpy as np
import wfdb
import ast
import torch
import os
from scipy import signal
from tqdm import tqdm

TARGET_FS = 250
TARGET_PTS = 30 * TARGET_FS # 7500 个点，完美对齐下游的 WindowEncoder

def load_ptbxl_data(df, sampling_rate, path):
    dataset_x, dataset_y = [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 读取 500Hz 高频数据
            record = wfdb.rdrecord(os.path.join(path, row['filename_hr']))
            # PTB-XL 的导联顺序中，Lead II 通常是 index 1
            lead_ii = record.p_signal[:, 1]
            
            # 降采样到 250Hz (5000 points -> 2500 points)
            resampled = signal.resample_poly(lead_ii, TARGET_FS, sampling_rate)
            
            # 独立标准化 (Z-score)
            resampled = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-8)
            
            # 补零至 30 秒 (7500 个点)
            pad_len = TARGET_PTS - len(resampled)
            if pad_len > 0:
                padded = np.pad(resampled, (0, pad_len), 'constant', constant_values=0)
            else:
                padded = resampled[:TARGET_PTS]
                
            dataset_x.append(padded.astype(np.float32))
            dataset_y.append(row['diagnostic_superclass'])
            
        except Exception as e:
            continue
            
    return np.array(dataset_x), np.array(dataset_y)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PTBXL_DIR = os.path.join(BASE_DIR, 'data', 'ptbxl')
    
    print("📊 正在加载 PTB-XL 标签元数据...")
    df = pd.read_csv(os.path.join(PTBXL_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # 将数百种细分诊断映射到 5 大类 (正常, 心梗, STTC改变, 传导异常, 肥厚)
    agg_df = pd.read_csv(os.path.join(PTBXL_DIR, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    
    # 转换为 5 维的多标签 (Multi-Hot) 向量
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    def multi_hot_encode(labels):
        return [1.0 if c in labels else 0.0 for c in classes]
        
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(multi_hot_encode)
    
    # 官方推荐的划分：fold 10 为验证集，其余 1-9 为训练集
    train_df = df[df.strat_fold != 10]
    val_df = df[df.strat_fold == 10]
    
    print("⏳ 开始提取训练集波形 (耗时约 5-10 分钟)...")
    X_train, Y_train = load_ptbxl_data(train_df, 500, PTBXL_DIR)
    
    print("⏳ 开始提取验证集波形...")
    X_val, Y_val = load_ptbxl_data(val_df, 500, PTBXL_DIR)
    
    os.makedirs('dataset', exist_ok=True)
    # 保存时统一扩展维度，变成 [Batch, 1, 7500] 格式
    torch.save({'X': torch.tensor(X_train).unsqueeze(1), 'Y': torch.tensor(Y_train, dtype=torch.float32)}, 'dataset/ptbxl_train.pt')
    torch.save({'X': torch.tensor(X_val).unsqueeze(1), 'Y': torch.tensor(Y_val, dtype=torch.float32)}, 'dataset/ptbxl_val.pt')
    print("🎉 PTB-XL 数据集构建成功！已存入 dataset/ 目录。")