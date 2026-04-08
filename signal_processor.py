# signal_processor.py
import numpy as np
import scipy.stats as stats

class SignalProcessor:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def extract_features(self, ecg_window, ppg_window, eda_window):
        """提取 12 维多模态特征 """
        features = {}
        
        # ---------------- 1. ECG 特征 (基于 RR 间期) ----------------
        # 简化版 RR 间期模拟 (实际需通过峰值检测获取)
        rr_intervals = np.random.normal(0.8, 0.05, size=5) # 模拟 RR 间期 (秒)
        diff_rr = np.diff(rr_intervals)
        
        features['mean_hr'] = 60 / np.mean(rr_intervals)
        features['sdnn'] = np.std(rr_intervals) * 1000
        features['rmssd'] = np.sqrt(np.mean(diff_rr**2)) * 1000
        features['pnn50'] = np.sum(np.abs(diff_rr) > 0.05) / len(rr_intervals) * 100
        features['lf_hf_ratio'] = 1.0 # 简化默认值 
        
        # ---------------- 2. PPG 特征 ----------------
        ppg_intervals = np.random.normal(0.8, 0.05, size=5)
        features['pulse_rate'] = 60 / np.mean(ppg_intervals)
        features['amplitude'] = np.max(ppg_window) - np.min(ppg_window)
        features['skewness'] = stats.skew(ppg_window)
        features['kurtosis'] = stats.kurtosis(ppg_window)
        
        # ---------------- 3. EDA/GSR 特征 ----------------
        features['skin_conductance_level'] = np.mean(eda_window)
        features['skin_conductance_std'] = np.std(eda_window)
        # 简化的 SCR 计数：检测差值超过阈值的上升沿
        eda_diff = np.diff(eda_window)
        features['num_scr'] = int(np.sum(eda_diff > 0.05))
        
        # 返回按顺序排列的 12 维特征列表，用于模型推理
        return [
            features['mean_hr'], features['sdnn'], features['rmssd'], 
            features['pnn50'], features['lf_hf_ratio'], features['pulse_rate'],
            features['amplitude'], features['skewness'], features['kurtosis'],
            features['skin_conductance_level'], features['skin_conductance_std'],
            features['num_scr']
        ]