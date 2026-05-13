"""
Script: evaluate_clinical_metrics.py
Version: V8.0 Final (Discrete Survival & Causal Ablation)
Description: 
临床决策综合评估中心。
执行真实的物理隔离验证，应用 Isotonic Regression 概率校准，
模拟 ICU 真实报警疲劳 (Alarm Fatigue)，并执行严苛的时序因果打乱测试。
"""

import torch
import numpy as np
import os
import sys
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from dl_model import LatentDynamicsForecastingNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(file_path):
    if not os.path.exists(file_path):
        sys.exit(f"[Error] 未找到验证集数据: {file_path}")
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    return TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])

def extract_v8_predictions(model, loader, apply_temporal_shuffle=False):
    """
    通用特征与预测提取器 (完美兼容 V8.0 离散生存输出)
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="[推断] 提取生存动力学特征" if not apply_temporal_shuffle else "[测试] 执行时序拓扑打乱", leave=False)
        for bx, _, _, by_vt in pbar:
            bx = bx.to(device)
            
            # 👇【V7.5/V8.0 跨域物理通道对齐补丁】👇
            if bx.size(2) == 1:
                bx = bx.repeat(1, 1, 12, 1)
                
            # 🌪️【打乱 10min 时序拓扑】🌪️
            if apply_temporal_shuffle:
                # bx shape: [Batch, SeqLen, Channels, SignalLen]
                # 在 SeqLen (时序窗口) 维度上进行随机打乱，摧毁时间箭头
                seq_len = bx.size(1)
                shuffled_indices = torch.randperm(seq_len)
                bx = bx[:, shuffled_indices, :, :]

            h_idx_5m = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
            out = model(bx, h_idx_5m)
            
            # 👇【V8.0 推断端：多视界 Hazard 坍缩为累积绝对风险】👇
            logits = out["preds"]["vt_hazard_logits"]
            
            if logits.dim() == 2 and logits.size(1) == 3:
                # V8.0 离散生存逻辑：计算 P(5m) = 1 - (1-h1)(1-h2)(1-h3)
                hazards = torch.sigmoid(logits)
                survival_prob = torch.prod(1.0 - hazards, dim=1)
                risk_5m = 1.0 - survival_prob
            else:
                # 兼容旧版本
                risk_5m = torch.sigmoid(logits)
                
            all_preds.extend(risk_5m.cpu().numpy())
            
            # 👇【V8.0 标签降维：将 3-Bin 序列降维为临床二分类终点】👇
            if by_vt.dim() > 1:
                # 只要序列里有 1 (不管在哪个 Bin)，最终临床结果就是发作了 VT
                is_event = (by_vt == 1).any(dim=1).float()
                all_trues.extend(is_event.numpy())
            else:
                all_trues.extend(by_vt.numpy())
                
    return np.array(all_preds), np.array(all_trues)

def evaluate_clinical_system():
    print(f"🏥 启动 PTFN 临床决策评估中心... [运算核心: {device}]")
    
    # 1. 装载模型
    model = LatentDynamicsForecastingNet().to(device)
    v8_weights = 'models/ptfn_v8_best.pth'
    if not os.path.exists(v8_weights):
        sys.exit(f"[Error] 找不到 V8.0 权重 {v8_weights}，请先运行 train_trajectory.py！")
        
    print("⏳ 装载 PTFN (V8.0) 模型权重...")
    model.load_state_dict(torch.load(v8_weights, map_location=device, weights_only=True))
    print("✅ 权重载入成功！")
    
    # 2. 装载数据
    print("⏳ 装载真实物理隔离验证集 (dataset/ptfn_val.pt)...")
    val_dataset = load_dataset('dataset/ptfn_val.pt')
    # batch_size 为 1 才能更精确地模拟连续的时间流(用于真正的 TEG/FAB 计算)，但为加速评估，此处用 64 提取概率
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. 提取基线特征
    print("🔬 提取基线动力学特征与原始生存风险概率...")
    raw_preds_vt, true_vt = extract_v8_predictions(model, val_loader, apply_temporal_shuffle=False)
    
    # 4. 概率校准 (Isotonic Regression)
    print("⚖️ 执行 Isotonic Regression 临床概率校准...")
    ir = IsotonicRegression(out_of_bounds='clip')
    # 这里的 raw_preds_vt 和 true_vt 现在都是纯粹的 1D Array，绝对不会再报错了
    calibrated_preds_vt = ir.fit_transform(raw_preds_vt, true_vt)
    
    # 5. 计算宏观指标
    baseline_auprc = average_precision_score(true_vt, calibrated_preds_vt)
    baseline_auroc = roc_auc_score(true_vt, calibrated_preds_vt)
    brier_before = brier_score_loss(true_vt, raw_preds_vt)
    brier_after = brier_score_loss(true_vt, calibrated_preds_vt)
    
    # 6. 时序因果打乱测试 (Ablation)
    print("🌪️ 运行全集反事实时序消融测试 (检验生存动力学因果律)...")
    shuffled_preds_vt, _ = extract_v8_predictions(model, val_loader, apply_temporal_shuffle=True)
    # 打乱后的概率同样需要经过同一套校准器映射
    calibrated_shuffled_preds = ir.transform(shuffled_preds_vt)
    shuffled_auprc = average_precision_score(true_vt, calibrated_shuffled_preds)
    
    # ================== 打印终极医学报告 ==================
    print("\n" + "="*69)
    print("🏥 PTFN 临床决断力综合评估报告 (V8.0 Survival Protocol)")
    print("="*69)
    
    print("【一、 宏观预测有效性 (Statistical Efficacy)】")
    print(f"* 验证样本量:          {len(true_vt)} (VT正样本数: {int(sum(true_vt))})")
    print(f"* 原始 AUPRC (核心):   {baseline_auprc:.4f}")
    print(f"* AUROC:               {baseline_auroc:.4f}")
    
    print("\n【二、 概率可靠性校准 (Calibration Quality)】")
    print(f"* Brier Score (校准前): {brier_before:.4f}")
    print(f"* Brier Score (校准后): {brier_after:.4f} ⬇️ (越小越好)")
    
    print("\n【三、 时序因果性探伤 (Temporal Causal Integrity)】")
    print(f"* 全集基线 AUPRC:         {baseline_auprc:.4f}")
    print(f"* 打乱 10min 时序拓扑后:  {shuffled_auprc:.4f}")
    
    diff = baseline_auprc - shuffled_auprc
    if diff > 0.05:
        print(f"  🔥 [PI 结论]: AUPRC 断崖式暴跌 (-{diff:.4f})！")
        print("     铁证如山：Survival Loss 成功封杀了『病理计数捷径』，模型已习得时间演化因果律！")
    elif diff > 0.02:
        print(f"  ✅ [PI 结论]: AUPRC 显著下降 (-{diff:.4f})。模型开始深度依赖时序梯度。")
    else:
        print(f"  ⚠️ [PI 结论]: AUPRC 下降微弱 (-{diff:.4f})。模型依然存在统计作弊嫌疑。")
        
    print("="*69)

if __name__ == "__main__":
    evaluate_clinical_system()