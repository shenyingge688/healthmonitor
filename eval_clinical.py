"""
Script: eval_clinical.py
功能: 零样本加载 V10 Best Model，输出临床级可视化评估图表
"""

import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score

from dl_model import HierarchicalHazardNet

# 设置字体防止中文乱码 (根据你的系统可能需要微调)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sharded_datasets(split_prefix="val"):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    datasets = []
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        datasets.append(TensorDataset(
            data["X"], data["Y_rhythm"], data["Y_criticality"], data["Y_hazard"],
            data["M_rhythm"], data["M_criticality"], data["M_hazard"]
        ))
    return ConcatDataset(datasets)

@torch.inference_mode()
def main():
    print("=" * 60)
    print("🏥 启动 V10 临床评估可视化舱 ...")
    print("=" * 60)

    # 1. 加载数据
    val_dataset = load_sharded_datasets("val")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"📦 已加载验证集样本数: {len(val_dataset)}")

    # 2. 挂载模型与权重
    model = HierarchicalHazardNet().to(device)
    ckpt_path = "models/v10_master_best.pth"
    if not os.path.exists(ckpt_path):
        exit("❌ 找不到 v10_master_best.pth，请先完成训练！")
        
    print(f"🧠 正在加载最佳 EMA 权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"]) # 注意：一定要加载 EMA 的影子权重！
    model.eval()

    all_rhy_preds, all_rhy_targets = [], []
    all_cri_preds, all_cri_targets = [], []
    all_haz_probs, all_haz_targets = [], []

    print("🚀 正在执行全量临床推理，请稍候...")
    for batch in val_loader:
        bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch
        
        if not torch.isfinite(bx).all(): continue
            
        bx = bx.to(device, dtype=torch.float32)
        y_rhy, y_cri, y_haz = y_rhy.to(device), y_cri.to(device), y_haz.to(device)
        m_rhy, m_cri, m_haz = m_rhy.to(device), m_cri.to(device), m_haz.to(device)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        # 利用 Mask 过滤出真实有效的数据
        idx_rhy = (m_rhy > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_rhy) > 0:
            all_rhy_preds.extend(torch.argmax(rhythm_logits[idx_rhy], dim=-1).cpu().numpy())
            all_rhy_targets.extend(y_rhy[idx_rhy].cpu().numpy())
            
        idx_cri = (m_cri > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_cri) > 0:
            all_cri_preds.extend(torch.argmax(criticality_logits[idx_cri], dim=-1).cpu().numpy())
            all_cri_targets.extend(y_cri[idx_cri].cpu().numpy())
            
        idx_haz = (m_haz > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_haz) > 0:
            all_haz_probs.extend(hazard_probs[idx_haz].float().cpu().numpy())
            all_haz_targets.extend(y_haz[idx_haz].cpu().numpy())

    os.makedirs("results", exist_ok=True)
    
    # ==========================================
    # 图表 1：绘制 危急度 (Criticality) 混淆矩阵
    # ==========================================
    print("\n📊 生成临床评估报告...")
    cri_names = ["Safe(0)", "PVC(1)", "VT(2)", "VF(3)"]
    cm = confusion_matrix(all_cri_targets, all_cri_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cri_names, yticklabels=cri_names)
    plt.title('危急度预测 - 混淆矩阵 (Criticality Confusion Matrix)')
    plt.ylabel('真实标签 (True Label)')
    plt.xlabel('模型预测 (Predicted Label)')
    plt.savefig('results/cm_criticality.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 图表 2：绘制 崩溃风险 (Hazard) PR 曲线
    # ==========================================
    # 提取最远端 (5分钟) 的风险概率用于画图
    haz_probs_5m = np.array(all_haz_probs)[:, -1] 
    haz_targets_5m = np.array(all_haz_targets)[:, -1]
    
    precision, recall, _ = precision_recall_curve(haz_targets_5m, haz_probs_5m)
    ap = average_precision_score(haz_targets_5m, haz_probs_5m)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'5-Min Hazard PR Curve (AUPRC = {ap:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='darkorange')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('5分钟级别临床崩溃风险预警 (Hazard PR Curve)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('results/pr_curve_hazard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("✅ 评估完成！图表已保存至 results/ 目录：")
    print("   👉 1. results/cm_criticality.png (查看 VT/VF 漏报情况)")
    print("   👉 2. results/pr_curve_hazard.png (查看风险预警能力)")
    print("=" * 60)

if __name__ == "__main__":
    main()