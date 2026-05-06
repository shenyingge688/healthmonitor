"""
HealthMonitor V3.0 - 模型训练与算力分配引擎
功能描述：构建基于真实医疗场景中正负样本极端失衡的解决方案。
        支持验证集分离、早停控制、加权随机重采样以及 SoftFocalLoss 处理机制。
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, random_split
import os
from tqdm import tqdm
from dl_model import HybridWarningNet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score

class SoftFocalLoss(nn.Module):
    """
    针对软标签平滑处理后的焦点损失函数。
    为维持对难分类样本的有效惩罚（Gamma焦点效应），
    交叉熵基于软标签计算，而调节系数 pt 基于原硬标签索引计算。
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.clamp(torch.exp(log_probs), min=1e-8, max=1.0)
        
        # 交叉熵部分 (使用 Soft Labels)
        ce_loss = -(targets * log_probs).sum(dim=1)
        
        # pt 计算部分 (使用 Hard Labels 确保 pt 区间具有区分度)
        hard_targets = targets.argmax(dim=1)
        pt = probs.gather(1, hard_targets.unsqueeze(1)).squeeze(1)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 应用类别代价矩阵
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[hard_targets]
            
        return focal_loss.mean()

print("🌌 正在启动多分类训练引擎...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train_massive_v5.pt')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# [管线调度] 数据张量装载与动态平衡构建
# ==========================================
try:
    data = torch.load(DATASET_PATH, map_location='cpu', weights_only=False)
    full_dataset = TensorDataset(data['X'], data['Y'])
    
    # 预留 15% 建立同源验证集，监控未见样本的泛化情况
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 获取属于训练集那部分的真实标签集合，用于构建频率权重
    train_indices = train_dataset.indices
    train_Y = data['Y'][train_indices]
    hard_labels = train_Y.argmax(dim=1)
    
    # 计算类别的反比例权重，分配给加权随机采样器
    class_counts = torch.bincount(hard_labels, minlength=6)
    class_weights_total = 1.0 / torch.clamp(class_counts.float(), min=1.0)
    sample_weights = class_weights_total[hard_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 装载数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"📦 数据装载完毕 -> 训练集: {train_size} | 验证集: {val_size}")
    
except FileNotFoundError:
    exit(f"❌ 严重错误：未发现数据集 {DATASET_PATH}")

model = HybridWarningNet().to(device)

# 代价敏感博弈权重设定，针对极危重症类分配高位惩罚
class_weights = torch.tensor([1.0, 1.5, 2.5, 5.0, 4.5, 3.0], dtype=torch.float32).to(device)
criterion = SoftFocalLoss(alpha=class_weights, gamma=2.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ==========================================
# [迭代循环]
# ==========================================
best_val_recall = 0.0 
EPOCHS = 50
patience_counter = 0
EARLY_STOP_PATIENCE = 12

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    # 批次遍历
    for bx, by in pbar:
        if torch.isnan(bx).any() or torch.isinf(bx).any(): continue
        bx, by = bx.to(device), by.to(device)
        
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs["logits"], by)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_train_loss = running_loss / len(train_loader)
    
    # 独立评估环境
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            loss = criterion(outputs["logits"], by)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs["prob"], dim=1).cpu().numpy()
            targets = torch.argmax(by, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = val_loss / len(val_loader)
    
    # 计算评估指标
    recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0,1,2,3,4,5], zero_division=0)
    vf_recall, vt_recall = recall_per_class[3], recall_per_class[4]
    
    # 以两个核心极危重症类的召回平均数作为评价指标
    critical_recall = (vf_recall + vt_recall) / 2.0
    
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | VF Recall: {vf_recall:.4f} | VT Recall: {vt_recall:.4f}")
    
    # 权重留存决策
    if critical_recall > best_val_recall:
        best_val_recall = critical_recall
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'hybrid_v5_massive_best.pth'))
        print(f"   ✅ 找到更优模型，已保存 (综合重症召回={critical_recall:.4f})")
    else:
        patience_counter += 1
        
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⏹️ 验证集召回率连续 {EARLY_STOP_PATIENCE} 轮未提升，停止训练。")
        break

print(f"\n✅ 训练管线结束。")