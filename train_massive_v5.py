"""
HealthMonitor V4.0 - 模型训练与算力分配引擎
功能描述：构建基于多任务学习 (Multi-Task Learning) 的预警机制。
        引入数据增强 (Augmentation)、平滑的 Alpha 惩罚系数，以及 Macro F1 综合监控指标。
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, random_split
import os
from tqdm import tqdm
from dl_model import HybridWarningNet
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, f1_score

class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        # 防数值下溢，防止计算中出现 NaN
        probs = torch.clamp(torch.exp(log_probs), min=1e-6, max=1.0 - 1e-6)
        
        ce_loss = -(targets * log_probs).sum(dim=1)
        hard_targets = targets.argmax(dim=1)
        pt = probs.gather(1, hard_targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[hard_targets]
            
        return focal_loss.mean()

def apply_augmentation(batch_x):
    """在线数据增强：振幅随机缩放与时域掩蔽，提升泛化能力"""
    # 1. 振幅随机缩放 (0.8x ~ 1.2x)
    scale = torch.empty(batch_x.size(0), 1, 1, 1, device=batch_x.device).uniform_(0.8, 1.2)
    batch_x = batch_x * scale
    
    # 2. 时域随机掩蔽 (模拟突发干扰/导联松动)
    if torch.rand(1).item() > 0.5:
        mask_len = 20
        start_idx = torch.randint(0, 500 - mask_len, (1,)).item()
        batch_x[:, :, :, start_idx : start_idx + mask_len] = 0.0
        
    return batch_x

print("🌌 正在启动多任务分类训练引擎 (V4.0)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train_massive_v5.pt')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

try:
    data = torch.load(DATASET_PATH, map_location='cpu', weights_only=False)
    # 【适配 V4.0】传入四个维度的张量数据
    full_dataset = TensorDataset(data['X'], data['Y_future'], data['Y_current'], data['HRV'])
    
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 基于主任务 (Future_Label) 进行平衡采样
    train_indices = train_dataset.indices
    train_Y_future = data['Y_future'][train_indices]
    hard_labels = train_Y_future.argmax(dim=1)
    
    class_counts = torch.bincount(hard_labels, minlength=6)
    class_weights_total = 1.0 / torch.clamp(class_counts.float(), min=1.0)
    sample_weights = class_weights_total[hard_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"📦 数据装载完毕 -> 训练集: {train_size} | 验证集: {val_size}")
    
except FileNotFoundError:
    exit(f"❌ 严重错误：未发现数据集 {DATASET_PATH}")

model = HybridWarningNet().to(device)

# 平缓的类别惩罚矩阵，防止出现把所有正常类预测为恶性的“惊弓之鸟”现象
class_weights = torch.tensor([1.0, 1.0, 1.2, 1.5, 1.5, 1.2], dtype=torch.float32).to(device)
criterion = SoftFocalLoss(alpha=class_weights, gamma=2.0)

# 【极度关键】：去除了 weight_decay，彻底根除了长周期由于 weight_norm 被压成0导致的 NaN 问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

best_score = 0.0 
EPOCHS = 80 # 既然放宽了早停，可以把总轮数上限也稍微调高一点，防止没触碰早停就结束了
patience_counter = 0

# 【用户修改需求】：将早停耐心值从 12 放宽到 20
EARLY_STOP_PATIENCE = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for bx, by_future, by_current, b_hrv in pbar:
        # 物理极值硬截断
        if torch.isnan(bx).any() or torch.isinf(bx).any() or (bx.abs() > 50.0).any(): 
            continue
            
        bx = bx.to(device)
        bx = apply_augmentation(bx)  # 应用数据增强
        by_future, by_current, b_hrv = by_future.to(device), by_current.to(device), b_hrv.to(device)
        
        optimizer.zero_grad()
        # 传递 HRV 特征以激活 Transformer (虽然目前是 MLP 占位符)
        outputs = model(bx, hrv_features=b_hrv)
        
        # 【多任务损失计算】
        loss_future = criterion(outputs["logits"], by_future)
        loss_current = criterion(outputs["logits_current"], by_current)
        
        # 权重分配：预警未来是主任务 (0.7权重)，理解当下是作为中间跳板的辅助任务 (0.3权重)
        total_loss = 0.7 * loss_future + 0.3 * loss_current
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")
        
    avg_train_loss = running_loss / len(train_loader)
    
    # --- 独立评估环境 (仅评估核心任务：Future Prediction) ---
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by_future, by_current, b_hrv in val_loader:
            bx, by_future, b_hrv = bx.to(device), by_future.to(device), b_hrv.to(device)
            outputs = model(bx, hrv_features=b_hrv)
            
            loss = criterion(outputs["logits"], by_future)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs["prob"], dim=1).cpu().numpy()
            targets = torch.argmax(by_future, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    avg_val_loss = val_loss / len(val_loader)
    
    recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0,1,2,3,4,5], zero_division=0)
    vf_recall, vt_recall = recall_per_class[3], recall_per_class[4]
    
    # 引入 Macro F1，强迫模型照顾 Class 0 (正常)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # 评估融合：重症召回占 40%，整体 F1 占 60%
    critical_recall = (vf_recall + vt_recall) / 2.0
    combined_score = (0.4 * critical_recall) + (0.6 * macro_f1)
    
    scheduler.step(combined_score)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | VF Rec: {vf_recall:.4f} | Macro F1: {macro_f1:.4f}")
    
    if combined_score > best_score:
        best_score = combined_score
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'hybrid_v5_massive_best.pth'))
        print(f"   ✅ 找到更优模型，已保存 (综合评分={combined_score:.4f})")
    else:
        patience_counter += 1
        
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⏹️ 综合评价分数连续 {EARLY_STOP_PATIENCE} 轮未提升，触发早停。")
        break

print(f"\n✅ 训练管线结束。")