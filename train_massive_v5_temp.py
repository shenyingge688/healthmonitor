"""
模块名称：智能预警训练引擎 (Training Engine - 多分类加权版)
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
from dl_model import HybridWarningNet
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 自定义 Focal Loss (焦点损失函数)
# 作用：动态降低容易分类样本的权重，迫使模型仔细分辨 PVC 和 VT 的细微差别
# ==========================================
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha # 传入您的 class_weights 张量
        self.gamma = gamma # 聚焦参数，通常设为 2.0

    def forward(self, inputs, targets):
        # 计算基础交叉熵 (支持 Label Smoothing 的软标签)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        # 计算 pt (模型对正确类别的预测概率)
        pt = torch.exp(-ce_loss)
        # 施加 Focal 动态调节因子
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# 训练环境与配置初始化
# ==========================================
print("[start] 正在启动临床级多分类预警引擎训练...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'train_massive_v5.pt')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# 数据加载模块
# ==========================================
try:
    data = torch.load(DATASET_PATH, map_location=device, weights_only=False)
    dataloader = DataLoader(TensorDataset(data['X'], data['Y']), batch_size=32, shuffle=True)
except FileNotFoundError:
    exit(f"[error] 严重错误：未发现数据集，请确认数据流水线已执行成功，目标路径: {DATASET_PATH}")
    
model = HybridWarningNet().to(device)

# ==========================================
# 优化器与多分类临床加权损失策略
# ==========================================
# 类别索引: [0: Normal, 1: PVC, 2: AFib, 3: VF, 4: VT, 5: AT]
# 临床重症防漏报权重：极度重罚室颤(VF)和室速(VT)的漏报
class_weights = torch.tensor([1.0, 1.5, 3.0, 6.0, 5.0, 4.0], dtype=torch.float32).to(device)

criterion = WeightedFocalLoss(alpha=class_weights, gamma=2.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ==========================================
# 主训练循环
# ==========================================
best_loss = float('inf')
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False)
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        
        outputs = model(bx)
        # by 已经是 shape 为 (Batch, 6) 的 Label Smoothing 概率分布
        loss = criterion(outputs["logits"], by) 
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d} | 训练 Loss: {avg_loss:.4f} | 学习率 LR: {current_lr:.6f}")
    
    # 最佳权重存档点
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(MODEL_SAVE_DIR, 'hybrid_v5_massive_best.pth')
        torch.save(model.state_dict(), best_model_path)
        
print("[done] 模型训练全部结束，最佳权重已保存至 models 目录。")