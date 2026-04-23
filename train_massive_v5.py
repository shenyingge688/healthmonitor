"""
模块名称：智能预警训练引擎 (Training Engine)
模块功能：
    1. 加载并装配经过清洗和类别平衡处理的数据集。
    2. 定义损失函数与优化器。本模块采用带类权重的非对称损失（Asymmetric BCEWithLogitsLoss），
       以加大对漏报（False Negative）的惩罚，符合医疗重症场景的逻辑。
    3. 执行迭代训练、学习率自适应衰减以及模型早停与最佳权重保存。
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
from dl_model import HybridWarningNet

# ==========================================
# 训练环境与配置初始化
# ==========================================
print("🌌 正在启动临床级预警引擎训练...")
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
    exit(f"❌ 严重错误：未发现平衡数据集，请确认数据流水线已执行成功，目标路径: {DATASET_PATH}")

model = HybridWarningNet().to(device)

# ==========================================
# 优化器与非对称损失策略
# ==========================================
# 配置正样本惩罚权重：2.5倍权重意味着模型如果漏报风险样本，将付出更大的 Loss 代价
pos_weight = torch.tensor([2.5]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    
    # 使用 tqdm 进度条实现优美的单行控制台刷新
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False)
    
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        
        optimizer.zero_grad()
        
        # 接收字典输出，取未经过 sigmoid 激活的 logits 计算损失
        outputs = model(bx)
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

print("✅ 模型训练全部结束，最佳权重已保存至 models 目录。")