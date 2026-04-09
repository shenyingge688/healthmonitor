import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from dl_model import HybridWarningNet 

print("🌌 正在启动泛化预警引擎训练 (松绑版)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 请确保您之前运行的是最新的 1:1 自动平衡版 build_dataset_factory.py
    data = torch.load('dataset/train_massive_v5.pt', weights_only=False)
    dataloader = DataLoader(TensorDataset(data['X'], data['Y']), batch_size=32, shuffle=True)
except FileNotFoundError:
    exit("❌ 未发现平衡数据集，请运行 build_dataset_factory.py")

model = HybridWarningNet().to(device)

# 🎯 松绑：极小的权重衰减，既防噪又不至于让模型摆烂
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_loss = float('inf')

# 🎯 给足时间：训练 50 轮，让它把 50% 的摇摆概率压下去
for epoch in range(50): 
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/50", unit="batch", leave=False)
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        _, prob = model(bx) 
        loss = criterion(prob, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"📅 Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/hybrid_v5_massive_best.pth')
    
    # 🎯 适度早停：门槛设低一点，让它充分学习
    if avg_loss < 0.015:
        print("✨ 模型已深刻掌握泛化特征，提前结束训练。")
        break