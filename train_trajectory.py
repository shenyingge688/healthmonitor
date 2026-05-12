"""
核心: 异构多任务 Loss (Poisson + BCE) | 状态演化平滑正则化
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import warnings
from dl_model import LatentDynamicsForecastingNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    data = torch.load(path, map_location='cpu')
    return TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])

train_loader = DataLoader(load_data('dataset/ptfn_train.pt'), batch_size=32, shuffle=True)
model = LatentDynamicsForecastingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 异构损失函数群
poisson_loss = nn.PoissonNLLLoss(log_input=True)
bce_logits_loss = nn.BCEWithLogitsLoss()
# VT 极度稀缺，赋予高正样本权重 (例如 20.0)
vt_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))

LAMBDA_SMOOTH = 0.05
EPOCHS = 50
os.makedirs('models', exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for bx, by_pvc, by_afib, by_vt in pbar:
        bx = bx.to(device)
        by_pvc = by_pvc.to(device)
        by_afib = by_afib.to(device)
        by_vt = by_vt.to(device)
        
        # 演示 5 分钟视界 (idx=2) 的联合训练
        bs = bx.size(0)
        h_idx_5m = torch.full((bs,), 2, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        out = model(bx, h_idx_5m)
        preds = out["preds"]
        
        # 计算异构 Loss
        loss_pvc = poisson_loss(preds["pvc_log_rate"], by_pvc)
        loss_afib = bce_logits_loss(preds["afib_logits"], by_afib)
        loss_vt = vt_bce_loss(preds["vt_hazard_logits"], by_vt)
        
        # 潜状态生物学惯性平滑
        h_seq = out["h_seq"] 
        diff_squared = (h_seq[:, 1:, :] - h_seq[:, :-1, :]) ** 2
        loss_smooth = torch.mean(torch.sum(diff_squared, dim=-1))
        
        total_loss = 0.1 * loss_pvc + 0.3 * loss_afib + 0.6 * loss_vt + LAMBDA_SMOOTH * loss_smooth
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix(L_VT=f"{loss_vt.item():.2f}", Smth=f"{loss_smooth.item():.3f}")
        
    torch.save(model.state_dict(), f'models/ptfn_epoch_latest.pth')