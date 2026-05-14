"""
Script: train_trajectory.py
Version: V8.3 Final (Synchronized Tensor Flow)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import os

from dl_model import LatentDynamicsForecastingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(file_path):
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    # 返回四元组，张量维度匹配: X[B, S, 1, L], Y_pvc[B, S], Y_afib[B, S], Y_vt[B, S, 3]
    return TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])

def discrete_survival_loss(logits, targets, pos_weight=6.0):
    mask = (targets != -1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    weighted_bce = bce * (1.0 + (pos_weight - 1.0) * (targets == 1).float())
    return (weighted_bce * mask).sum() / torch.clamp(mask.sum(), min=1.0)

def main():
    print(f"🚀 初始化 PTFN V8.3-Final 核心动力学引擎... [Device: {device}]")
    train_loader = DataLoader(load_dataset('dataset/ptfn_train.pt'), batch_size=8, shuffle=True)
    val_loader = DataLoader(load_dataset('dataset/ptfn_val.pt'), batch_size=8, shuffle=False)

    model = LatentDynamicsForecastingNet().to(device)
    backbone_path = 'models/ptbxl_backbone.pth'
    if os.path.exists(backbone_path):
        model.window_encoder.load_state_dict(torch.load(backbone_path, map_location=device, weights_only=True), strict=False)

    for param in model.window_encoder.parameters(): param.requires_grad = False
    model.window_encoder.lead_projector.weight.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': model.window_encoder.lead_projector.parameters(), 'lr': 1e-3},
        {'params': model.macro_gru.parameters(), 'lr': 1e-4},
        {'params': model.structured_heads.parameters(), 'lr': 1e-4}
    ])

    bce_loss_fn = nn.BCEWithLogitsLoss()
    EPOCHS = 50
    best_val_auprc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for name, module in model.window_encoder.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): module.eval()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]", leave=False, dynamic_ncols=True)
        
        # 解包 4 个张量，精准对齐
        for bx, by_pvc_seq, by_afib_seq, by_vt_seq in pbar:
            bx, by_pvc_seq, by_afib_seq, by_vt_seq = bx.to(device), by_pvc_seq.to(device), by_afib_seq.to(device), by_vt_seq.to(device)
            optimizer.zero_grad()
            
            out = model(bx)
            logits_vt = out["preds"]["vt_hazard_logits"] 
            logits_pvc = out["preds"]["pvc_logits"]
            logits_afib = out["preds"]["afib_logits"]
            
            seq_len = logits_vt.size(1)
            total_loss = 0
            
            # 逐时间步密集推演 Loss 计算
            for t in range(seq_len):
                loss_vt = discrete_survival_loss(logits_vt[:, t, :], by_vt_seq[:, t, :], pos_weight=6.0)
                loss_pvc = bce_loss_fn(logits_pvc[:, t], by_pvc_seq[:, t])
                loss_afib = bce_loss_fn(logits_afib[:, t], by_afib_seq[:, t])
                
                # 动态分配权重，主攻 VT
                total_loss += (0.1 * loss_pvc + 0.1 * loss_afib + 1.0 * loss_vt)
            
            total_loss = total_loss / seq_len
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix(Loss=f"{total_loss.item():>6.4f}")
            
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, _, _, by_vt_seq in val_loader:
                bx = bx.to(device)
                out = model(bx)
                
                final_logits = out["preds"]["vt_hazard_logits"][:, -1, :] 
                hazards = 0.5 * torch.sigmoid(final_logits)
                log_survival = torch.sum(torch.log(1.0 - hazards + 1e-6), dim=1)
                risk_5m = 1.0 - torch.exp(log_survival)
                
                val_preds.extend(risk_5m.cpu().numpy())
                is_event = (by_vt_seq[:, -1, :] == 1).any(dim=1).float()
                val_trues.extend(is_event.numpy())
                
        val_auprc = average_precision_score(val_trues, val_preds) if sum(val_trues)>0 else 0
        print(f"Epoch {epoch+1:02d} | Train Loss: {running_loss/len(train_loader):.4f} | Val AUPRC: {val_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 'models/ptfn_v83_core.pth')

if __name__ == "__main__":
    main()