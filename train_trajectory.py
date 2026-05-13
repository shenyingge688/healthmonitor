"""
Script: train_trajectory.py
Version: V8.0 (Discrete-Time Survival Dynamics Protocol)
Description: 
PTFN 终极形态。在 V7.5 域适应基线的基础上，抛弃二分类交叉熵，
全面引入 Discrete-Time Survival Analysis (离散时间生存分析) 作为优化目标，
强迫模型学习时间梯度与病理演化的因果箭头。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import os
import warnings
import sys

from dl_model import LatentDynamicsForecastingNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(file_path):
    if not os.path.exists(file_path):
        sys.exit(f"[Error] 数据集文件未找到: {file_path}")
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    # [极其重要提示] 这里的 data['Y_vt'] 必须是已经被你转换成 [Batch, 3] 形状的 3-Bin 序列标签！
    return TensorDataset(data['X'], data['Y_pvc'], data['Y_afib'], data['Y_vt'])

# 👇【V8.0 核心损失函数：带有掩码与正样本加权的生存交叉熵】👇
def discrete_survival_loss(logits, targets, pos_weight=20.0):
    """
    logits: [Batch, 3]
    targets: [Batch, 3], 包含 0, 1 和 -1 (删失掩码)
    """
    # 生成有效掩码 (过滤掉 -1 的位置)
    mask = (targets != -1).float()
    
    # 基础 BCE Loss (不对外输出，内部计算)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    
    # 手动实现 pos_weight，仅对 target == 1 的位置施加惩罚权重
    weighted_bce = bce * (1.0 + (pos_weight - 1.0) * (targets == 1).float())
    
    # 掩码过滤并求平均
    return (weighted_bce * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def main():
    print(f"[System] 初始化 PTFN V8.0 离散生存动力学引擎. 运算设备: {device}")
    
    train_dataset = load_dataset('dataset/ptfn_train.pt')
    val_dataset = load_dataset('dataset/ptfn_val.pt')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = LatentDynamicsForecastingNet().to(device)
    
    # V8.0 依然从最纯粹的 PTB-XL 底座开始，以保持单变量控制原则
    backbone_path = 'models/ptbxl_backbone.pth'
    if os.path.exists(backbone_path):
        print(f"[I/O] 挂载预训练形态学特征提取器权重: {backbone_path}")
        model.window_encoder.load_state_dict(
            torch.load(backbone_path, map_location=device, weights_only=True), 
            strict=False
        )
    else:
        sys.exit("[Error] 未检测到预训练权重 ptbxl_backbone.pth！")

    # 初始绝对冻结 CNN 编码器
    for param in model.window_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': model.macro_gru.parameters(), 'lr': 1e-4},
        {'params': model.cross_attn_decoder.parameters(), 'lr': 1e-4},
        {'params': model.structured_heads.parameters(), 'lr': 1e-4}
    ])

    poisson_loss = nn.PoissonNLLLoss(log_input=True)     
    bce_logits_loss = nn.BCEWithLogitsLoss()             
    # 注意：这里删除了 vt_bce_loss 的初始化，改用我们自定义的 survival_loss

    LAMBDA_SMOOTH = 0.05
    EPOCHS = 50
    best_val_auprc = 0.0
    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        
        # --- V7.5 沿用的严格渐进式解冻调度器 ---
        if epoch == 5:
            print(f"\n[Scheduler] Epoch {epoch+1:02d}: 触发 Stage 4 严格解冻 (lr=1e-5)...")
            unfrozen_count = 0
            for name, param in model.window_encoder.named_parameters():
                if 'layer4' in name or 'block4' in name or 'stage4' in name:
                    param.requires_grad = True
                    optimizer.add_param_group({'params': param, 'lr': 1e-5})
                    unfrozen_count += 1
            if unfrozen_count == 0:
                raise ValueError("[Fatal Error] 严格解冻失败！未匹配到 stage4。")

        elif epoch == 10:
            print(f"\n[Scheduler] Epoch {epoch+1:02d}: 触发 Stage 3 严格解冻 (lr=1e-5)...")
            for name, param in model.window_encoder.named_parameters():
                if 'layer3' in name or 'block3' in name or 'stage3' in name:
                    if not param.requires_grad: 
                        param.requires_grad = True
                        optimizer.add_param_group({'params': param, 'lr': 1e-5})
        # ----------------------------------------

        # ------------------- 训练前向与反向传播 -------------------
        model.train()
        
        # BatchNorm 强制静默
        for name, module in model.window_encoder.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval() 

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1:02d}/{EPOCHS}] Training", leave=False, dynamic_ncols=True)
        
        for bx, by_pvc, by_afib, by_vt_survival in pbar:
            bx, by_pvc, by_afib, by_vt_survival = bx.to(device), by_pvc.to(device), by_afib.to(device), by_vt_survival.to(device)
            
            # 单通道复制扩维伪装
            if bx.size(2) == 1:
                bx = bx.repeat(1, 1, 12, 1)
                
            h_idx_5m = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            out = model(bx, h_idx_5m)
            preds = out["preds"]
            
            # 异构损失群计算
            loss_pvc = poisson_loss(preds["pvc_log_rate"], by_pvc)
            loss_afib = bce_logits_loss(preds["afib_logits"], by_afib)
            
            # 👇 V8.0: 调用自定义的离散生存损失
            loss_vt = discrete_survival_loss(preds["vt_hazard_logits"], by_vt_survival, pos_weight=20.0)
            
            h_seq = out["h_seq"] 
            diff_squared = (h_seq[:, 1:, :] - h_seq[:, :-1, :]) ** 2
            loss_smooth = torch.mean(torch.sum(diff_squared, dim=-1))
            
            total_loss = 0.1 * loss_pvc + 0.3 * loss_afib + 0.6 * loss_vt + LAMBDA_SMOOTH * loss_smooth
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix(Loss=f"{total_loss.item():.3f}", VT=f"{loss_vt.item():.3f}", Smth=f"{loss_smooth.item():.3f}")
            
        pbar.close()
            
        # ------------------- 验证评估模块 (推导累积风险) -------------------
        model.eval()
        val_preds_vt = []
        val_true_vt = []
        
        with torch.no_grad():
            for bx, _, _, by_vt_survival in val_loader:
                bx = bx.to(device)
                
                # 单通道复制扩维伪装
                if bx.size(2) == 1:
                    bx = bx.repeat(1, 1, 12, 1)

                h_idx_5m = torch.full((bx.size(0),), 2, dtype=torch.long).to(device)
                out = model(bx, h_idx_5m)
                
                # 👇 V8.0: 提取 Hazard Logits 并计算 5 分钟累积风险
                logits = out["preds"]["vt_hazard_logits"] # shape: [Batch, 3]
                hazards = torch.sigmoid(logits)           # 转为概率 h1, h2, h3
                
                # 计算生存概率 S(5m) = (1-h1)*(1-h2)*(1-h3)
                survival_prob = torch.prod(1.0 - hazards, dim=1)
                
                # 累积爆发风险 P(5m) = 1 - S(5m)
                risk_5m = 1.0 - survival_prob
                val_preds_vt.extend(risk_5m.cpu().numpy())
                
                # 从 3-Bin 序列中提取验证集所需的单一标签 (只要序列里有 1，就是正样本)
                is_event = (by_vt_survival == 1).any(dim=1).float()
                val_true_vt.extend(is_event.numpy())
                
        try:
            val_auprc = average_precision_score(val_true_vt, val_preds_vt)
        except Exception:
            val_auprc = 0.0
            
        print(f"[Summary] Epoch {epoch+1:02d} | Mean Train Loss: {running_loss/len(train_loader):.4f} | Validation VT AUPRC (Cumulative): {val_auprc:.4f}")
        
        torch.save(model.state_dict(), 'models/ptfn_epoch_latest.pth')
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            # 保存为 v8 专属后缀，防覆盖之前的基线
            torch.save(model.state_dict(), 'models/ptfn_v8_best.pth') 
            print(f"[Checkpoint] 记录刷新。已序列化保存 V8.0 最优生存动力学模型至 models/ptfn_v8_best.pth")

    print(f"[System] V8.0 训练协议执行完毕。最优验证集 AUPRC: {best_val_auprc:.4f}")

if __name__ == "__main__":
    main()