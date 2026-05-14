"""
Script: train_trajectory.py
Version: V10.2 (Turbo Speed + AMP + Sharded Dataloading)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import os
import glob
from dl_model import HierarchicalHazardNet

# 🚀 提速黑科技 1：开启 cuDNN 自动算法寻优，对固定输入形状的 CNN 提速极大
torch.backends.cudnn.benchmark = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None
        
    datasets = []
    print(f"📦 正在加载 {split_prefix} 数据集，共发现 {len(shard_paths)} 个 Shard...")
    for path in shard_paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        datasets.append(TensorDataset(
            data['X'], data['Y_rhythm'], data['Y_criticality'], data['Y_hazard'],
            data['M_rhythm'], data['M_criticality'], data['M_hazard']
        ))
    return ConcatDataset(datasets)

def main():
    print(f"🚀 初始化 V10.2 极速演化引擎 ... [Device: {device}]")
    
    train_dataset = load_sharded_datasets('train')
    val_dataset = load_sharded_datasets('val')
    
    if train_dataset is None:
        print("⚠️ 未找到任何训练 Shard 文件，请先运行 build_dataset_factory.py")
        return

    # 🚀 提速黑科技 2：解锁多进程 CPU 搬运工，开启锁页内存 (pin_memory)
    # 注意：Windows 下必须确保该逻辑在 main() 且被 __main__ 调用下运行
    workers = 4 # 如果你的 CPU 是 8 核以上，可以尝试开到 6 或 8
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers, pin_memory=True)

    model = HierarchicalHazardNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 🚀 提速黑科技 3：初始化 AMP (自动混合精度) 梯度缩放器
    scaler = torch.amp.GradScaler('cuda')

    crit_weights = torch.tensor([1.0, 2.0, 5.0, 8.0], device=device)

    EPOCHS = 50
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        valid_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]", leave=False, dynamic_ncols=True)
        
        for bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz in pbar:
            if torch.isnan(bx).any(): continue
            
            # 转移到显卡，保持原始 dtype (Float16) 或者让 PyTorch 自己管
            bx = bx.to(device, non_blocking=True) 
            y_rhy = y_rhy.to(device, non_blocking=True)
            y_cri = y_cri.to(device, non_blocking=True)
            y_haz = y_haz.to(device, non_blocking=True)
            m_rhy = m_rhy.to(device, non_blocking=True)
            m_cri = m_cri.to(device, non_blocking=True)
            m_haz = m_haz.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # 比传统的 zero_grad() 更快
            
            # 在 autocast 上下文中仅执行前向传播
            with torch.amp.autocast('cuda'):
                out = model(bx)["preds"]
                
                # 取出最后时间步
                rhythm_logits_last = out["rhythm_logits"][:, -1, :]      
                criticality_logits_last = out["criticality_logits"][:, -1, :] 
                hazard_probs_last = out["hazard_probs"][:, -1, :]        
                
            # 将 Loss 计算移出 autocast 并强转回 float32 
            # (这是 PyTorch AMP 官方针对自定义概率 Loss 的强制安全规范，能极大地防止梯度爆炸)
            loss_r = F.cross_entropy(rhythm_logits_last.float(), y_rhy, reduction='none')
            loss_r = (loss_r * m_rhy).sum() / torch.clamp(m_rhy.sum(), min=1e-5)
            
            loss_c = F.cross_entropy(criticality_logits_last.float(), y_cri, weight=crit_weights.float(), reduction='none')
            loss_c = (loss_c * m_cri).sum() / torch.clamp(m_cri.sum(), min=1e-5)
            
            loss_h = F.binary_cross_entropy(hazard_probs_last.float(), y_haz.float(), reduction='none').mean(dim=-1)
            loss_h = (loss_h * m_haz).sum() / torch.clamp(m_haz.sum(), min=1e-5)
            
            total_loss = 1.0 * loss_r + 1.5 * loss_c + 1.0 * loss_h
            
            # 使用 scaler 缩放梯度并反向传播
            scaler.scale(total_loss).backward()
            
            # 梯度裁剪前需要先 unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
            valid_batches += 1
            pbar.set_postfix(Loss=f"{total_loss.item():>6.4f}")
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {running_loss/max(1, valid_batches):.4f}")
        
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/ptfn_v10_core.pth')

if __name__ == "__main__":
    # 在 Windows 系统上强制要求这句入口保护，否则开启 DataLoader num_workers 会无限套娃报错
    main()