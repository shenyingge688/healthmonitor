"""
Script: train_trajectory.py
Version: V10.11 (The Flawless Master - BCE精度溢出修复 & 学习率动力学重构)
"""

import os
import glob
import math
import time
import gc
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)

from dl_model import HierarchicalHazardNet

# =========================================================
# 🚀 CUDA 极致加速
# =========================================================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 📦 加载 shard 数据
# =========================================================
def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None

    datasets = []
    print(f"📦 加载 {split_prefix} shards: {len(shard_paths)}")

    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        X = data["X"] 

        datasets.append(
            TensorDataset(
                X,
                data["Y_rhythm"],
                data["Y_criticality"],
                data["Y_hazard"],
                data["M_rhythm"],
                data["M_criticality"],
                data["M_hazard"],
            )
        )
    return ConcatDataset(datasets)

# =========================================================
# 🧠 EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(p.data)

# =========================================================
# 📉 修复后的动力学 Scheduler (倍率映射)
# =========================================================
def lr_lambda(step, warmup=1500, total=60000):
    min_factor = 0.01  # 保证 LR 永远不会低于 base_lr 的 1%
    
    if step < warmup:
        # 预热期：倍率从 1% 线性爬升到 100%
        return min_factor + (1.0 - min_factor) * (step / warmup)
        
    # 退火期：余弦平滑下降，最低回到 1%
    progress = (step - warmup) / max(total - warmup, 1)
    return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))

# =========================================================
# 🔥 单 Epoch 训练 
# =========================================================
def train_one_epoch(
    epoch,
    total_epochs,
    model,
    ema,
    loader,
    optimizer,
    scheduler,
    scaler,
    crit_weights,
    step_counter,
):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs}]", leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch

        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32, non_blocking=True)
        y_rhy = y_rhy.to(device, non_blocking=True)
        y_cri = y_cri.to(device, non_blocking=True)
        y_haz = y_haz.to(device, non_blocking=True)
        m_rhy = m_rhy.to(device, non_blocking=True)
        m_cri = m_cri.to(device, non_blocking=True)
        m_haz = m_haz.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        # 🚀 修复雷区 1：强行 Clamp，切断由于 Float32 精度漂移导致的 BCE NaN 崩溃
        safe_hazard_probs = hazard_probs.float().clamp(1e-7, 1.0 - 1e-7)

        loss_r = F.cross_entropy(rhythm_logits.float(), y_rhy, reduction="none")
        loss_c = F.cross_entropy(criticality_logits.float(), y_cri, weight=crit_weights, reduction="none")
        loss_h = F.binary_cross_entropy(safe_hazard_probs, y_haz.float(), reduction="none").mean(dim=-1)

        loss_r = (loss_r * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)
        loss_c = (loss_c * m_cri).sum() / m_cri.sum().clamp_min(1e-6)
        loss_h = (loss_h * m_haz).sum() / m_haz.sum().clamp_min(1e-6)

        total = 1.0 * loss_r + 1.5 * loss_c + 1.0 * loss_h

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            scheduler.step() 
            ema.update(model) 
            
            total_loss += total.item()
            valid_batches += 1
            step_counter[0] += 1

        elapsed = time.time() - start_time
        avg_time = elapsed / max(batch_idx + 1, 1)
        remain = avg_time * (len(loader) - batch_idx - 1)
        lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "Loss": f"{total.item():.3f}",
            "R": f"{loss_r.item():.2f}",
            "C": f"{loss_c.item():.2f}",
            "H": f"{loss_h.item():.2f}",
            "LR": f"{lr:.2e}",
            "ETA": f"{remain/60:.1f}m"
        })

    return total_loss / max(valid_batches, 1)

# =========================================================
# 📊 Validation
# =========================================================
@torch.inference_mode()
def validate(model, loader, crit_weights):
    model.eval()
    total_loss = 0.0
    n = 0

    all_rhythm_preds, all_rhythm_targets = [], []
    all_crit_preds, all_crit_targets = [], []
    all_hazard_probs, all_hazard_targets = [], []

    for batch in loader:
        bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch
        
        if not torch.isfinite(bx).all():
            continue
            
        bx = bx.to(device, dtype=torch.float32, non_blocking=True)
        y_rhy = y_rhy.to(device, non_blocking=True)
        y_cri = y_cri.to(device, non_blocking=True)
        y_haz = y_haz.to(device, non_blocking=True)
        m_rhy = m_rhy.to(device, non_blocking=True)
        m_cri = m_cri.to(device, non_blocking=True)
        m_haz = m_haz.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        safe_hazard_probs = hazard_probs.float().clamp(1e-7, 1.0 - 1e-7)

        loss_r = F.cross_entropy(rhythm_logits.float(), y_rhy, reduction="none")
        loss_c = F.cross_entropy(criticality_logits.float(), y_cri, weight=crit_weights, reduction="none")
        loss_h = F.binary_cross_entropy(safe_hazard_probs, y_haz.float(), reduction="none").mean(dim=-1)

        loss_r = (loss_r * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)
        loss_c = (loss_c * m_cri).sum() / m_cri.sum().clamp_min(1e-6)
        loss_h = (loss_h * m_haz).sum() / m_haz.sum().clamp_min(1e-6)

        loss = 1.0 * loss_r + 1.5 * loss_c + 1.0 * loss_h
        total_loss += loss.item()
        n += 1

        idx_rhy = (m_rhy > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_rhy) > 0:
            all_rhythm_preds.extend(torch.argmax(rhythm_logits[idx_rhy], dim=-1).cpu().numpy())
            all_rhythm_targets.extend(y_rhy[idx_rhy].cpu().numpy())
            
        idx_cri = (m_cri > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_cri) > 0:
            all_crit_preds.extend(torch.argmax(criticality_logits[idx_cri], dim=-1).cpu().numpy())
            all_crit_targets.extend(y_cri[idx_cri].cpu().numpy())
            
        idx_haz = (m_haz > 0.5).nonzero(as_tuple=True)[0]
        if len(idx_haz) > 0:
            all_hazard_probs.extend(hazard_probs[idx_haz].float().cpu().numpy())
            all_hazard_targets.extend(y_haz[idx_haz].cpu().numpy())

    rhythm_acc = accuracy_score(all_rhythm_targets, all_rhythm_preds) if len(all_rhythm_targets) > 0 else 0.0
    crit_f1 = f1_score(all_crit_targets, all_crit_preds, average="macro", zero_division=0) if len(all_crit_targets) > 0 else 0.0
    
    try:
        if len(all_hazard_targets) > 0:
            hazard_auprc = average_precision_score(all_hazard_targets, all_hazard_probs, average="macro")
        else:
            hazard_auprc = 0.0
    except:
        hazard_auprc = 0.0

    return {
        "loss": total_loss / max(n, 1),
        "rhythm_acc": rhythm_acc,
        "crit_f1": crit_f1,
        "hazard_auprc": hazard_auprc,
    }

# =========================================================
# 🚀 启动控制台
# =========================================================
def main():
    print("=" * 72)
    print(f"🚀 V10.11 Flawless Master Engine | Device = {device}")
    print("=" * 72)

    train_dataset = load_sharded_datasets("train")
    val_dataset = load_sharded_datasets("val")

    if train_dataset is None:
        print("❌ 未找到训练数据，请先运行 build_dataset_factory.py")
        return

    EPOCHS = 30
    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=True
    )

    print(f"✅ Train Samples : {len(train_dataset)}")
    print(f"✅ Val Samples   : {len(val_dataset)}")

    model = HierarchicalHazardNet().to(device)

    backbone_path = "models/ptbxl_backbone.pth"
    if os.path.exists(backbone_path):
        print("🧠 检测到 PTBXL 预训练骨干，正在进行知识迁移...")
        backbone = torch.load(backbone_path, map_location=device, weights_only=True)
        model.window_encoder.load_state_dict(backbone)
        print("✅ 视神经皮层 (Backbone) 挂载成功")

    ema = EMA(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit_weights = torch.tensor([1.0, 2.0, 5.0, 8.0], device=device)

    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, warmup=1500, total=total_steps)
    )

    best_clinical_score = -1.0 
    step_counter = [0]

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            epoch, EPOCHS, model, ema, train_loader, optimizer, 
            scheduler, scaler, crit_weights, step_counter
        )

        val_metrics = validate(ema.shadow, val_loader, crit_weights)

        epoch_time = time.time() - epoch_start
        
        print("\n" + "=" * 72)
        print(f"📊 Epoch [{epoch:02d}/{EPOCHS}] 报告 | 用时: {epoch_time/60:.1f} 分钟")
        print(f"🔥 Train Loss       : {train_loss:.4f}")
        print(f"📉 EMA Val Loss     : {val_metrics['loss']:.4f}")
        print(f"🎯 Rhythm Acc       : {val_metrics['rhythm_acc']:.4f}")
        print(f"🫀 Criticality F1   : {val_metrics['crit_f1']:.4f}")
        print(f"⚠️ Hazard AUPRC     : {val_metrics['hazard_auprc']:.4f}")
        print(f"📈 最终学习率       : {optimizer.param_groups[0]['lr']:.2e}")
        
        current_clinical_score = (0.4 * val_metrics['crit_f1']) + (0.6 * val_metrics['hazard_auprc'])
        print(f"🌟 复合临床终点得分 : {current_clinical_score:.4f}")

        if current_clinical_score > best_clinical_score:
            best_clinical_score = current_clinical_score
            os.makedirs("models", exist_ok=True)
            save_path = "models/v10_master_best.pth"
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_score": best_clinical_score,
                "metrics": val_metrics,
            }, save_path)
            print(f"💾 ★ 临床指标突破！模型已锁定并保存 -> {save_path}")
        print("=" * 72)

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()