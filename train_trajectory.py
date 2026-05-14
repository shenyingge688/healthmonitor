"""
Script: train_trajectory.py
Version: V10.7 (Clinical Stable Engine - Scheduler BugFix & Inference Optimized)
"""

import os
import glob
import math
import time
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
# 📦 加载 shard 数据 (防爆内存 + 统一精度)
# =========================================================
def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None

    datasets = []
    print(f"📦 加载 {split_prefix} shards: {len(shard_paths)}")

    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        # ⚠️ 统一 float32 数值流，防止 AMP 混合计算时精度溢出
        X = data["X"].float()

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
# 🧠 EMA (避震器，稳定核心)
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# =========================================================
# 📉 Warmup + Cosine 调度策略 (基于 Global Step)
# =========================================================
def lr_lambda(step, warmup=1500, total=60000):
    if step < warmup:
        return max(step / warmup, 1e-4) # 加个底限，防止除以 0 或过小
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


# =========================================================
# 🔥 单 Epoch 训练 (修复 Bug 版)
# =========================================================
def train_one_epoch(
    epoch,
    total_epochs,
    model,
    ema,
    loader,
    optimizer,
    scheduler, # ✅ 修改：传入 scheduler
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

        # ❗ 脏数据熔断
        if not torch.isfinite(bx).all():
            continue

        # 🚀 GPU 搬运
        bx = bx.to(device, non_blocking=True)
        y_rhy = y_rhy.to(device, non_blocking=True)
        y_cri = y_cri.to(device, non_blocking=True)
        y_haz = y_haz.to(device, non_blocking=True)
        m_rhy = m_rhy.to(device, non_blocking=True)
        m_cri = m_cri.to(device, non_blocking=True)
        m_haz = m_haz.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ⚡ AMP Forward
        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        # 📉 多任务 Loss (强转 float32 保证概率计算绝对安全)
        loss_r = F.cross_entropy(rhythm_logits.float(), y_rhy, reduction="none")
        loss_c = F.cross_entropy(criticality_logits.float(), y_cri, weight=crit_weights, reduction="none")
        loss_h = F.binary_cross_entropy(hazard_probs.float(), y_haz.float(), reduction="none").mean(dim=-1)

        # 🎭 Mask 掩码阻断
        loss_r = (loss_r * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)
        loss_c = (loss_c * m_cri).sum() / m_cri.sum().clamp_min(1e-6)
        loss_h = (loss_h * m_haz).sum() / m_haz.sum().clamp_min(1e-6)

        # 🧮 总 Loss 融合
        total = 1.0 * loss_r + 1.5 * loss_c + 1.0 * loss_h # 恢复 C 和 H 的权重比例

        # 🔥 Backward
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)

        # 🚨 物理防爆护盾：梯度的最大模长限制为 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # ✅ 核心修复：按 Batch 推进学习率
        scheduler.step() 
        ema.update(model)

        total_loss += total.item()
        valid_batches += 1
        step_counter[0] += 1

        # 📊 实时日志刷新
        elapsed = time.time() - start_time
        avg_time = elapsed / (batch_idx + 1)
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
# 📊 Validation (使用底层 C++ 极速模式)
# =========================================================
@torch.inference_mode() # ✅ 优化：比 no_grad 更快的极速模式
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    n = 0

    all_rhythm_preds, all_rhythm_targets = [], []
    all_crit_preds, all_crit_targets = [], []
    all_hazard_probs, all_hazard_targets = [], []

    for batch in loader:
        bx, y_rhy, y_cri, y_haz, _, _, _ = batch
        bx = bx.to(device, non_blocking=True)
        y_rhy = y_rhy.to(device, non_blocking=True)
        y_cri = y_cri.to(device, non_blocking=True)
        y_haz = y_haz.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        # 仅用于记录的伪 Loss
        loss_r = F.cross_entropy(rhythm_logits.float(), y_rhy)
        loss_c = F.cross_entropy(criticality_logits.float(), y_cri)
        loss = loss_r + loss_c
        total_loss += loss.item()
        n += 1

        # 收集结果
        all_rhythm_preds.extend(torch.argmax(rhythm_logits, dim=-1).cpu().numpy())
        all_rhythm_targets.extend(y_rhy.cpu().numpy())
        
        all_crit_preds.extend(torch.argmax(criticality_logits, dim=-1).cpu().numpy())
        all_crit_targets.extend(y_cri.cpu().numpy())
        
        all_hazard_probs.extend(hazard_probs.float().cpu().numpy())
        all_hazard_targets.extend(y_haz.cpu().numpy())

    # 计算高级 Metrics
    rhythm_acc = accuracy_score(all_rhythm_targets, all_rhythm_preds)
    crit_f1 = f1_score(all_crit_targets, all_crit_preds, average="macro", zero_division=0)
    
    try:
        hazard_auprc = average_precision_score(all_hazard_targets, all_hazard_probs, average="macro")
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
    print(f"🚀 V10.7 Master Clinical Engine | Device = {device}")
    print("=" * 72)

    train_dataset = load_sharded_datasets("train")
    val_dataset = load_sharded_datasets("val")

    if train_dataset is None:
        print("❌ 未找到训练数据，请先运行 build_dataset_factory.py")
        return

    EPOCHS = 30
    BATCH_SIZE = 32

    # ✅ 优化：drop_last=True 防止尾部残缺 batch 污染 BatchNorm
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    print(f"✅ Train Samples : {len(train_dataset)}")
    print(f"✅ Val Samples   : {len(val_dataset)}")

    model = HierarchicalHazardNet().to(device)

    # 🔥 挂载 PTBXL 预训练骨干
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

    # 计算真实的 Total Steps 用于 Cosine 退火
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, warmup=1500, total=total_steps)
    )

    best_val = float("inf")
    step_counter = [0]

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            epoch, EPOCHS, model, ema, train_loader, optimizer, 
            scheduler, scaler, crit_weights, step_counter # ✅ 修改：传入 scheduler
        )

        val_metrics = validate(ema.shadow, val_loader)

        epoch_time = time.time() - epoch_start
        
        print("\n" + "=" * 72)
        print(f"📊 Epoch [{epoch:02d}/{EPOCHS}] 报告 | 用时: {epoch_time/60:.1f} 分钟")
        print(f"🔥 Train Loss       : {train_loss:.4f}")
        print(f"📉 EMA Val Loss    : {val_metrics['loss']:.4f}")
        print(f"🎯 Rhythm Acc      : {val_metrics['rhythm_acc']:.4f}")
        print(f"🫀 Criticality F1  : {val_metrics['crit_f1']:.4f}")
        print(f"⚠️ Hazard AUPRC    : {val_metrics['hazard_auprc']:.4f}")
        print(f"📈 最终学习率      : {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            os.makedirs("models", exist_ok=True)
            save_path = "models/v10_master_best.pth"
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "metrics": val_metrics,
            }, save_path)
            print(f"💾 ★ 全新记录！模型已锁定并保存 -> {save_path}")
        print("=" * 72)

if __name__ == "__main__":
    main()