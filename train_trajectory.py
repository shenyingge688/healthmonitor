"""
Script: train_trajectory.py
超前 5 分钟心律失常预测 — 训练引擎

训练策略:
  - PTB-XL 骨干预训练加载
  - EMA 权重平均
  - Focal Loss + Label Smoothing
  - 余弦退火 warmup + 学习率衰减
  - 6 类临床固定权重
  - 早期停止
"""
import os
import gc
import glob
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

from dl_model import ArrhythmiaWarningNet

torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# =========================================================
def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None
    datasets = []
    print(f"加载 {split_prefix} shards: {len(shard_paths)} 个文件")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "X_rr" in data:
            datasets.append(TensorDataset(data["X"], data["X_rr"], data["Y"]))
        else:
            dummy_rr = torch.zeros(len(data["X"]), 39, 9, dtype=torch.float16)
            datasets.append(TensorDataset(data["X"], dummy_rr, data["Y"]))
    return ConcatDataset(datasets) if datasets else None


def compute_class_counts(dataset, num_classes=6):
    counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        _, _, y = dataset[i]
        counts[y] += 1
    return counts.clamp_min(1).to(device)


# =========================================================
# Loss
# =========================================================

SMOOTH = 0.12


def soft_focal_loss(logits, targets_soft, class_weights, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -(targets_soft * log_probs).sum(dim=-1)
    target_idx = targets_soft.argmax(dim=-1)
    w = class_weights[target_idx]
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce * w


# =========================================================
# EMA
# =========================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = ArrhythmiaWarningNet(n_windows=39).to(device)
        self.shadow.load_state_dict(model.state_dict())
        self.shadow.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(p.data)


# =========================================================
# LR Schedule
# =========================================================

def lr_lambda(step, warmup=1500, total=40000):
    min_factor = 0.01
    if step < warmup:
        return min_factor + (1.0 - min_factor) * (step / warmup)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))


# =========================================================
# 训练
# =========================================================

def train_one_epoch(epoch, total_epochs, model, ema, loader, optimizer,
                    scheduler, scaler, class_weights, step_counter):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs}]",
                leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        bx, bx_rr, y = batch
        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(bx, x_rr=bx_rr)["logits"]

        # Label smoothing
        y_oh = F.one_hot(y, 6).float()
        y_sm = y_oh * (1.0 - SMOOTH) + SMOOTH / 6

        loss = soft_focal_loss(logits.float(), y_sm, class_weights)
        loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_before <= scale_after:
            scheduler.step()
            ema.update(model)
            total_loss += loss.item()
            valid_batches += 1
            step_counter[0] += 1

        if batch_idx % 100 == 0 and batch_idx > 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        avg_time = elapsed / max(batch_idx + 1, 1)
        remain = avg_time * (len(loader) - batch_idx - 1)
        lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "Loss": f"{loss.item():.3f}",
            "LR": f"{lr:.2e}",
            "ETA": f"{remain / 60:.1f}m",
        })

    return total_loss / max(valid_batches, 1)


@torch.inference_mode()
def validate(model, loader, class_weights):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_targets = [], []

    for batch in loader:
        bx, bx_rr, y = batch
        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y = y.to(device)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(bx, x_rr=bx_rr)["logits"]

        y_oh = F.one_hot(y, 6).float()
        y_sm = y_oh * (1.0 - SMOOTH) + SMOOTH / 6

        loss = soft_focal_loss(logits.float(), y_sm, class_weights).mean()
        total_loss += loss.item()
        n += 1

        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0) if all_targets else 0.0

    return {"loss": total_loss / max(n, 1), "acc": acc, "f1_macro": f1_macro}


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 60)
    print(f"超前 5 分钟心律失常预测训练 | Device = {device}")
    print("=" * 60)

    train_dataset = load_sharded_datasets("train")
    val_dataset = load_sharded_datasets("val")
    if train_dataset is None:
        print("No training data found. Run build_dataset_factory.py first.")
        return

    BATCH_SIZE = 12
    EPOCHS = 30
    EARLY_STOP_PATIENCE = 8

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    class_counts = compute_class_counts(train_dataset)
    print(f"类别分布: {class_counts.int().tolist()}")
    # Normal=0, PVC=1, AFIB=2, VF=3, VT=4, AT/SVT=5
    class_weights = torch.tensor([1.0, 2.0, 3.0, 8.0, 5.0, 4.0], device=device)
    print(f"类别权重: {class_weights.tolist()}")

    # Model
    model = ArrhythmiaWarningNet(n_windows=39).to(device)

    backbone_path = "models/ptbxl_backbone.pth"
    if os.path.exists(backbone_path):
        print("加载 PTB-XL 骨干网络...")
        backbone = torch.load(backbone_path, map_location=device, weights_only=True)
        missing, unexpected = model.window_encoder.load_state_dict(backbone, strict=False)
        if missing:
            print(f"  {len(missing)} keys 缺失 (新架构)，从零初始化")
        else:
            print("  骨干网络加载成功。")

    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    step_counter = [0]
    os.makedirs("models", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup=1500, total=total_steps),
    )

    best_score = -1.0
    best_epoch = 0
    no_improve_count = 0
    save_path = "models/arrhythmia_warning_best.pth"

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            epoch, EPOCHS, model, ema, train_loader, optimizer,
            scheduler, scaler, class_weights, step_counter,
        )
        val_metrics = validate(ema.shadow, val_loader, class_weights)

        epoch_time = time.time() - epoch_start
        clinical_score = 0.5 * val_metrics["acc"] + 0.5 * val_metrics["f1_macro"]

        print(f"\nEpoch [{epoch:02d}/{EPOCHS}] | {epoch_time / 60:.1f}m")
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc: {val_metrics['acc']:.4f}  |  F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Clinical Score: {clinical_score:.4f}")

        if clinical_score > best_score + 1e-4:
            best_score = clinical_score
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "class_weights": class_weights,
                "metrics": val_metrics,
            }, save_path)
            print(f"  → 已保存 (clinical={best_score:.4f})")
        else:
            no_improve_count += 1
            print(f"  → 无提升 ({no_improve_count}/{EARLY_STOP_PATIENCE})")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch} score {best_score:.4f})")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n✅ 训练完成。最佳临床得分: {best_score:.4f} @ epoch {best_epoch} → {save_path}")


if __name__ == "__main__":
    main()
