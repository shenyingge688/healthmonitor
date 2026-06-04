"""
Script: train_trajectory.py
Version: 3-class Rhythm (Causal TCN + Energy Envelope + RR Features)
"""
import os
import gc
import glob
import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from dl_model import HierarchicalHazardNet

# =========================================================
torch.backends.cudnn.benchmark = False  # 关闭算法搜索，减少内存碎片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # 清理残留 GPU 状态
torch.cuda.synchronize()


# =========================================================
def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/v15_*_{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None
    datasets = []
    print(f"Loading {split_prefix} shards: {len(shard_paths)}")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        has_rr = "X_rr" in data
        if has_rr:
            datasets.append(TensorDataset(
                data["X"], data["X_rr"],
                data["Y_rhythm"], data["Y_criticality"], data["Y_hazard"],
                data["M_rhythm"], data["M_criticality"], data["M_hazard"],
            ))
        else:
            # backward compat
            dummy_rr = torch.zeros(len(data["X"]), 19, 9, dtype=torch.float16)
            datasets.append(TensorDataset(
                data["X"], dummy_rr,
                data["Y_rhythm"], data["Y_criticality"], data["Y_hazard"],
                data["M_rhythm"], data["M_criticality"], data["M_hazard"],
            ))
    return ConcatDataset(datasets) if datasets else None


def compute_class_counts_from_dataset(dataset, num_rhy=3, num_cri=4):
    rhy_counts = torch.zeros(num_rhy)
    cri_counts = torch.zeros(num_cri)
    for i in range(len(dataset)):
        _, _, y_rhy, y_cri, _, m_rhy, m_cri, _ = dataset[i]
        if m_rhy > 0.5: rhy_counts[y_rhy] += 1
        if m_cri > 0.5: cri_counts[y_cri] += 1
    return rhy_counts.clamp_min(1).to(device), cri_counts.clamp_min(1).to(device)


# =========================================================
# Per-head loss
# =========================================================
def soft_focal_loss(logits, targets_soft, class_weights, gamma=2.0):
    """Focal loss with label-smoothed soft targets + class weights.
    targets_soft: [B, num_classes] probability distributions.
    class_weights: [num_classes] per-class weight tensor."""
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -(targets_soft * log_probs).sum(dim=-1)
    target_idx = targets_soft.argmax(dim=-1)
    w = class_weights[target_idx]
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce * w


SMOOTH = 0.12  # label smoothing: 0.88 target, 0.04 per other class
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
def lr_lambda(step, warmup=1500, total=60000):
    min_factor = 0.01
    if step < warmup:
        return min_factor + (1.0 - min_factor) * (step / warmup)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))


# =========================================================
def train_one_epoch(epoch, total_epochs, model, ema, loader, optimizer,
                    scheduler, scaler, rhy_weights, cri_weights, step_counter):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs}]",
                leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        bx, bx_rr, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch
        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y_rhy = y_rhy.to(device)
        y_cri = y_cri.to(device)
        y_haz = y_haz.to(device)
        m_rhy = m_rhy.to(device)
        m_cri = m_cri.to(device)
        m_haz = m_haz.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            out = model(bx, x_rr=bx_rr)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]
        safe_hazard = (
            torch.nan_to_num(hazard_probs.float(), nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
            .clamp(1e-7, 1.0 - 1e-7)
        )

        # 合并 SVT/AT(3) → AFIB(2)
        y_rhy = torch.where(y_rhy == 3, torch.tensor(2, device=device), y_rhy)
        # Label smoothing: 0.88 target / 0.04 per other class (3 classes)
        y_rhy_oh = F.one_hot(y_rhy, 3).float()
        y_rhy_sm = y_rhy_oh * (1.0 - SMOOTH) + SMOOTH / 3
        y_cri_oh = F.one_hot(y_cri, 4).float()
        y_cri_sm = y_cri_oh * (1.0 - SMOOTH) + SMOOTH / 4

        loss_r_raw = soft_focal_loss(rhythm_logits.float(), y_rhy_sm, rhy_weights)
        loss_r = (loss_r_raw * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)

        loss_c_raw = soft_focal_loss(criticality_logits.float(), y_cri_sm, cri_weights)
        loss_c = (loss_c_raw * m_cri).sum() / m_cri.sum().clamp_min(1e-6)

        loss_h_raw = F.mse_loss(safe_hazard, y_haz.float(), reduction="none").mean(dim=-1)
        loss_h = (loss_h_raw * m_haz).sum() / m_haz.sum().clamp_min(1e-6)

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

        if batch_idx % 100 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()

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
            "ETA": f"{remain / 60:.1f}m",
        })

    return total_loss / max(valid_batches, 1)


# =========================================================
@torch.inference_mode()
def validate(model, loader, rhy_weights, cri_weights):
    model.eval()
    total_loss = 0.0
    n = 0
    all_rhythm_preds, all_rhythm_targets = [], []
    all_crit_preds, all_crit_targets = [], []
    all_hazard_probs, all_hazard_targets = [], []

    for batch in loader:
        bx, bx_rr, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch
        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y_rhy = y_rhy.to(device)
        y_cri = y_cri.to(device)
        y_haz = y_haz.to(device)
        m_rhy = m_rhy.to(device)
        m_cri = m_cri.to(device)
        m_haz = m_haz.to(device)

        with torch.amp.autocast("cuda"):
            out = model(bx, x_rr=bx_rr)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]
        safe_hazard = (
            torch.nan_to_num(hazard_probs.float(), nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
            .clamp(1e-7, 1.0 - 1e-7)
        )

        # 合并 SVT/AT(3) → AFIB(2)
        y_rhy = torch.where(y_rhy == 3, torch.tensor(2, device=device), y_rhy)
        # Label smoothing: 0.88 target / 0.04 per other class (3 classes)
        y_rhy_oh = F.one_hot(y_rhy, 3).float()
        y_rhy_sm = y_rhy_oh * (1.0 - SMOOTH) + SMOOTH / 3
        y_cri_oh = F.one_hot(y_cri, 4).float()
        y_cri_sm = y_cri_oh * (1.0 - SMOOTH) + SMOOTH / 4

        loss_r_raw = soft_focal_loss(rhythm_logits.float(), y_rhy_sm, rhy_weights)
        loss_r = (loss_r_raw * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)

        loss_c_raw = soft_focal_loss(criticality_logits.float(), y_cri_sm, cri_weights)
        loss_c = (loss_c_raw * m_cri).sum() / m_cri.sum().clamp_min(1e-6)

        loss_h_raw = F.mse_loss(safe_hazard, y_haz.float(), reduction="none").mean(dim=-1)
        loss_h = (loss_h_raw * m_haz).sum() / m_haz.sum().clamp_min(1e-6)

        total_loss += (1.0 * loss_r + 1.5 * loss_c + 1.0 * loss_h).item()
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

    rhythm_acc = accuracy_score(all_rhythm_targets, all_rhythm_preds) if all_rhythm_targets else 0.0
    crit_f1 = (
        f1_score(all_crit_targets, all_crit_preds, average="macro", zero_division=0)
        if all_crit_targets else 0.0
    )
    # Hazard: continuous → Pearson correlation (not AUPRC)
    if all_hazard_targets and len(all_hazard_targets) > 1:
        t = np.array(all_hazard_targets)
        p = np.array(all_hazard_probs)
        # Average correlation across 3 horizons
        cors = []
        for h in range(t.shape[1]):
            if np.std(t[:, h]) > 0 and np.std(p[:, h]) > 0:
                cors.append(np.corrcoef(t[:, h], p[:, h])[0, 1])
        hazard_corr = float(np.mean(cors)) if cors else 0.0
    else:
        hazard_corr = 0.0

    return {
        "loss": total_loss / max(n, 1),
        "rhythm_acc": rhythm_acc,
        "crit_f1": crit_f1,
        "hazard_auprc": hazard_corr,  # keep key name for compatibility
    }


# =========================================================
# Main — single stage, all heads train together
# =========================================================
def main():
    print("=" * 72)
    print(f"三分类节律训练 | Device = {device}")
    print("=" * 72)

    train_dataset = load_sharded_datasets("train")
    val_dataset = load_sharded_datasets("val")
    if train_dataset is None:
        print("No training data found. Run build_dataset_factory.py first.")
        return

    BATCH_SIZE = 16
    EPOCHS = 35
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

    # ---- 临床固定权重 ----
    # PVC 标签修复后样本量从 ~0 增至 6474，高权重不再必要，降至 2.0
    # Normal 从 0.3→1.0，避免对正常心搏过于"宽容"
    # AFib 从 1.0→2.0，增强与 PVC 的边界竞争能力
    rhy_counts, cri_counts = compute_class_counts_from_dataset(train_dataset)
    print(f"Rhythm      counts: {rhy_counts.int().tolist()}")
    print(f"Criticality counts: {cri_counts.int().tolist()}")
    rhy_weights = torch.tensor([1.0, 2.0, 2.0], device=device)  # Normal,PVC,室上性心律失常
    cri_weights = torch.tensor([0.2, 3.0, 4.0, 6.0], device=device)  # Safe,PVC-Load,VT,VF
    print(f"  → weights (clinical fixed): R={rhy_weights.tolist()}  C={cri_weights.tolist()}")

    # ---- Model ----
    model = HierarchicalHazardNet().to(device)
    backbone_path = "models/ptbxl_backbone.pth"
    if os.path.exists(backbone_path):
        print("Loading PTBXL backbone...")
        backbone = torch.load(backbone_path, map_location=device, weights_only=True)
        missing, unexpected = model.window_encoder.load_state_dict(backbone, strict=False)
        if missing:
            print(f"  架构变更，backbone 不兼容 ({len(missing)} keys)，从零训练")
        else:
            print("  Backbone loaded.")

    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda")
    step_counter = [0]
    os.makedirs("models", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup=1500, total=total_steps),
    )

    best_clinical_score = -1.0
    best_epoch = 0
    no_improve_count = 0
    save_path = "models/v20_master_best.pth"

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            epoch, EPOCHS, model, ema, train_loader, optimizer,
            scheduler, scaler, rhy_weights, cri_weights, step_counter,
        )
        val_metrics = validate(ema.shadow, val_loader, rhy_weights, cri_weights)

        epoch_time = time.time() - epoch_start
        temp_val = model.heads.hazard_temperature.item()
        clinical_score = 0.35 * val_metrics["rhythm_acc"] + 0.35 * val_metrics["crit_f1"] + 0.3 * val_metrics["hazard_auprc"]

        print(f"\nEpoch [{epoch:02d}/{EPOCHS}] | {epoch_time / 60:.1f}m")
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Rhythm Acc: {val_metrics['rhythm_acc']:.4f}  |  Crit F1: {val_metrics['crit_f1']:.4f}  |  Hazard Corr: {val_metrics['hazard_auprc']:.4f}")
        print(f"  Temp: {temp_val:.3f}  |  Clinical Score: {clinical_score:.4f}")

        if clinical_score > best_clinical_score + 1e-4:
            best_clinical_score = clinical_score
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "epoch": epoch,
                "best_score": best_clinical_score,
                "temperature": temp_val,
                "rhy_weights": rhy_weights,
                "cri_weights": cri_weights,
                "metrics": val_metrics,
            }, save_path)
            print(f"  -> saved (clinical={best_clinical_score:.4f})")
        else:
            no_improve_count += 1
            print(f"  -> no improvement ({no_improve_count}/{EARLY_STOP_PATIENCE})")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch} with score {best_clinical_score:.4f})")
            break

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Best clinical score: {best_clinical_score:.4f} @ epoch {best_epoch}  ->  {save_path}")


if __name__ == "__main__":
    main()
