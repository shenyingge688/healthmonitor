"""
Script: train_trajectory.py
Version: V11.0 (Single-Stage Training — no curriculum, no GRU pollution)
"""
import os
import gc
import glob
import math
import time
from copy import deepcopy

import torch
import torch.nn as nn
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
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
def load_sharded_datasets(split_prefix):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    if not shard_paths:
        return None
    datasets = []
    print(f"Loading {split_prefix} shards: {len(shard_paths)}")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        datasets.append(TensorDataset(
            data["X"], data["Y_rhythm"], data["Y_criticality"], data["Y_hazard"],
            data["M_rhythm"], data["M_criticality"], data["M_hazard"],
        ))
    return ConcatDataset(datasets)


def compute_class_counts_from_dataset(dataset, num_rhy=4, num_cri=4):
    rhy_counts = torch.zeros(num_rhy)
    cri_counts = torch.zeros(num_cri)
    for i in range(len(dataset)):
        _, y_rhy, y_cri, _, m_rhy, m_cri, _ = dataset[i]
        if m_rhy > 0.5: rhy_counts[y_rhy] += 1
        if m_cri > 0.5: cri_counts[y_cri] += 1
    return rhy_counts.clamp_min(1).to(device), cri_counts.clamp_min(1).to(device)


# =========================================================
# Per-head loss
# =========================================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


def label_smoothing_bce_loss(probs, targets, eps=0.1):
    targets_smooth = targets * (1.0 - 2.0 * eps) + eps
    return F.binary_cross_entropy(probs, targets_smooth, reduction="none")


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
                    scheduler, scaler, rhy_loss_fn, cri_weights,
                    hazard_smooth_eps, step_counter):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs}]",
                leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz = batch
        if not torch.isfinite(bx).all():
            continue

        bx = bx.to(device, dtype=torch.float32)
        y_rhy = y_rhy.to(device)
        y_cri = y_cri.to(device)
        y_haz = y_haz.to(device)
        m_rhy = m_rhy.to(device)
        m_cri = m_cri.to(device)
        m_haz = m_haz.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        safe_hazard = (
            torch.nan_to_num(hazard_probs.float(), nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
            .clamp(1e-7, 1.0 - 1e-7)
        )

        loss_r_raw = rhy_loss_fn(rhythm_logits.float(), y_rhy)
        loss_r = (loss_r_raw * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)

        loss_c_raw = F.cross_entropy(
            criticality_logits.float(), y_cri, weight=cri_weights, reduction="none"
        )
        loss_c = (loss_c_raw * m_cri).sum() / m_cri.sum().clamp_min(1e-6)

        loss_h_raw = label_smoothing_bce_loss(
            safe_hazard, y_haz.float(), eps=hazard_smooth_eps
        ).mean(dim=-1)
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
def validate(model, loader, rhy_loss_fn, cri_weights, hazard_smooth_eps):
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

        bx = bx.to(device, dtype=torch.float32)
        y_rhy = y_rhy.to(device)
        y_cri = y_cri.to(device)
        y_haz = y_haz.to(device)
        m_rhy = m_rhy.to(device)
        m_cri = m_cri.to(device)
        m_haz = m_haz.to(device)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            rhythm_logits = out["rhythm_logits"][:, -1]
            criticality_logits = out["criticality_logits"][:, -1]
            hazard_probs = out["hazard_probs"][:, -1]

        safe_hazard = (
            torch.nan_to_num(hazard_probs.float(), nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
            .clamp(1e-7, 1.0 - 1e-7)
        )

        loss_r_raw = rhy_loss_fn(rhythm_logits.float(), y_rhy)
        loss_r = (loss_r_raw * m_rhy).sum() / m_rhy.sum().clamp_min(1e-6)

        loss_c_raw = F.cross_entropy(
            criticality_logits.float(), y_cri, weight=cri_weights, reduction="none"
        )
        loss_c = (loss_c_raw * m_cri).sum() / m_cri.sum().clamp_min(1e-6)

        loss_h_raw = label_smoothing_bce_loss(
            safe_hazard, y_haz.float(), eps=hazard_smooth_eps
        ).mean(dim=-1)
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
    try:
        hazard_auprc = (
            average_precision_score(all_hazard_targets, all_hazard_probs, average="macro")
            if all_hazard_targets else 0.0
        )
    except Exception:
        hazard_auprc = 0.0

    return {
        "loss": total_loss / max(n, 1),
        "rhythm_acc": rhythm_acc,
        "crit_f1": crit_f1,
        "hazard_auprc": hazard_auprc,
    }


# =========================================================
# Main — single stage, all heads train together
# =========================================================
def main():
    print("=" * 72)
    print(f"V11.0 Single-Stage Training | Device = {device}")
    print("=" * 72)

    train_dataset = load_sharded_datasets("train")
    val_dataset = load_sharded_datasets("val")
    if train_dataset is None:
        print("No training data found. Run build_dataset_factory.py first.")
        return

    BATCH_SIZE = 32  # TemporalConv 轻量化后可加大批次
    EPOCHS = 25  # 数据更均衡后收敛更快
    HAZARD_SMOOTH_EPS = 0.1
    FOCAL_GAMMA = 2.0

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    # ---- 临床固定权重 ----
    # Class-Balanced 公式在样本量 >1000 时 β^n≈0，所有大类权重相同，无法区分
    # 改用临床动机权重：重罚 VT(4x)/VF(6x)，适度提升 PVC(3x)/SVT(2x)
    rhy_counts, cri_counts = compute_class_counts_from_dataset(train_dataset)
    print(f"Rhythm      counts: {rhy_counts.int().tolist()}")
    print(f"Criticality counts: {cri_counts.int().tolist()}")
    rhy_weights = torch.tensor([0.5, 3.0, 1.0, 2.0], device=device)
    cri_weights = torch.tensor([0.3, 2.0, 4.0, 6.0], device=device)
    print(f"  → weights (clinical fixed): R={rhy_weights.tolist()}  C={cri_weights.tolist()}")

    # ---- Model ----
    model = HierarchicalHazardNet().to(device)
    backbone_path = "models/ptbxl_backbone.pth"
    if os.path.exists(backbone_path):
        print("Loading PTBXL backbone...")
        backbone = torch.load(backbone_path, map_location=device, weights_only=True)
        model.window_encoder.load_state_dict(backbone)
        print("Backbone loaded.")

    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda")
    rhy_loss_fn = ClassBalancedFocalLoss(rhy_weights, gamma=FOCAL_GAMMA)
    step_counter = [0]
    os.makedirs("models", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup=1500, total=total_steps),
    )

    best_clinical_score = -1.0
    save_path = "models/v10_master_best.pth"

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            epoch, EPOCHS, model, ema, train_loader, optimizer,
            scheduler, scaler, rhy_loss_fn, cri_weights,
            HAZARD_SMOOTH_EPS, step_counter,
        )
        val_metrics = validate(ema.shadow, val_loader, rhy_loss_fn, cri_weights, HAZARD_SMOOTH_EPS)

        epoch_time = time.time() - epoch_start
        temp_val = model.heads.hazard_temperature.item()
        clinical_score = 0.3 * val_metrics["rhythm_acc"] + 0.3 * val_metrics["crit_f1"] + 0.4 * val_metrics["hazard_auprc"]

        print(f"\nEpoch [{epoch:02d}/{EPOCHS}] | {epoch_time / 60:.1f}m")
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Rhythm Acc: {val_metrics['rhythm_acc']:.4f}  |  Crit F1: {val_metrics['crit_f1']:.4f}  |  Hazard AUPRC: {val_metrics['hazard_auprc']:.4f}")
        print(f"  Temp: {temp_val:.3f}  |  Clinical Score: {clinical_score:.4f}")

        if clinical_score > best_clinical_score:
            best_clinical_score = clinical_score
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

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Best clinical score: {best_clinical_score:.4f}  ->  {save_path}")


if __name__ == "__main__":
    main()
