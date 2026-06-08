"""
Script: train_trajectory.py
ECG early-warning training engine V6.1 (minority-aware)

Fixes over V6:
  - Inverse-frequency CLASS WEIGHTS on BOTH heads (was: future head unweighted -> Normal dominated -> PVC/VT collapse)
  - WeightedRandomSampler for rare-class / transition oversampling (was: imported but unused)
  - Transition weight applied to the FUTURE term only (was: scaled current head too)
  - Minority-aware model selection: future macro-F1 + rare-class recall + macro-AUROC
    (was: 0.4*acc_fut micro-accuracy -> rewarded the all-Normal lazy predictor)
  - Frozen encoder set to eval() so its BatchNorm stats stop drifting
"""
import os, gc, glob, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from dl_model import ArrhythmiaWarningNet

torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

NUM_CLASSES = 6
CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
# Arrhythmia classes we actually evaluate/optimize for (VF/AT currently data-sparse)
RARE_CLASSES = [1, 2, 4]   # PVC, AFib, VT


def load_sharded_datasets(split_prefix):
    shard_paths = sorted(glob.glob(f"dataset/{split_prefix}_shard_*.pt"))
    if not shard_paths:
        return None
    datasets = []
    print(f"Loading {split_prefix} shards: {len(shard_paths)} files")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        x_rr = data.get("X_rr", torch.zeros(len(data["X"]), 39, 9, dtype=torch.float16))
        y_cur = data.get("Y_cur", data.get("Y", torch.zeros(len(data["X"]), dtype=torch.long)))
        y_fut = data.get("Y_fut", torch.zeros(len(data["X"]), 6, dtype=torch.float16))
        t_w = data.get("T_weight", torch.ones(len(data["X"]), dtype=torch.float16))
        datasets.append(TensorDataset(data["X"], x_rr, y_cur, y_fut, t_w))
    return ConcatDataset(datasets) if datasets else None


def collect_labels(dataset):
    """Pull Y_cur and T_weight for the whole dataset (for class weights + sampler)."""
    y_curs = np.empty(len(dataset), dtype=np.int64)
    t_ws = np.empty(len(dataset), dtype=np.float32)
    for i in range(len(dataset)):
        s = dataset[i]
        y_curs[i] = int(s[2])
        t_ws[i] = float(s[4])
    return y_curs, t_ws


def compute_class_weights(y_curs):
    """Inverse-sqrt-frequency class weights, clamped — used in BOTH head losses."""
    counts = np.bincount(y_curs, minlength=NUM_CLASSES).astype(np.float64)
    n_total = counts.sum()
    w = np.sqrt(n_total / (NUM_CLASSES * np.maximum(counts, 1.0)))
    w = np.clip(w, 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def build_sampler_weights(y_curs, t_ws):
    """Per-sample sampling weight = inverse-sqrt class freq * transition weight."""
    counts = np.bincount(y_curs, minlength=NUM_CLASSES).astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    sw = inv[y_curs] * np.maximum(t_ws, 1.0)
    return torch.tensor(sw, dtype=torch.double)


class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = ArrhythmiaWarningNet().to(device)
        self.shadow.load_state_dict(model.state_dict())
        self.shadow.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(p.data)


def lr_schedule(step, warmup=800, total=20000):
    mn = 0.01
    if step < warmup:
        return mn + (1.0 - mn) * (step / warmup)
    p = (step - warmup) / max(total - warmup, 1)
    return mn + 0.5 * (1.0 - mn) * (1 + math.cos(math.pi * p))


# =========================================================
# V6.1: class-weighted dual-head loss
# =========================================================

SMOOTH = 0.06


def v6_loss(logits_cur, logits_fut, y_cur, y_fut, t_weight, class_w):
    """
      - Current head: class-weighted focal (gamma=2) on smoothed one-hot
      - Future head:  class-weighted soft cross-entropy on the future distribution
      - Transition weight scales the FUTURE term only
    """
    cw = class_w.view(1, -1)

    # Current head: class-weighted focal CE
    y_cur_oh = F.one_hot(y_cur, NUM_CLASSES).float()
    y_cur_sm = y_cur_oh * (1.0 - SMOOTH) + SMOOTH / NUM_CLASSES
    log_probs_cur = F.log_softmax(logits_cur.float(), dim=-1)
    ce_cur = -(cw * y_cur_sm * log_probs_cur).sum(dim=-1)
    pt = torch.exp(-torch.clamp(ce_cur, max=20.0))
    loss_cur = ((1 - pt) ** 2.0) * ce_cur

    # Future head: class-weighted soft cross-entropy
    log_probs_fut = F.log_softmax(logits_fut.float(), dim=-1)
    loss_fut = -(cw * y_fut.float() * log_probs_fut).sum(dim=-1)

    # Transition weight on the future term only
    loss = (loss_cur + 0.8 * t_weight.float() * loss_fut).mean()
    return loss, loss_cur.mean().detach(), loss_fut.mean().detach()


# =========================================================
# Training
# =========================================================

def train_epoch(epoch, total_epochs, model, ema, loader, opt, sched, scaler, step_ctr, class_w):
    model.train()
    model.window_encoder.eval()   # keep frozen BN stats fixed
    total_l, n = 0.0, 0
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{total_epochs}]", leave=False)
    for bi, (bx, bx_rr, y_cur, y_fut, t_w) in enumerate(pbar):
        if not torch.isfinite(bx).all():
            continue
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y_cur = y_cur.to(device)
        y_fut = y_fut.to(device)
        t_w = t_w.to(device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)
            loss, lc, lf = v6_loss(out["logits_cur"], out["logits_fut"], y_cur, y_fut, t_w, class_w)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        sb = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= sb:
            sched.step()
            ema.update(model)
            total_l += loss.item()
            n += 1
            step_ctr[0] += 1

        if bi % 200 == 0 and bi > 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        pbar.set_postfix({"L": f"{loss.item():.3f}", "Lf": f"{lf.item():.3f}",
                          "LR": f"{opt.param_groups[0]['lr']:.1e}"})

    return total_l / max(n, 1)


@torch.inference_mode()
def validate(model, loader, class_w):
    model.eval()
    tl, n = 0.0, 0
    ap_cur, at_cur, ap_fut, at_fut, prob_fut = [], [], [], [], []

    for bx, bx_rr, y_cur, y_fut, t_w in loader:
        if not torch.isfinite(bx).all():
            continue
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y_cur = y_cur.to(device)
        y_fut = y_fut.to(device)
        t_w = t_w.to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)
            loss, _, _ = v6_loss(out["logits_cur"], out["logits_fut"], y_cur, y_fut, t_w, class_w)
        tl += loss.item(); n += 1
        ap_cur.extend(out["probs_cur"].argmax(dim=-1).cpu().numpy())
        at_cur.extend(y_cur.cpu().numpy())
        pf = out["probs_fut"].float().cpu().numpy()
        prob_fut.extend(pf)
        ap_fut.extend(pf.argmax(axis=-1))
        at_fut.extend(y_fut.argmax(dim=-1).cpu().numpy())

    at_cur, ap_cur = np.array(at_cur), np.array(ap_cur)
    at_fut, ap_fut = np.array(at_fut), np.array(ap_fut)
    prob_fut = np.array(prob_fut)

    acc_cur = accuracy_score(at_cur, ap_cur) if len(at_cur) else 0.0
    f1_cur = f1_score(at_cur, ap_cur, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0) if len(at_cur) else 0.0
    acc_fut = accuracy_score(at_fut, ap_fut) if len(at_fut) else 0.0
    f1_fut = f1_score(at_fut, ap_fut, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0) if len(at_fut) else 0.0

    # rare-class recall (mean over PVC/AFib/VT that have support)
    rec_per = recall_score(at_fut, ap_fut, labels=RARE_CLASSES, average=None, zero_division=0) if len(at_fut) else np.zeros(len(RARE_CLASSES))
    present = [i for i, c in enumerate(RARE_CLASSES) if (at_fut == c).sum() > 0]
    recall_rare = float(np.mean([rec_per[i] for i in present])) if present else 0.0

    # future macro-AUROC over classes with both pos & neg present
    aucs = []
    for c in range(NUM_CLASSES):
        yb = (at_fut == c).astype(int)
        if 0 < yb.sum() < len(yb):
            try:
                aucs.append(roc_auc_score(yb, prob_fut[:, c]))
            except Exception:
                pass
    auroc_fut = float(np.mean(aucs)) if aucs else 0.0

    return {"loss": tl / max(n, 1), "acc_cur": acc_cur, "f1_cur": f1_cur,
            "acc_fut": acc_fut, "f1_fut": f1_fut, "recall_rare": recall_rare,
            "auroc_fut": auroc_fut}


def main():
    print("=" * 60)
    print(f"V6.1 Minority-Aware Dual-Head Training | Device = {device}")
    print("=" * 60)

    train_ds = load_sharded_datasets("train")
    val_ds = load_sharded_datasets("val")
    if train_ds is None or val_ds is None:
        print("No data. Run build_dataset_factory.py first.")
        return

    B = 24
    EPOCHS = 60
    PATIENCE = 15

    # Class weights + sampler from train labels
    y_curs, t_ws = collect_labels(train_ds)
    class_w = compute_class_weights(y_curs)
    counts = np.bincount(y_curs, minlength=NUM_CLASSES)
    print(f"Train Y_cur counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(f"Class weights: {[round(float(x),2) for x in class_w.tolist()]}")
    sampler = WeightedRandomSampler(build_sampler_weights(y_curs, t_ws),
                                    num_samples=len(train_ds), replacement=True)

    tl = DataLoader(train_ds, batch_size=B, sampler=sampler, num_workers=0,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=B, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = ArrhythmiaWarningNet().to(device)
    backbone_path = "models/ptbxl_backbone.pth"
    if os.path.exists(backbone_path):
        print("Loading PTB-XL backbone...")
        bb = torch.load(backbone_path, map_location=device, weights_only=True)
        model.window_encoder.load_state_dict(bb, strict=False)
        print("  Loaded.")
    model.freeze_encoder()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")

    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    step_ctr = [0]
    os.makedirs("models", exist_ok=True)

    trainable_p = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_p, lr=5e-4, weight_decay=3e-4)
    total_steps = len(tl) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_schedule(s, warmup=600, total=total_steps))

    best_score, best_ep, no_imp = -1.0, 0, 0
    save_p = "models/arrhythmia_warning_best.pth"

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(ep, EPOCHS, model, ema, tl, opt, sched, scaler, step_ctr, class_w)
        vm = validate(ema.shadow, vl, class_w)
        dt = time.time() - t0

        # Minority-aware selection (NO Normal-accuracy reward)
        score = (0.30 * vm["f1_cur"] + 0.30 * vm["f1_fut"]
                 + 0.20 * vm["auroc_fut"] + 0.20 * vm["recall_rare"])
        print(f"\nEpoch [{ep:02d}/{EPOCHS}] | {dt/60:.1f}m | Loss {tr_loss:.4f}")
        print(f"  Cur: acc {vm['acc_cur']:.3f} f1 {vm['f1_cur']:.3f}  |  "
              f"Fut: acc {vm['acc_fut']:.3f} f1 {vm['f1_fut']:.3f} auroc {vm['auroc_fut']:.3f} rareRec {vm['recall_rare']:.3f}")
        print(f"  Score: {score:.4f}")

        if score > best_score + 1e-4:
            best_score, best_ep, no_imp = score, ep, 0
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                       "epoch": ep, "best_score": best_score, "metrics": vm,
                       "class_weights": class_w.cpu()}, save_p)
            print(f"  [save] score={best_score:.4f}")
        else:
            no_imp += 1
            print(f"  [wait] {no_imp}/{PATIENCE}")

        if no_imp >= PATIENCE:
            print(f"\nEarly stop @ ep {ep} (best: {best_ep}, score={best_score:.4f})")
            break
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. best={best_score:.4f} @ ep {best_ep} -> {save_p}")


if __name__ == "__main__":
    main()
