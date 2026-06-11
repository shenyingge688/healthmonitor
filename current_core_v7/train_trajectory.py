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
import os, gc, glob, math, time, csv, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from dl_model import ArrhythmiaWarningNet

torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

NUM_CLASSES = 6
CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
# Arrhythmia classes we actually evaluate/optimize for (VF/AT currently data-sparse)
RARE_CLASSES = [1, 2, 4]   # PVC, AFib, VT


def set_seed(seed):
    """Seed all RNGs for run-to-run reproducibility (sampler uses the global torch RNG)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sharded_datasets(split_prefix, dataset_dir="dataset"):
    shard_paths = sorted(glob.glob(os.path.join(dataset_dir, f"{split_prefix}_shard_*.pt")))
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
    """Pull Y_cur, Y_fut, and T_weight for class weights, sampler, and risk heads."""
    y_curs = np.empty(len(dataset), dtype=np.int64)
    y_futs = np.empty((len(dataset), NUM_CLASSES), dtype=np.float32)
    t_ws = np.empty(len(dataset), dtype=np.float32)
    for i in range(len(dataset)):
        s = dataset[i]
        y_curs[i] = int(s[2])
        y_futs[i] = s[3].float().numpy()
        t_ws[i] = float(s[4])
    return y_curs, y_futs, t_ws


def compute_class_weights(y_curs):
    """Inverse-sqrt-frequency class weights, clamped — used in BOTH head losses."""
    counts = np.bincount(y_curs, minlength=NUM_CLASSES).astype(np.float64)
    n_total = counts.sum()
    w = np.sqrt(n_total / (NUM_CLASSES * np.maximum(counts, 1.0)))
    w = np.clip(w, 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def compute_future_class_weights(y_futs):
    """Inverse-sqrt-frequency class weights from soft future label mass."""
    mass = np.asarray(y_futs, dtype=np.float64).sum(axis=0)
    n_total = mass.sum()
    w = np.sqrt(n_total / (NUM_CLASSES * np.maximum(mass, 1.0)))
    w = np.clip(w, 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32, device=device), mass


def build_sampler_weights(y_curs, t_ws, y_futs=None, sampler_mode="current"):
    """Per-sample sampling weight = inverse-sqrt class freq * transition weight."""
    counts = np.bincount(y_curs, minlength=NUM_CLASSES).astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    sw = inv[y_curs] * np.maximum(t_ws, 1.0)
    if sampler_mode == "current":
        pass
    elif sampler_mode == "future_event":
        if y_futs is None:
            raise ValueError("future_event sampler requires y_futs")
        y_futs = np.asarray(y_futs, dtype=np.float64)
        fut_major = y_futs.argmax(axis=1)
        fut_counts = np.bincount(fut_major, minlength=NUM_CLASSES).astype(np.float64)
        fut_inv = 1.0 / np.sqrt(np.maximum(fut_counts, 1.0))
        fut_inv = fut_inv / max(fut_inv[0], 1e-12)
        arr_mass = np.clip(y_futs[:, 1:].sum(axis=1), 0.0, 1.0)
        high_mass = np.clip(y_futs[:, [3, 4]].sum(axis=1), 0.0, 1.0)
        transition_boost = np.sqrt(np.maximum(t_ws, 1.0))
        event_boost = 1.0 + 0.75 * arr_mass + 1.0 * high_mass
        sw = sw * np.clip(fut_inv[fut_major], 0.75, 2.0) * event_boost * transition_boost
    else:
        raise ValueError(f"unsupported sampler_mode: {sampler_mode}")
    sw = sw / max(float(np.mean(sw)), 1e-12)
    sw = np.clip(sw, 0.10, 12.0)
    return torch.tensor(sw, dtype=torch.double)


def summarize_sampler_weights(weights, y_curs, y_futs, t_ws, sampler_mode):
    w = np.asarray(weights, dtype=np.float64)
    y_futs = np.asarray(y_futs, dtype=np.float64)
    arr = y_futs[:, 1:].sum(axis=1) > 0.01
    high = y_futs[:, [3, 4]].sum(axis=1) > 0.01
    trans = np.asarray(t_ws) > 1.0
    rows = {
        "mode": sampler_mode,
        "mean": float(w.mean()),
        "p50": float(np.percentile(w, 50)),
        "p95": float(np.percentile(w, 95)),
        "p99": float(np.percentile(w, 99)),
        "max": float(w.max()),
        "arrhythmia_mean": float(w[arr].mean()) if arr.any() else 0.0,
        "normal_future_mean": float(w[~arr].mean()) if (~arr).any() else 0.0,
        "high_risk_mean": float(w[high].mean()) if high.any() else 0.0,
        "transition_mean": float(w[trans].mean()) if trans.any() else 0.0,
    }
    print(f"Sampler mode: {sampler_mode}")
    print(
        "Sampler weights: "
        f"mean={rows['mean']:.3f} p50={rows['p50']:.3f} "
        f"p95={rows['p95']:.3f} p99={rows['p99']:.3f} max={rows['max']:.3f}"
    )
    print(
        "Sampler groups: "
        f"future_arr={rows['arrhythmia_mean']:.3f} "
        f"future_normal={rows['normal_future_mean']:.3f} "
        f"future_high={rows['high_risk_mean']:.3f} "
        f"transition={rows['transition_mean']:.3f}"
    )
    return rows


def binary_pos_weight(soft_targets, max_weight=20.0):
    """Stable pos_weight for soft binary BCE targets."""
    pos = float(np.clip(np.asarray(soft_targets, dtype=np.float64).sum(), 1.0, None))
    neg = float(max(len(soft_targets) - pos, 1.0))
    return min(neg / pos, float(max_weight))


def expected_calibration_error(probs, targets, n_bins=10):
    """Top-label multiclass ECE for checkpoint tracking only."""
    if len(targets) == 0:
        return 0.0
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = ((conf >= lo) & (conf <= hi)) if i == 0 else ((conf > lo) & (conf <= hi))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return float(ece)


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


def v6_loss(logits_cur, logits_fut, y_cur, y_fut, t_weight, class_w_cur,
            class_w_fut=None,
            risk_loss_weight=0.0, arr_pos_weight=None, high_pos_weight=None,
            risk_high_weight=1.5):
    """
      - Current head: class-weighted focal (gamma=2) on smoothed one-hot
      - Future head:  class-weighted soft cross-entropy on the future distribution
      - Transition weight scales the FUTURE term only
    """
    cw_cur = class_w_cur.view(1, -1)
    cw_fut = (class_w_fut if class_w_fut is not None else class_w_cur).view(1, -1)

    # Current head: class-weighted focal CE
    y_cur_oh = F.one_hot(y_cur, NUM_CLASSES).float()
    y_cur_sm = y_cur_oh * (1.0 - SMOOTH) + SMOOTH / NUM_CLASSES
    log_probs_cur = F.log_softmax(logits_cur.float(), dim=-1)
    ce_cur = -(cw_cur * y_cur_sm * log_probs_cur).sum(dim=-1)
    pt = torch.exp(-torch.clamp(ce_cur, max=20.0))
    loss_cur = ((1 - pt) ** 2.0) * ce_cur

    # Future head: class-weighted soft cross-entropy
    log_probs_fut = F.log_softmax(logits_fut.float(), dim=-1)
    loss_fut = -(cw_fut * y_fut.float() * log_probs_fut).sum(dim=-1)

    # Transition weight on the future term only
    loss = (loss_cur + 0.8 * t_weight.float() * loss_fut).mean()

    if risk_loss_weight > 0:
        # Binary risks derived from the same 6-class future logits. No model
        # structure changes are needed, so old checkpoints and serving stay compatible.
        arr_logit = torch.logsumexp(logits_fut[:, 1:].float(), dim=-1) - logits_fut[:, 0].float()
        high_logit = (
            torch.logsumexp(logits_fut[:, [3, 4]].float(), dim=-1) -
            torch.logsumexp(logits_fut[:, [0, 1, 2, 5]].float(), dim=-1)
        )
        arr_target = y_fut[:, 1:].float().sum(dim=-1).clamp(0.0, 1.0)
        high_target = y_fut[:, [3, 4]].float().sum(dim=-1).clamp(0.0, 1.0)
        arr_pw = arr_pos_weight if arr_pos_weight is not None else torch.tensor(1.0, device=logits_fut.device)
        high_pw = high_pos_weight if high_pos_weight is not None else torch.tensor(1.0, device=logits_fut.device)
        arr_loss = F.binary_cross_entropy_with_logits(
            arr_logit, arr_target, pos_weight=arr_pw, reduction="mean"
        )
        high_loss = F.binary_cross_entropy_with_logits(
            high_logit, high_target, pos_weight=high_pw, reduction="mean"
        )
        loss = loss + float(risk_loss_weight) * (arr_loss + float(risk_high_weight) * high_loss)
    return loss, loss_cur.mean().detach(), loss_fut.mean().detach()


# =========================================================
# Training
# =========================================================

def train_epoch(epoch, total_epochs, model, ema, loader, opt, sched, scaler, step_ctr,
                class_w_cur, class_w_fut=None, risk_cfg=None):
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
            loss, lc, lf = v6_loss(
                out["logits_cur"], out["logits_fut"], y_cur, y_fut, t_w,
                class_w_cur, class_w_fut,
                **(risk_cfg or {}),
            )

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
def validate(model, loader, class_w_cur, class_w_fut=None, risk_cfg=None):
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
            loss, _, _ = v6_loss(
                out["logits_cur"], out["logits_fut"], y_cur, y_fut, t_w,
                class_w_cur, class_w_fut,
                **(risk_cfg or {}),
            )
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
    ece_fut = expected_calibration_error(prob_fut, at_fut)

    arr_target = (at_fut != 0).astype(int)
    arr_score = 1.0 - prob_fut[:, 0]
    if 0 < arr_target.sum() < len(arr_target):
        arr_auc = float(roc_auc_score(arr_target, arr_score))
        arr_ap = float(average_precision_score(arr_target, arr_score))
    else:
        arr_auc = 0.0
        arr_ap = 0.0

    high_target = np.isin(at_fut, [3, 4]).astype(int)
    high_score = prob_fut[:, 3] + prob_fut[:, 4]
    if 0 < high_target.sum() < len(high_target):
        high_auc = float(roc_auc_score(high_target, high_score))
        high_ap = float(average_precision_score(high_target, high_score))
    else:
        high_auc = 0.0
        high_ap = 0.0

    trans = at_fut != at_cur
    if trans.sum() > 0:
        transition_acc = accuracy_score(at_fut[trans], ap_fut[trans])
        transition_recall = recall_score(
            at_fut[trans], ap_fut[trans],
            labels=list(range(NUM_CLASSES)), average="macro", zero_division=0
        )
    else:
        transition_acc = 0.0
        transition_recall = 0.0

    return {"loss": tl / max(n, 1), "acc_cur": acc_cur, "f1_cur": f1_cur,
            "acc_fut": acc_fut, "f1_fut": f1_fut, "recall_rare": recall_rare,
            "auroc_fut": auroc_fut, "ece_fut": ece_fut,
            "arrhythmia_auroc": arr_auc, "arrhythmia_ap": arr_ap,
            "high_risk_auroc": high_auc, "high_risk_ap": high_ap,
            "transition_acc": float(transition_acc),
            "transition_recall": float(transition_recall),
            "transition_n": int(trans.sum())}


def save_checkpoint(path, model, ema, epoch, score_name, score_value, metrics,
                    class_w_cur, class_w_fut=None):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "epoch": epoch,
        "score_name": score_name,
        "score_value": float(score_value),
        "metrics": metrics,
        "class_weights": class_w_cur.cpu(),
        "class_weights_current": class_w_cur.cpu(),
        "class_weights_future": (class_w_fut if class_w_fut is not None else class_w_cur).cpu(),
    }, path)


def update_metric_checkpoints(trackers, model, ema, epoch, metrics, class_w_cur, class_w_fut=None):
    saved = []
    for name, cfg in trackers.items():
        value = cfg["value_fn"](metrics)
        improved = value < cfg["best"] - 1e-4 if cfg["mode"] == "min" else value > cfg["best"] + 1e-4
        if improved:
            cfg["best"] = float(value)
            cfg["epoch"] = epoch
            save_checkpoint(cfg["path"], model, ema, epoch, name, value, metrics,
                            class_w_cur, class_w_fut)
            saved.append((name, value, cfg["path"]))
    return saved


def append_history(path, epoch, train_loss, score, metrics):
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "composite_score": score,
        **metrics,
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(seed=None, dataset_dir="dataset", output_root="models", epochs=60, batch_size=24,
         backbone_path="models/ptbxl_backbone.pth", risk_loss_weight=0.0,
         risk_high_weight=1.5, selection_mode="v6", future_weight_mode="cur",
         encoder_tune="frozen", encoder_tail_lr=1e-4, sampler_mode="current"):
    print("=" * 60)
    print(f"V6.1 Minority-Aware Dual-Head Training | Device = {device}")
    if seed is not None:
        set_seed(seed)
        out_dir = f"{output_root}/seeds/seed{seed}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"SEED = {seed}  |  output dir = {out_dir}")
    else:
        out_dir = output_root
    print(f"Dataset dir = {dataset_dir}")
    print("=" * 60)

    train_ds = load_sharded_datasets("train", dataset_dir=dataset_dir)
    val_ds = load_sharded_datasets("val", dataset_dir=dataset_dir)
    if train_ds is None or val_ds is None:
        print("No data. Run build_dataset_factory.py first.")
        return

    B = batch_size
    EPOCHS = epochs
    PATIENCE = None  # fixed-run validation phase: run all 60 epochs unless interrupted

    # Class weights + sampler from train labels
    y_curs, y_futs, t_ws = collect_labels(train_ds)
    class_w_cur = compute_class_weights(y_curs)
    class_w_fut = class_w_cur
    counts = np.bincount(y_curs, minlength=NUM_CLASSES)
    print(f"Train Y_cur counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(f"Current-head class weights: {[round(float(x),2) for x in class_w_cur.tolist()]}")
    if future_weight_mode == "soft":
        class_w_fut, fut_mass = compute_future_class_weights(y_futs)
        fut_mass_print = [round(float(x), 2) for x in fut_mass.tolist()]
        print(f"Train Y_fut soft mass: {dict(zip(CLASS_NAMES, fut_mass_print))}")
    elif future_weight_mode != "cur":
        raise ValueError(f"unsupported future_weight_mode: {future_weight_mode}")
    print(f"Future-head class weights ({future_weight_mode}): {[round(float(x),2) for x in class_w_fut.tolist()]}")
    arr_pos_w = binary_pos_weight(y_futs[:, 1:].sum(axis=1))
    high_pos_w = binary_pos_weight(y_futs[:, [3, 4]].sum(axis=1))
    risk_cfg = None
    if risk_loss_weight > 0:
        risk_cfg = {
            "risk_loss_weight": float(risk_loss_weight),
            "risk_high_weight": float(risk_high_weight),
            "arr_pos_weight": torch.tensor(arr_pos_w, dtype=torch.float32, device=device),
            "high_pos_weight": torch.tensor(high_pos_w, dtype=torch.float32, device=device),
        }
        print(
            f"Risk-aware auxiliary loss ON: weight={risk_loss_weight} "
            f"high_weight={risk_high_weight} arr_pos_w={arr_pos_w:.2f} high_pos_w={high_pos_w:.2f}"
        )
    print(f"Selection mode: {selection_mode}")
    sampler_w = build_sampler_weights(
        y_curs, t_ws, y_futs=y_futs, sampler_mode=sampler_mode
    )
    sampler_summary = summarize_sampler_weights(
        sampler_w.numpy(), y_curs, y_futs, t_ws, sampler_mode=sampler_mode
    )
    sampler = WeightedRandomSampler(sampler_w, num_samples=len(train_ds), replacement=True)

    tl = DataLoader(train_ds, batch_size=B, sampler=sampler, num_workers=0,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=B, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = ArrhythmiaWarningNet().to(device)
    if os.path.exists(backbone_path):
        print("Loading PTB-XL backbone...")
        bb = torch.load(backbone_path, map_location=device, weights_only=True)
        model.window_encoder.load_state_dict(bb, strict=False)
        print("  Loaded.")
    if encoder_tune == "tail":
        model.unfreeze_encoder_tail()
        print(f"Encoder tuning: tail (stage4 + pool_proj), lr={encoder_tail_lr:g}")
    elif encoder_tune == "frozen":
        model.freeze_encoder()
        print("Encoder tuning: frozen")
    else:
        raise ValueError(f"unsupported encoder_tune: {encoder_tune}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")

    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    step_ctr = [0]
    os.makedirs(output_root, exist_ok=True)

    if encoder_tune == "tail":
        encoder_param_ids = {
            id(p)
            for module in (model.window_encoder.stage4, model.window_encoder.pool_proj)
            for p in module.parameters()
            if p.requires_grad
        }
        encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) in encoder_param_ids]
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
        opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": 5e-4},
                {"params": encoder_params, "lr": float(encoder_tail_lr)},
            ],
            weight_decay=3e-4,
        )
    else:
        trainable_p = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_p, lr=5e-4, weight_decay=3e-4)
    total_steps = len(tl) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_schedule(s, warmup=600, total=total_steps))

    best_score, best_ep, no_imp = -1.0, 0, 0
    save_p = f"{out_dir}/arrhythmia_warning_best.pth"
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    history_p = f"{out_dir}/training_history_v6_1_{run_stamp}.csv"
    trackers = {
        "composite": {
            "path": save_p,
            "mode": "max",
            "best": -float("inf"),
            "epoch": 0,
            "value_fn": lambda m: m["composite_score"],
        },
        "future_macro_f1": {
            "path": f"{out_dir}/arrhythmia_warning_best_macro_f1.pth",
            "mode": "max",
            "best": -float("inf"),
            "epoch": 0,
            "value_fn": lambda m: m["f1_fut"],
        },
        "transition_recall": {
            "path": f"{out_dir}/arrhythmia_warning_best_transition_recall.pth",
            "mode": "max",
            "best": -float("inf"),
            "epoch": 0,
            "value_fn": lambda m: m["transition_recall"],
        },
        "future_auroc": {
            "path": f"{out_dir}/arrhythmia_warning_best_auroc.pth",
            "mode": "max",
            "best": -float("inf"),
            "epoch": 0,
            "value_fn": lambda m: m["auroc_fut"],
        },
        "lowest_ece": {
            "path": f"{out_dir}/arrhythmia_warning_lowest_ece.pth",
            "mode": "min",
            "best": float("inf"),
            "epoch": 0,
            "value_fn": lambda m: m["ece_fut"],
        },
    }
    epoch_metrics = []  # per-epoch validation dicts for cross-epoch aggregation

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            ep, EPOCHS, model, ema, tl, opt, sched, scaler, step_ctr,
            class_w_cur, class_w_fut, risk_cfg=risk_cfg,
        )
        vm = validate(ema.shadow, vl, class_w_cur, class_w_fut, risk_cfg=risk_cfg)
        dt = time.time() - t0

        # Minority-aware selection (NO Normal-accuracy reward)
        if selection_mode == "risk_aware":
            score = (
                0.20 * vm["f1_fut"] +
                0.20 * vm["recall_rare"] +
                0.20 * vm["auroc_fut"] +
                0.20 * vm["arrhythmia_auroc"] +
                0.20 * vm["high_risk_auroc"]
            )
        else:
            score = (0.30 * vm["f1_cur"] + 0.30 * vm["f1_fut"]
                     + 0.20 * vm["auroc_fut"] + 0.20 * vm["recall_rare"])
        vm["composite_score"] = float(score)
        vm["epoch"] = ep
        epoch_metrics.append(dict(vm))
        print(f"\nEpoch [{ep:02d}/{EPOCHS}] | {dt/60:.1f}m | Loss {tr_loss:.4f}")
        print(f"  Cur: acc {vm['acc_cur']:.3f} f1 {vm['f1_cur']:.3f}  |  "
              f"Fut: acc {vm['acc_fut']:.3f} f1 {vm['f1_fut']:.3f} auroc {vm['auroc_fut']:.3f} "
              f"rareRec {vm['recall_rare']:.3f} ece {vm['ece_fut']:.3f}")
        print(f"  Risk: arrAUROC {vm['arrhythmia_auroc']:.3f} highAUROC {vm['high_risk_auroc']:.3f} "
              f"highAP {vm['high_risk_ap']:.3f}")
        print(f"  Transition: n={vm['transition_n']} acc {vm['transition_acc']:.3f} "
              f"recall {vm['transition_recall']:.3f}")
        print(f"  Score: {score:.4f}")
        append_history(history_p, ep, tr_loss, score, vm)

        if score > best_score + 1e-4:
            best_score, best_ep, no_imp = score, ep, 0
        else:
            no_imp += 1

        saved = update_metric_checkpoints(trackers, model, ema, ep, vm, class_w_cur, class_w_fut)
        if saved:
            for name, value, path in saved:
                print(f"  [save:{name}] {value:.4f} -> {path}")
        else:
            wait_msg = f"  [wait] composite no-improve={no_imp}"
            if PATIENCE is not None:
                wait_msg += f"/{PATIENCE}"
            else:
                wait_msg += " (full-run mode)"
            print(wait_msg)

        if PATIENCE is not None and no_imp >= PATIENCE:
            print(f"\nEarly stop @ ep {ep} (best: {best_ep}, score={best_score:.4f})")
            break
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(
        f"{out_dir}/arrhythmia_warning_final.pth", model, ema, ep, "final_epoch",
        score, vm, class_w_cur, class_w_fut,
    )
    print(f"\nDone. composite best={best_score:.4f} @ ep {best_ep} -> {save_p}")
    print("Checkpoint summary:")
    for name, cfg in trackers.items():
        print(f"  {name:18s} epoch={cfg['epoch']:02d} value={cfg['best']:.4f} -> {cfg['path']}")
    print("  final_epoch        " + f"epoch={ep:02d} -> {out_dir}/arrhythmia_warning_final.pth")
    print(f"History -> {history_p}")

    # Per-seed summary: final-epoch, best-composite-epoch, and per-metric peaks across the run
    def peak(key, mode="max"):
        vals = [(m[key], m["epoch"]) for m in epoch_metrics]
        return (max if mode == "max" else min)(vals, key=lambda x: x[0])

    best_comp = max(epoch_metrics, key=lambda m: m["composite_score"])
    summary = {
        "seed": seed,
        "epochs_run": ep,
        "final_epoch": epoch_metrics[-1],
        "best_composite_epoch": best_comp,
        "peaks": {
            "f1_cur": peak("f1_cur"), "f1_fut": peak("f1_fut"),
            "auroc_fut": peak("auroc_fut"), "recall_rare": peak("recall_rare"),
            "transition_recall": peak("transition_recall"),
            "arrhythmia_auroc": peak("arrhythmia_auroc"),
            "high_risk_auroc": peak("high_risk_auroc"),
            "high_risk_ap": peak("high_risk_ap"),
            "ece_fut_min": peak("ece_fut", "min"),
        },
        "risk_loss_weight": risk_loss_weight,
        "risk_high_weight": risk_high_weight,
        "selection_mode": selection_mode,
        "future_weight_mode": future_weight_mode,
        "encoder_tune": encoder_tune,
        "encoder_tail_lr": encoder_tail_lr,
        "sampler_mode": sampler_mode,
        "sampler_summary": sampler_summary,
        "history_csv": history_p,
    }
    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_dir}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="fixed RNG seed; tags outputs under models/seeds/seedN")
    ap.add_argument("--dataset-dir", default="dataset", help="directory containing train/val shard tensors")
    ap.add_argument("--output-root", default="models", help="root directory for checkpoints and summaries")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--backbone", default="models/ptbxl_backbone.pth")
    ap.add_argument("--risk-loss-weight", type=float, default=0.0)
    ap.add_argument("--risk-high-weight", type=float, default=1.5)
    ap.add_argument("--selection-mode", choices=["v6", "risk_aware"], default="v6")
    ap.add_argument("--future-weight-mode", choices=["cur", "soft"], default="cur")
    ap.add_argument("--encoder-tune", choices=["frozen", "tail"], default="frozen")
    ap.add_argument("--encoder-tail-lr", type=float, default=1e-4)
    ap.add_argument("--sampler-mode", choices=["current", "future_event"], default="current")
    args = ap.parse_args()
    main(seed=args.seed, dataset_dir=args.dataset_dir, output_root=args.output_root,
         epochs=args.epochs, batch_size=args.batch_size, backbone_path=args.backbone,
         risk_loss_weight=args.risk_loss_weight, risk_high_weight=args.risk_high_weight,
         selection_mode=args.selection_mode, future_weight_mode=args.future_weight_mode,
         encoder_tune=args.encoder_tune, encoder_tail_lr=args.encoder_tail_lr,
         sampler_mode=args.sampler_mode)
