"""
validation_multiclass.py — V6 patient-isolated clinical validation

Evaluates:
  - Current state classification (6-class)
  - Future tendency prediction (soft distribution -> argmax)
  - Per-class P/R/F1/AUC/AP
  - Confusion matrix + CAM examples
"""
import os, glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve,
)
from dl_model import ArrhythmiaWarningNet

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

CLASS_NAMES = [
    "Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"
]


def load_sharded_datasets(split_prefix="val"):
    shard_paths = glob.glob(f"dataset/{split_prefix}_shard_*.pt")
    datasets = []
    names = [os.path.basename(p) for p in shard_paths]
    print(f"  Found {len(shard_paths)} shards: {names}")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        x_rr = data.get("X_rr", torch.zeros(len(data["X"]), 39, 9, dtype=torch.float16))
        y_cur = data.get("Y_cur", data.get("Y", torch.zeros(len(data["X"]), dtype=torch.long)))
        y_fut = data.get("Y_fut", torch.zeros(len(data["X"]), 6, dtype=torch.float16))
        datasets.append(TensorDataset(data["X"], x_rr, y_cur, y_fut))
    return ConcatDataset(datasets)


def plot_cm(y_true, y_pred, names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main():
    print("=" * 60)
    print("V6 Dual-Head Clinical Validation (Patient-Isolated)")
    print("=" * 60)

    val_dataset = load_sharded_datasets("val")
    if val_dataset is None or len(val_dataset) == 0:
        print("[ERROR] No validation data found")
        return
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Val samples: {len(val_dataset)}")

    model = ArrhythmiaWarningNet().to(device)
    ckpt_path = "models/arrhythmia_warning_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", {})), strict=False)
    model.eval()

    all_probs_cur, all_preds_cur, all_targets_cur = [], [], []
    all_probs_fut, all_preds_fut, all_targets_fut = [], [], []

    for bx, bx_rr, y_cur, y_fut in val_loader:
        if not torch.isfinite(bx).all():
            continue
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        y_fut = y_fut.float()

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)

        all_probs_cur.extend(out["probs_cur"].cpu().numpy())
        all_preds_cur.extend(out["probs_cur"].argmax(dim=-1).cpu().numpy())
        all_targets_cur.extend(y_cur.cpu().numpy())

        all_probs_fut.extend(out["probs_fut"].cpu().numpy())
        all_preds_fut.extend(out["probs_fut"].argmax(dim=-1).cpu().numpy())
        all_targets_fut.extend(y_fut.argmax(dim=-1).cpu().numpy())

    pc = np.array(all_preds_cur); tc = np.array(all_targets_cur); prc = np.array(all_probs_cur)
    pf = np.array(all_preds_fut); tf = np.array(all_targets_fut); prf = np.array(all_probs_fut)

    # ---- Current Head ----
    print("\n" + "=" * 50)
    print("=== CURRENT STATE HEAD ===")
    print("=" * 50)
    print(f"\nTotal: {len(tc)}")
    for i, n in enumerate(CLASS_NAMES):
        c = np.sum(tc == i)
        print(f"  {n}: {c} ({c/len(tc)*100:.1f}%)")

    print("\nConfusion Matrix (Current):")
    plot_cm(tc, pc, CLASS_NAMES, "Current State CM", "results/cm_cur_v6.png")
    print(classification_report(tc, pc, target_names=CLASS_NAMES,
                                labels=list(range(6)), zero_division=0, digits=4))

    print("Current AUROC:")
    for i, n in enumerate(CLASS_NAMES):
        yb = (tc == i).astype(int)
        if yb.sum() == 0:
            print(f"  {n}: no samples")
            continue
        try:
            auc = roc_auc_score(yb, prc[:, i])
            print(f"  {n}: AUC={auc:.4f}")
        except:
            pass

    # ---- Future Head ----
    print("\n" + "=" * 50)
    print("=== FUTURE TENDENCY HEAD ===")
    print("=" * 50)
    print(f"\nTotal: {len(tf)}")
    for i, n in enumerate(CLASS_NAMES):
        c = np.sum(tf == i)
        print(f"  {n}: {c} ({c/len(tf)*100:.1f}%)")

    print("\nConfusion Matrix (Future):")
    plot_cm(tf, pf, CLASS_NAMES, "Future Tendency CM", "results/cm_fut_v6.png")
    print(classification_report(tf, pf, target_names=CLASS_NAMES,
                                labels=list(range(6)), zero_division=0, digits=4))

    print("Future AUROC:")
    for i, n in enumerate(CLASS_NAMES):
        yb = (tf == i).astype(int)
        if yb.sum() == 0:
            print(f"  {n}: no samples")
            continue
        try:
            auc = roc_auc_score(yb, prf[:, i])
            print(f"  {n}: AUC={auc:.4f}")
        except:
            pass

    # ---- VT/VF recall ----
    print("\n=== High-risk Recall ===")
    for head_name, preds, targets in [("Current", pc, tc), ("Future", pf, tf)]:
        for cls_idx, cls_name in [(4, "VT"), (3, "VF")]:
            m = targets == cls_idx
            if m.sum() > 0:
                rec = (preds[m] == cls_idx).mean()
                print(f"  {head_name} {cls_name}: Recall={rec:.4f} ({m.sum()} samples)")

    # ---- CAM ----
    print("\n=== CAM Examples ===")
    try:
        bx, bx_rr, _, _ = next(iter(val_loader))
        bx = bx[:4].to(device, dtype=torch.float32)
        bx_rr = bx_rr[:4].to(device, dtype=torch.float32)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)
            cams = out["cam"].cpu().numpy()

        fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        for i in range(4):
            axes[i].fill_between(np.arange(len(cams[i])), cams[i], color='red', alpha=0.7)
            axes[i].set_ylabel(f"Sample {i+1}")
            axes[i].set_ylim(0, 1)
        axes[-1].set_xlabel("Window index")
        fig.suptitle("1D-CAM Attention")
        fig.tight_layout()
        fig.savefig("results/cam_v6.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  CAM failed: {e}")

    print("\nDone. Results in results/")
    for f in sorted(glob.glob("results/*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
