"""
validation_multiclass.py — 全量多层级临床验证 (V11)

评估内容:
  1. Rhythm (4-class)          混淆矩阵 + per-class P/R/F1 + OVR AUC
  2. Criticality (4-class)     混淆矩阵 + VT/VF 召回率 + OVR AUC
  3. Hazard (3-label)          Per-horizon AUPRC/AUC + 校准曲线 + Brier Score
  4. Warning Fusion            effective_risk ROC / AUPRC
  5. 可视化                      保存至 results/
"""
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)

from dl_model import HierarchicalHazardNet

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)


# =========================================================
def load_sharded_datasets(split_prefix="val"):
    shard_paths = glob.glob(f"dataset/v10_*_{split_prefix}_shard_*.pt")
    datasets = []
    names = [__import__('os').path.basename(p) for p in shard_paths]
    print(f"  found {len(shard_paths)} shards: {names}")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        datasets.append(TensorDataset(
            data["X"], data["Y_rhythm"], data["Y_criticality"], data["Y_hazard"],
            data["M_rhythm"], data["M_criticality"], data["M_hazard"],
        ))
    return ConcatDataset(datasets)


def plot_cm(y_true, y_pred, names, title, path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y_true, y_prob, label, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP={ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(label); ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ap


def plot_cal(y_true, y_prob, n_bins=10, title="", path=""):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs = [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() > 0:
            centers.append(y_prob[m].mean())
            freqs.append(y_true[m].mean())
    if not centers: return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(centers, freqs, "o-", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True Frequency")
    ax.set_title(title); ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_score, label, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(label); ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc


# =========================================================
@torch.inference_mode()
def main():
    print("=" * 60)
    print("V11 Multi-Level Clinical Validation")
    print("=" * 60)

    val_dataset = load_sharded_datasets("val")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Val samples: {len(val_dataset)}")

    model = HierarchicalHazardNet().to(device)
    ckpt_path = "models/v10_master_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found"); return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["ema"], strict=False)
    if missing:
        critical = [k for k in missing if "temperature" not in k]
        if critical:
            raise RuntimeError(f"Model mismatch: missing keys {critical}")
    model.eval()
    print(f"Loaded checkpoint (temp={model.heads.hazard_temperature.item():.3f})")

    # Collect
    all_rhy_preds, all_rhy_targets, all_rhy_probs = [], [], []
    all_cri_preds, all_cri_targets, all_cri_probs = [], [], []
    all_haz_probs, all_haz_targets = [], []
    # Warning Fusion 需要 aligned cri+haz，单独收集
    all_fusion_cri_probs, all_fusion_haz_probs, all_fusion_haz_targets = [], [], []

    for bx, y_rhy, y_cri, y_haz, m_rhy, m_cri, m_haz in val_loader:
        if not torch.isfinite(bx).all(): continue
        bx = bx.to(device, dtype=torch.float32)
        y_rhy = y_rhy.to(device); y_cri = y_cri.to(device); y_haz = y_haz.to(device)
        m_rhy = m_rhy.to(device); m_cri = m_cri.to(device); m_haz = m_haz.to(device)

        with torch.amp.autocast("cuda"):
            out = model(bx)["preds"]
            r_logits = out["rhythm_logits"][:, -1]
            c_logits = out["criticality_logits"][:, -1]
            h_probs = out["hazard_probs"][:, -1]

        idx = (m_rhy > 0.5).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            p = F.softmax(r_logits[idx].float(), dim=-1)
            all_rhy_probs.extend(p.cpu().numpy())
            all_rhy_preds.extend(torch.argmax(p, dim=-1).cpu().numpy())
            all_rhy_targets.extend(y_rhy[idx].cpu().numpy())

        idx = (m_cri > 0.5).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            p = F.softmax(c_logits[idx].float(), dim=-1)
            all_cri_probs.extend(p.cpu().numpy())
            all_cri_preds.extend(torch.argmax(p, dim=-1).cpu().numpy())
            all_cri_targets.extend(y_cri[idx].cpu().numpy())

        idx = (m_haz > 0.5).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            all_haz_probs.extend(h_probs[idx].float().cpu().numpy())
            all_haz_targets.extend(y_haz[idx].cpu().numpy())
            # 同时收集 cri_probs 用于 Fusion 对齐
            p = F.softmax(c_logits[idx].float(), dim=-1)
            all_fusion_cri_probs.extend(p.cpu().numpy())
            all_fusion_haz_probs.extend(h_probs[idx].float().cpu().numpy())
            all_fusion_haz_targets.extend(y_haz[idx].cpu().numpy())

    arr_rhy_p = np.array(all_rhy_preds); arr_rhy_t = np.array(all_rhy_targets); arr_rhy_pr = np.array(all_rhy_probs)
    arr_cri_p = np.array(all_cri_preds); arr_cri_t = np.array(all_cri_targets); arr_cri_pr = np.array(all_cri_probs)
    arr_haz_p = np.array(all_haz_probs); arr_haz_t = np.array(all_haz_targets)

    print(f"Rhythm: {len(arr_rhy_t)}  Criticality: {len(arr_cri_t)}  Hazard: {len(arr_haz_t)}")

    # ---- Rhythm ----
    print("\n=== Rhythm ===")
    rhy_n = ["Normal", "PVC", "AFIB", "SVT/AT"]
    rhy_labels = [0, 1, 2, 3]
    plot_cm(arr_rhy_t, arr_rhy_p, rhy_n, "Rhythm Confusion Matrix", "results/cm_rhythm.png", labels=rhy_labels)
    print(classification_report(arr_rhy_t, arr_rhy_p, labels=rhy_labels, target_names=rhy_n, zero_division=0))
    try: print(f"OVR Macro AUC: {roc_auc_score(arr_rhy_t, arr_rhy_pr, multi_class='ovr', average='macro'):.4f}")
    except: pass

    # ---- Criticality ----
    print("\n=== Criticality ===")
    cri_n = ["Safe", "PVC-Load", "VT", "VF"]
    cri_labels = [0, 1, 2, 3]
    plot_cm(arr_cri_t, arr_cri_p, cri_n, "Criticality Confusion Matrix", "results/cm_criticality.png", labels=cri_labels)
    print(classification_report(arr_cri_t, arr_cri_p, labels=cri_labels, target_names=cri_n, zero_division=0))
    try: print(f"OVR Macro AUC: {roc_auc_score(arr_cri_t, arr_cri_pr, multi_class='ovr', average='macro'):.4f}")
    except: pass
    for lbl, name in [(2, "VT"), (3, "VF")]:
        if lbl in arr_cri_t:
            m = arr_cri_t == lbl
            print(f"{name} Recall: {(arr_cri_p[m] == lbl).mean():.4f}")

    # ---- Hazard ----
    print("\n=== Hazard ===")
    for h_idx, h_name in enumerate(["30s", "1m", "5m"]):
        y_t = arr_haz_t[:, h_idx]; y_p = arr_haz_p[:, h_idx]
        if y_t.sum() == 0: print(f"  {h_name}: no positives"); continue
        ap = plot_pr(y_t, y_p, f"Hazard {h_name} PR", f"results/pr_hazard_{h_name}.png")
        try: auc = roc_auc_score(y_t, y_p)
        except: auc = 0.0
        b = brier_score_loss(y_t, y_p)
        print(f"  {h_name}: AUPRC={ap:.4f}  AUC={auc:.4f}  Brier={b:.4f}")
    if arr_haz_t[:, -1].sum() > 0:
        plot_cal(arr_haz_t[:, -1], arr_haz_p[:, -1], title="Hazard 5m Calibration", path="results/cal_hazard_5m.png")

    # ---- Warning Fusion (使用对齐后的 cri + haz 样本) ----
    print("\n=== Warning Fusion ===")
    if len(all_fusion_cri_probs) > 0:
        arr_f_cri = np.array(all_fusion_cri_probs)
        arr_f_haz = np.array(all_fusion_haz_probs)
        arr_f_haz_t = np.array(all_fusion_haz_targets)
        eff = (1.0 - arr_f_cri[:, 0]) * arr_f_haz[:, -1]
        y_col = (arr_f_haz_t[:, -1] > 0.5).astype(int)
        if y_col.sum() > 0:
            plot_roc(y_col, eff, "Warning Fusion ROC", "results/roc_warning_fusion.png")
            print(f"  Fusion AUPRC: {average_precision_score(y_col, eff):.4f}")

    print("\nDone — see results/")
    for f in sorted(glob.glob("results/*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
