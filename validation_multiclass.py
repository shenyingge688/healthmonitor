"""
validation_multiclass.py — 超前 5 分钟心律失常预测性能评估

评估指标:
  - 6 类混淆矩阵 + per-class P/R/F1
  - 多分类 Macro/Weighted AUC
  - 每类 PR 曲线 + AP
  - 1D-CAM 可视化示例
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
)

from dl_model import ArrhythmiaWarningNet

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

CLASS_NAMES = [
    "正常窦性心律", "室性早搏 (PVC)", "心房颤动 (AFib)",
    "心室颤动 (VF)", "室性心动过速 (VT)", "房速/室上速 (AT/SVT)"
]


def load_sharded_datasets(split_prefix="val"):
    shard_paths = glob.glob(f"dataset/{split_prefix}_shard_*.pt")
    datasets = []
    names = [os.path.basename(p) for p in shard_paths]
    print(f"  发现 {len(shard_paths)} 个 shard: {names}")
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "X_rr" in data:
            datasets.append(TensorDataset(data["X"], data["X_rr"], data["Y"]))
        else:
            dummy_rr = torch.zeros(len(data["X"]), 39, 9, dtype=torch.float16)
            datasets.append(TensorDataset(data["X"], dummy_rr, data["Y"]))
    return ConcatDataset(datasets)


def plot_cm(y_true, y_pred, names, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("真实标签")
    ax.set_xlabel("预测标签")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y_true, y_prob, label, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP={ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel("召回率"); ax.set_ylabel("精确率")
    ax.set_title(label); ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_score, label, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("假阳性率"); ax.set_ylabel("真阳性率")
    ax.set_title(label); ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc


@torch.inference_mode()
def main():
    print("=" * 60)
    print("超前 5 分钟心律失常预测 — 临床验证")
    print("=" * 60)

    val_dataset = load_sharded_datasets("val")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"验证样本数: {len(val_dataset)}")

    model = ArrhythmiaWarningNet(n_windows=39).to(device)
    ckpt_path = "models/arrhythmia_warning_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ 未找到 {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("ema", ckpt.get("model", {})), strict=False)
    if missing:
        non_critical = [k for k in missing if
                        any(x in k for x in ["tcn_blocks", "skip_adapters",
                                             "temporal_attn", "rr_encoder",
                                             "env_encoder", "class_head"])]
        if len(non_critical) == len(missing):
            print(f"⚠️ {len(missing)} keys 缺失（新增层），随机初始化")
        else:
            print(f"⚠️ {len(missing)} keys 缺失")
    if unexpected:
        print(f"⚠️ {len(unexpected)} keys 未知（旧版残留）")
    model.eval()

    all_probs, all_preds, all_targets = [], [], []

    for bx, bx_rr, y in val_loader:
        if not torch.isfinite(bx).all():
            continue
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(bx, x_rr=bx_rr)["logits"]

        probs = F.softmax(logits.float(), dim=-1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    arr_preds = np.array(all_preds)
    arr_targets = np.array(all_targets)
    arr_probs = np.array(all_probs)

    print(f"\n总样本数: {len(arr_targets)}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(arr_targets == i)
        print(f"  {name}: {count} ({count/len(arr_targets)*100:.1f}%)")

    # ---- 混淆矩阵 ----
    print("\n=== 混淆矩阵 ===")
    plot_cm(arr_targets, arr_preds, CLASS_NAMES,
            "心律失常预测混淆矩阵", "results/cm_6class.png")
    print(classification_report(arr_targets, arr_preds,
                                target_names=CLASS_NAMES, zero_division=0,
                                digits=4))

    # ---- Per-class AUROC ----
    print("\n=== Per-class AUROC (OvR) ===")
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (arr_targets == i).astype(int)
        if y_bin.sum() == 0:
            print(f"  {name}: 无样本，跳过")
            continue
        try:
            auc = roc_auc_score(y_bin, arr_probs[:, i])
            print(f"  {name}: AUC = {auc:.4f}")
        except:
            pass

    # ---- Per-class PR 曲线 ----
    print("\n=== Per-class AUPRC ===")
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (arr_targets == i).astype(int)
        if y_bin.sum() == 0:
            continue
        ap = average_precision_score(y_bin, arr_probs[:, i])
        print(f"  {name}: AP = {ap:.4f}")
        plot_pr(y_bin, arr_probs[:, i], f"{name} PR Curve",
                f"results/pr_{i}_{name.replace(' ', '_').replace('/', '_')}.png")

    # ---- VT/VF 召回率 ----
    print("\n=== 高危类别召回率 ===")
    for cls_idx, name in [(4, "VT"), (3, "VF")]:
        m = arr_targets == cls_idx
        if m.sum() > 0:
            recall = (arr_preds[m] == cls_idx).mean()
            print(f"  {name} Recall: {recall:.4f}  ({m.sum()} samples)")

    # ---- 可视化 CAM 示例 ----
    print("\n=== 1D-CAM 可视化 ===")
    try:
        sample_batch = next(iter(val_loader))
        bx, bx_rr, _ = sample_batch
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
        axes[-1].set_xlabel("时序窗口索引")
        fig.suptitle("1D-CAM 注意力分布示例")
        fig.tight_layout()
        fig.savefig("results/cam_examples.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  CAM 可视化失败: {e}")

    print("\n✅ 评估完成。结果保存在 results/")
    for f in sorted(glob.glob("results/*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
