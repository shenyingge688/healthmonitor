"""
eval_forecast.py — P4: does the future head actually FORECAST, or just persist?

The headline "future 2-5 min tendency" claim is only meaningful on samples where
the future majority class DIFFERS from the current state (a genuine upcoming
change). On steady samples, predicting future==current already wins.

This script reports:
  1. Full-val future-head metrics (reference).
  2. TRANSITION subset (argmax(Y_fut) != Y_cur): future-head recall/precision/F1
     vs a PERSISTENCE baseline (predict future = current). The model only earns
     a forecasting claim if it BEATS persistence here.
  3. Per-class future AUROC on the full val.

VF and AT/SVT are marked N/A where val support is 0 (Phase-1 scope: Normal/PVC/AFib/VT).
"""
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix)
from dl_model import ArrhythmiaWarningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
NUM_CLASSES = 6


def load_val():
    paths = sorted(glob.glob("dataset/val_shard_*.pt"))
    ds = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        ds.append(TensorDataset(d["X"], d["X_rr"], d["Y_cur"], d["Y_fut"], d["T_weight"]))
    return ConcatDataset(ds)


@torch.inference_mode()
def run():
    print("=" * 64)
    print("P4: Forecast vs Persistence — does the future head beat 'stay the same'?")
    print("=" * 64)

    val = load_val()
    loader = DataLoader(val, batch_size=64, shuffle=False)
    model = ArrhythmiaWarningNet().to(device)
    ckpt = torch.load("models/arrhythmia_warning_best.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", {})), strict=False)
    model.eval()

    y_cur, y_fut_maj, pred_fut, prob_fut = [], [], [], []
    for bx, bx_rr, yc, yf, tw in loader:
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)
        pf = out["probs_fut"].float().cpu().numpy()
        prob_fut.extend(pf)
        pred_fut.extend(pf.argmax(axis=-1))
        y_cur.extend(yc.numpy())
        y_fut_maj.extend(yf.argmax(dim=-1).numpy())

    y_cur = np.array(y_cur); y_fut = np.array(y_fut_maj)
    pred = np.array(pred_fut); prob = np.array(prob_fut)
    N = len(y_cur)

    # support
    print(f"\nVal N={N}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:8s}: cur={int((y_cur==c).sum()):4d}  futMaj={int((y_fut==c).sum()):4d}")

    # ---- 1. Full-val future head ----
    print("\n--- [1] FULL VAL: future head ---")
    acc = accuracy_score(y_fut, pred)
    f1m = f1_score(y_fut, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    print(f"  acc={acc:.3f}  macro-F1={f1m:.3f}")

    # ---- 2. Transition subset ----
    trans = y_fut != y_cur
    n_trans = int(trans.sum())
    print(f"\n--- [2] TRANSITION subset (futMaj != cur): N={n_trans} ({n_trans/N*100:.1f}% of val) ---")
    if n_trans >= 5:
        model_acc = accuracy_score(y_fut[trans], pred[trans])
        persist_pred = y_cur[trans]                      # persistence baseline
        persist_acc = accuracy_score(y_fut[trans], persist_pred)  # = 0 by construction on pure transitions
        model_f1 = f1_score(y_fut[trans], pred[trans], labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        print(f"  Model    : acc={model_acc:.3f}  macro-F1={model_f1:.3f}")
        print(f"  Persist. : acc={persist_acc:.3f}  (predict future=current; 0 on pure transitions)")
        print(f"  -> Model {'BEATS' if model_acc > persist_acc + 0.02 else 'does NOT beat'} persistence on transitions")
        # per-class on transition subset
        p, r, f, s = precision_recall_fscore_support(
            y_fut[trans], pred[trans], labels=list(range(NUM_CLASSES)), zero_division=0)
        print("  per-class (transition subset):")
        for c in range(NUM_CLASSES):
            if s[c] > 0:
                print(f"    {CLASS_NAMES[c]:8s} P={p[c]:.2f} R={r[c]:.2f} F1={f[c]:.2f} (n={int(s[c])})")
    else:
        print("  too few transition samples to evaluate")

    # ---- 3. Steady subset (sanity: model shouldn't be worse than persistence here) ----
    steady = ~trans
    if steady.sum() >= 5:
        m = accuracy_score(y_fut[steady], pred[steady])
        pb = accuracy_score(y_fut[steady], y_cur[steady])  # persistence = perfect on steady
        print(f"\n--- [3] STEADY subset N={int(steady.sum())} ---")
        print(f"  Model acc={m:.3f}  |  Persistence acc={pb:.3f} (perfect by definition)")

    # ---- 4. Per-class future AUROC (full val) ----
    print("\n--- [4] Future per-class AUROC (full val) ---")
    for c in range(NUM_CLASSES):
        yb = (y_fut == c).astype(int)
        if 0 < yb.sum() < N:
            try:
                print(f"  {CLASS_NAMES[c]:8s}: AUROC={roc_auc_score(yb, prob[:, c]):.3f} (n={int(yb.sum())})")
            except Exception:
                pass
        else:
            print(f"  {CLASS_NAMES[c]:8s}: N/A (support={int(yb.sum())})")

    # ---- 5. Confusion (future) ----
    print("\n--- [5] Future confusion matrix (rows=true, cols=pred) ---")
    cm = confusion_matrix(y_fut, pred, labels=list(range(NUM_CLASSES)))
    print("        " + " ".join(f"{n[:5]:>6s}" for n in CLASS_NAMES))
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i][:6]:6s} " + " ".join(f"{cm[i,j]:6d}" for j in range(NUM_CLASSES)))


if __name__ == "__main__":
    run()
