"""
stats_eval.py — validation-phase statistics on a FINALIZED checkpoint's outputs.

Consumes results/val_outputs.npz (from dump_val_outputs.py) and produces:
  [1] PATIENT-LEVEL bootstrap 95% CIs (resample RECORDS, not windows — because one
      record (afdb/08219) is ~48% of val, window bootstrap would be dishonestly tight)
  [2] Calibration of the future-tendency probability: ECE, Brier, reliability
      diagram, + temperature scaling (in-sample on val; flagged as such)
  [3] Threshold sweep / operating point: arrhythmia-alarm sensitivity vs
      false-alarms per patient-hour
  [4] Transition-subset vs persistence summary

Honest scope: Phase-1 evaluable classes = Normal/PVC/AFib/VT. VF/AT have 0 val
support and are reported N/A. Claims for VT are exploratory (few patients).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, roc_auc_score

CLASS_NAMES = ["Normal", "PVC", "AFib", "VF", "VT", "AT/SVT"]
STRIDE_SEC = 10.0  # FIXED_STRIDE_SEC -> one decision per 10s -> 360 decisions/hour
RNG = np.random.RandomState(20260608)
N_BOOT = 2000


def load():
    d = np.load("results/val_outputs.npz", allow_pickle=True)
    return d


def macro_f1_present(y_true, y_pred):
    present = sorted(set(y_true.tolist()))
    return f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)


def boot_metric(rec_id, fn):
    """Patient-level bootstrap: resample unique records with replacement."""
    recs = np.array(sorted(set(rec_id.tolist())))
    idx_by_rec = {r: np.where(rec_id == r)[0] for r in recs}
    vals = []
    for _ in range(N_BOOT):
        chosen = RNG.choice(recs, size=len(recs), replace=True)
        idx = np.concatenate([idx_by_rec[r] for r in chosen])
        v = fn(idx)
        if v is not None:
            vals.append(v)
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5), len(vals)


def section_bootstrap(d):
    print("\n" + "=" * 64)
    print("[1] PATIENT-LEVEL BOOTSTRAP 95%% CI (resample by record, B=%d)" % N_BOOT)
    print("=" * 64)
    y_fut = d["y_fut_maj"]; pred = d["prob_fut"].argmax(1); prob = d["prob_fut"]
    y_cur = d["y_cur"]; rec = d["rec_id"]; trans = d["transition"].astype(bool)
    n_rec = len(set(rec.tolist()))
    print(f"records={n_rec}  N={len(y_fut)}  (one record may dominate — see CI width)")

    def f1_fn(idx):
        return macro_f1_present(y_fut[idx], pred[idx])

    def recall_cls(idx, c):
        m = y_fut[idx] == c
        if m.sum() == 0:
            return None
        return float((pred[idx][m] == c).mean())

    def auroc_cls(idx, c):
        yb = (y_fut[idx] == c).astype(int)
        if 0 < yb.sum() < len(yb):
            try:
                return float(roc_auc_score(yb, prob[idx][:, c]))
            except Exception:
                return None
        return None

    def trans_recall(idx):
        ti = idx[trans[idx]]
        if len(ti) < 3:
            return None
        present = sorted(set(y_fut[ti].tolist()))
        return recall_score(y_fut[ti], pred[ti], labels=present, average="macro", zero_division=0)

    rows = [
        ("Future macro-F1", lambda idx: f1_fn(idx)),
        ("PVC future recall", lambda idx: recall_cls(idx, 1)),
        ("AFib future recall", lambda idx: recall_cls(idx, 2)),
        ("VT future recall (exploratory)", lambda idx: recall_cls(idx, 4)),
        ("PVC future AUROC", lambda idx: auroc_cls(idx, 1)),
        ("AFib future AUROC", lambda idx: auroc_cls(idx, 2)),
        ("Transition macro-recall", trans_recall),
    ]
    for name, fn in rows:
        m, lo, hi, nb = boot_metric(rec, fn)
        print(f"  {name:32s}: {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_boot={nb})")
    print("  NOTE: wide CIs reflect few independent patients — this is the honest")
    print("        data limit, reported rather than hidden.")


def temperature_scale(logits, y_true, iters=200, lr=0.05):
    import torch
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.long)
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=iters)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(z / T.clamp(min=1e-2), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-2).item())


def ece(prob, y_true, n_bins=10):
    conf = prob.max(1); pred = prob.argmax(1)
    correct = (pred == y_true).astype(float)
    e = 0.0; edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            acc = correct[m].mean(); cf = conf[m].mean(); w = m.mean()
            e += w * abs(acc - cf)
            bins.append((cf, acc, m.sum()))
    return e, bins


def brier(prob, y_dist):
    return float(((prob - y_dist) ** 2).sum(1).mean())


def section_calibration(d):
    print("\n" + "=" * 64)
    print("[2] CALIBRATION of future-tendency probability")
    print("=" * 64)
    prob = d["prob_fut"]; y = d["y_fut_maj"]; ydist = d["y_fut_dist"]; logit = d["logit_fut"]
    e0, bins0 = ece(prob, y)
    b0 = brier(prob, ydist)
    print(f"  Pre-scaling : ECE={e0:.3f}  Brier={b0:.3f}")
    T = temperature_scale(logit, y)
    pe = np.exp((logit / T) - (logit / T).max(1, keepdims=True))
    prob_t = pe / pe.sum(1, keepdims=True)
    e1, bins1 = ece(prob_t, y)
    b1 = brier(prob_t, ydist)
    print(f"  Temp scaling: T={T:.2f}  ECE={e1:.3f}  Brier={b1:.3f}  (in-sample on val; "
          f"needs external fold for a final claim)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    if bins0:
        cf, acc, _ = zip(*bins0); ax.plot(cf, acc, "o-", color="#EF4444", label=f"raw (ECE {e0:.2f})")
    if bins1:
        cf, acc, _ = zip(*bins1); ax.plot(cf, acc, "s-", color="#10B981", label=f"T={T:.2f} (ECE {e1:.2f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Future-head reliability")
    ax.legend(); fig.tight_layout(); fig.savefig("results/calibration_future.png", dpi=150)
    print("  saved results/calibration_future.png")


def section_threshold(d):
    print("\n" + "=" * 64)
    print("[3] OPERATING POINT: arrhythmia-alarm sensitivity vs false alarms/patient-hour")
    print("=" * 64)
    prob = d["prob_fut"]; y = d["y_fut_maj"]
    alarm_score = 1.0 - prob[:, 0]          # P(future is NOT normal)
    is_arr = y != 0                         # true upcoming arrhythmia
    is_norm = y == 0
    per_hour = 3600.0 / STRIDE_SEC          # decisions per patient-hour
    print(f"  decisions/hour={per_hour:.0f}  arrhythmia samples={int(is_arr.sum())}  normal={int(is_norm.sum())}")
    print(f"  {'thr':>5s} {'sens':>6s} {'spec':>6s} {'FA/hr':>7s}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        flag = alarm_score >= thr
        sens = (flag & is_arr).sum() / max(is_arr.sum(), 1)
        spec = (~flag & is_norm).sum() / max(is_norm.sum(), 1)
        fa_per_hr = (flag & is_norm).mean() * per_hour
        print(f"  {thr:5.2f} {sens:6.3f} {spec:6.3f} {fa_per_hr:7.1f}")
    print("  (FA/hr = false alarms per patient-hour at stride-10s decisions; lower is better)")


def section_transition(d):
    print("\n" + "=" * 64)
    print("[4] TRANSITION subset vs PERSISTENCE baseline")
    print("=" * 64)
    y = d["y_fut_maj"]; pred = d["prob_fut"].argmax(1); cur = d["y_cur"]; trans = d["transition"].astype(bool)
    nt = int(trans.sum())
    macc = (pred[trans] == y[trans]).mean()
    pacc = (cur[trans] == y[trans]).mean()  # persistence = current; 0 on pure transitions
    print(f"  transition N={nt}")
    print(f"  Model acc on transitions     : {macc:.3f}")
    print(f"  Persistence acc on transitions: {pacc:.3f}")
    print(f"  -> Model {'BEATS' if macc > pacc + 0.02 else 'does NOT beat'} persistence "
          f"(continuation baseline necessarily fails on pure transitions)")
    for c in [1, 2, 4]:
        m = trans & (y == c)
        if m.sum() > 0:
            print(f"    {CLASS_NAMES[c]:6s} transition recall: {(pred[m]==c).mean():.3f} (n={int(m.sum())})")


def main():
    d = load()
    print(f"Checkpoint: {d['checkpoint']} (epoch {int(d['ckpt_epoch'])})")
    section_bootstrap(d)
    section_calibration(d)
    section_threshold(d)
    section_transition(d)
    print("\n[done] statistics complete.")


if __name__ == "__main__":
    main()
