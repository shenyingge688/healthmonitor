"""
dump_val_outputs.py — run a checkpoint over the val set ONCE and save every
per-sample output, joined with the verified patient/record metadata. This npz is
the single foundation for all downstream statistics (bootstrap CI, calibration,
threshold sweep, lead-time, transition report) so they all reflect the SAME
finalized model.

Usage: py dump_val_outputs.py [checkpoint.pth]   (default = best composite)
Output: results/val_outputs.npz
"""
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from healthmonitor.dl_model import ArrhythmiaWarningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_val():
    ds = []
    for p in sorted(glob.glob("dataset/val_shard_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=True)
        ds.append(TensorDataset(d["X"], d["X_rr"], d["Y_cur"], d["Y_fut"], d["T_weight"]))
    return ConcatDataset(ds)


@torch.inference_mode()
def main(ckpt_path):
    meta = torch.load("dataset/val_meta.pt", weights_only=False)
    rec_id = np.asarray(meta["rec_id"])
    transition = np.asarray(meta["transition"])

    val = load_val()
    loader = DataLoader(val, batch_size=64, shuffle=False)
    model = ArrhythmiaWarningNet().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck.get("ema", ck.get("model", {})), strict=False)
    model.eval()

    logit_cur, logit_fut, prob_cur, prob_fut = [], [], [], []
    y_cur, y_fut = [], []
    for bx, bx_rr, yc, yf, tw in loader:
        bx = bx.to(device, dtype=torch.float32)
        bx_rr = bx_rr.to(device, dtype=torch.float32)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            out = model(bx, x_rr=bx_rr)
        logit_cur.append(out["logits_cur"].float().cpu().numpy())
        logit_fut.append(out["logits_fut"].float().cpu().numpy())
        prob_cur.append(out["probs_cur"].float().cpu().numpy())
        prob_fut.append(out["probs_fut"].float().cpu().numpy())
        y_cur.append(yc.numpy())
        y_fut.append(yf.float().numpy())

    logit_cur = np.concatenate(logit_cur)
    logit_fut = np.concatenate(logit_fut)
    prob_cur = np.concatenate(prob_cur)
    prob_fut = np.concatenate(prob_fut)
    y_cur = np.concatenate(y_cur)
    y_fut_dist = np.concatenate(y_fut)
    y_fut_maj = y_fut_dist.argmax(axis=1)

    assert len(rec_id) == len(y_cur), "meta/shard length mismatch"
    # sanity: stored Y_cur in meta must equal shard Y_cur
    assert (np.asarray(meta["cur"]) == y_cur).mean() > 0.999, "meta misaligned"

    os.makedirs("results", exist_ok=True)
    np.savez("results/val_outputs.npz",
             logit_cur=logit_cur, logit_fut=logit_fut,
             prob_cur=prob_cur, prob_fut=prob_fut,
             y_cur=y_cur, y_fut_maj=y_fut_maj, y_fut_dist=y_fut_dist,
             rec_id=rec_id, transition=transition,
             checkpoint=os.path.basename(ckpt_path),
             ckpt_epoch=int(ck.get("epoch", -1)))
    print(f"[dump] checkpoint={os.path.basename(ckpt_path)} epoch={ck.get('epoch')}")
    print(f"[dump] N={len(y_cur)}  records={len(set(rec_id.tolist()))}  "
          f"transitions={int(transition.sum())}")
    print("[dump] saved results/val_outputs.npz")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/arrhythmia_warning_best.pth"
    main(ckpt)
