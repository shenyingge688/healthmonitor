"""单独测试推理管线，找出 500 错误原因"""
import sys, os, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from dl_model import ArrhythmiaWarningNet
from main import build_window_sequence, device

print(f"Device: {device}")
print(f"Model loading...")
model = ArrhythmiaWarningNet(n_windows=39).to(device).eval()
print(f"Model OK, params: {sum(p.numel() for p in model.parameters()):,}")

# 模拟全零输入
arr = [0.0] * 180000
print(f"\nInput: {len(arr)} points")

try:
    # Step 1
    ecg_array = np.pad(arr, (180000 - len(arr), 0), 'constant')
    print(f"[1] pad OK, dense={ecg_array.shape}")

    # Step 2
    win_seq, rr_seq = build_window_sequence(ecg_array)
    print(f"[2] win={win_seq.shape} rr={rr_seq.shape}")

    # Step 3
    bx = torch.from_numpy(win_seq).unsqueeze(0).to(device)
    bx_rr = torch.from_numpy(rr_seq).unsqueeze(0).to(device)
    print(f"[3] tensor OK, bx={bx.shape} bx_rr={bx_rr.shape}")

    # Step 4 - try without autocast first
    use_amp = (device.type == "cuda")
    print(f"[4] use_amp={use_amp}")
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(bx, x_rr=bx_rr)
        else:
            out = model(bx, x_rr=bx_rr)
    print(f"[4] forward OK")

    # Step 5
    logits = out["logits"][0].float()
    cam = out["cam"][0].float()
    probs = torch.softmax(logits, dim=-1)
    print(f"[5] logits={logits.shape} probs={probs.tolist()}")

    print("\n✅ ALL OK")
except Exception as e:
    print(f"\n❌ FAILED at step above")
    traceback.print_exc()
