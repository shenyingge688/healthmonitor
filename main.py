"""
Script: main.py
心电监护预警系统 — FastAPI 推理服务
"""
import os, sys, traceback, uvicorn
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import find_peaks
from dl_model import ArrhythmiaWarningNet

# ---- 参数 ----
TARGET_FS = 250
HISTORY_SEC = 600
WINDOW_SEC = 30
OVERLAP_STRIDE_SEC = 15
PTS_PER_WIN = WINDOW_SEC * TARGET_FS          # 7500
STRIDE_PTS = OVERLAP_STRIDE_SEC * TARGET_FS    # 3750
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1  # 39
HISTORY_PTS = (N_WINDOWS - 1) * STRIDE_PTS + PTS_PER_WIN  # 150000
API_BUFFER_SIZE = 180000

CLASS_NAMES = ["正常窦性心律", "室性早搏 (PVC)", "心房颤动 (AFib)",
               "心室颤动 (VF)", "室性心动过速 (VT)", "房速/室上速 (AT/SVT)"]

app = FastAPI(title="心电监护预警推理引擎")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 加载模型 ----
model = ArrhythmiaWarningNet(n_windows=N_WINDOWS).to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), "models", "arrhythmia_warning_best.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", {})), strict=False)
    print("✅ 模型权重已加载")
else:
    print("⚠️ 未找到权重文件，使用随机初始化")
model.eval()
print(f"✅ 推理引擎就绪，设备={device}")

# ---- RR 特征提取 ----
def _extract_rr(win, fs=250):
    try:
        seg = np.asarray(win, dtype=np.float32)
        th = 0.5 * np.std(seg)
        if th < 0.02:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        peaks, _ = find_peaks(seg, height=th, distance=int(fs*0.25))
        if len(peaks) < 3:
            return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)
        rr = np.diff(peaks) / fs * 1000.0
        return (np.array([np.mean(rr)/1000.0, np.std(rr)/300.0,
                          np.sqrt(np.mean(np.diff(rr)**2))/100.0 if len(rr)>1 else 0.0,
                          np.mean(np.abs(np.diff(rr))>50) if len(rr)>1 else 0.0], dtype=np.float32),
                rr.astype(np.float32))
    except:
        return np.zeros(4, dtype=np.float32), np.array([], dtype=np.float32)

def _sample_entropy(rr, m=2, r_f=0.2):
    rr = np.asarray(rr, dtype=np.float64); N = len(rr)
    if N < m+2: return 0.0
    rv = r_f * np.std(rr)
    if rv < 1e-10: return 0.0
    def _phi(tl):
        t = np.array([rr[i:i+tl] for i in range(N-tl)])
        nt = len(t)
        if nt < 2: return 0
        return max(sum(np.sum(np.max(np.abs(t-t[i]), axis=1)<rv)-1 for i in range(nt)), 0)
    A, B = _phi(m+1), _phi(m)
    return float(-np.log(A/B)/2.0) if B>=1 and A>0 else 0.0

def _poincare(rr):
    rr = np.asarray(rr, dtype=np.float64)
    if len(rr) < 2: return np.array([0.0,0.0,0.0], dtype=np.float32)
    d = rr[1:]-rr[:-1]; vd = np.var(d); vr = np.var(rr)
    sd1 = np.sqrt(0.5*max(vd,0)); sd2 = np.sqrt(max(2*vr-0.5*vd,0))
    return np.array([sd1/100.0, sd2/200.0, sd1/max(sd2,1e-10)], dtype=np.float32)

def _traj_rr(all_rr):
    if len(all_rr) < 10: return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    p = _poincare(rr)
    return np.array([_sample_entropy(rr), p[0], p[1], p[2],
                     float(np.std(rr)/max(np.mean(rr),1e-8))], dtype=np.float32)

def build_window_sequence(ecg_array):
    rel = ecg_array[-HISTORY_PTS:]
    wins, f4s, all_rr = [], [], []
    for wi in range(N_WINDOWS):
        s, e = wi*STRIDE_PTS, wi*STRIDE_PTS+PTS_PER_WIN
        x = rel[s:e]
        x_n = (x-np.mean(x))/(np.std(x)+1e-8)
        wins.append(x_n.astype(np.float32))
        f4, iv = _extract_rr(x)
        f4s.append(f4)
        if len(iv)>0: all_rr.append(iv)
    tf = _traj_rr(np.concatenate(all_rr)).astype(np.float32) if all_rr else np.zeros(5, dtype=np.float32)
    rr_f = np.array([np.concatenate([f4s[wi], tf]) for wi in range(N_WINDOWS)], dtype=np.float32)
    return np.stack(wins)[:,np.newaxis,:], rr_f

# ---- API ----
class ECGPayload(BaseModel):
    ecg: list[float]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def predict(payload: ECGPayload):
    try:
        arr = np.array(payload.ecg[-API_BUFFER_SIZE:] if len(payload.ecg)>=API_BUFFER_SIZE
                       else list(payload.ecg) + [0]*(API_BUFFER_SIZE-len(payload.ecg)), dtype=np.float32)
        ws, rs = build_window_sequence(arr)
        bx = torch.from_numpy(ws).unsqueeze(0).to(device)
        bx_rr = torch.from_numpy(rs).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(bx, x_rr=bx_rr)
        logits = out["logits"][0].cpu().float()
        probs = F.softmax(logits, dim=-1).tolist()
        return {"pred_class": int(torch.argmax(logits).item()),
                "class_name": CLASS_NAMES[int(torch.argmax(logits).item())],
                "probabilities": probs,
                "cam": out["cam"][0].cpu().float().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc()[-500:])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
