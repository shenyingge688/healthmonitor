"""
Script: dashboard.py — 心电监护预警演示终端
"""
import streamlit as st
import requests
import numpy as np
import wfdb
import time
import pandas as pd
import altair as alt
from collections import deque
from scipy import signal
from scipy.signal import butter, filtfilt
import os

st.set_page_config(page_title="心电监护预警系统", layout="wide")

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("心电监护预警系统")

clinical_cases = {
    "100":   {"label": "案例一：正常窦性心律 (NSR)",
              "desc": "标准窦性心律，无明显心律失常事件，用于验证系统基线稳定性。",
              "default_start": 10.0, "db": "mitdb"},
    "119":   {"label": "案例二：频发室性期前收缩 (PVC)",
              "desc": "存在大量室性异位搏动，PVC 负荷显著增高，考验模型对室性节律的识别精度。",
              "default_start": 10.0, "db": "mitdb"},
    "209":   {"label": "案例三：房性心动过速 / 室上速 (AT / SVT)",
              "desc": "阵发性房速与室上性心动过速交替出现，窄 QRS 心动过速，验证室上性分类的稳定性。",
              "default_start": 10.0, "db": "mitdb"},
    "04015": {"label": "案例四：持续性心房颤动 (AF)",
              "desc": "长程房颤记录（10 小时以上），RR 间期绝对不齐，评估室上性分类在长程监测中的一致性。",
              "default_start": 10.0, "db": "afdb"},
    "421":   {"label": "案例五：持续性室性心动过速 (VT)",
              "desc": "持续室性心动过速发作，宽 QRS 心动过速，验证危急度分类对高危事件的响应能力。",
              "default_start": 23.0, "db": "vfdb"},
}

selected_id = st.sidebar.selectbox(
    "选择监测源", options=list(clinical_cases.keys()),
    format_func=lambda x: clinical_cases[x]["label"]
)
start_mins = st.sidebar.slider(
    "推演起始点 (分钟)", 10.0, 25.0,
    clinical_cases[selected_id]["default_start"], 0.5
)
st.sidebar.caption("前 10 分钟用于建立基线参考。")

if st.sidebar.button("清空追踪日志", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()

# =========================================================
# Data Loading
# =========================================================
def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data(ttl=3600)
def load_sim_data(rec_id, db):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', db, rec_id)
    if not os.path.exists(rec_path + ".dat"):
        return np.zeros(300000)
    if db == 'mitdb':
        record = wfdb.rdrecord(rec_path, sampto=int(20 * 60 * 360))
        src_fs = 360
    elif db == 'vfdb':
        # VFDB: full 35-min record for VT peak access
        record = wfdb.rdrecord(rec_path)
        src_fs = record.fs if hasattr(record, 'fs') and record.fs else 250
    else:
        record = wfdb.rdrecord(rec_path, sampto=int(30 * 60 * 250))
        src_fs = record.fs if hasattr(record, 'fs') and record.fs else 250
    sig = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    if src_fs == 250:
        return sig.astype(np.float32)
    if db == 'mitdb':
        sig = clean_ecg_signal(sig, fs=src_fs)
    return signal.resample_poly(sig, 250, src_fs).astype(np.float32)

data_source = load_sim_data(selected_id, clinical_cases[selected_id]["db"])

st.title(f"心电监护预警终端 — {clinical_cases[selected_id]['label']}")
st.caption(f"**病史/体征**：{clinical_cases[selected_id]['desc']}")

# =========================================================
# Unified classification
# =========================================================
diag_placeholder = st.empty()

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### 实时心电波形 (250Hz 单导联)")
    chart_placeholder = st.empty()
    st.markdown("### 演变趋势")
    traj_placeholder = st.empty()

with col2:
    st.markdown("### 推演时间")
    metric_time = st.empty()
    st.markdown("### 临床诊断")
    class_placeholder = st.empty()

st.divider()
st.markdown("### 事件日志")
log_placeholder = st.empty()

# =========================================================
# Unified classification
# =========================================================
def unified_diagnosis(rhythm, crit):
    """综合节律分类与危急度评估，输出主导诊断"""
    candidates = [
        ("正常窦性心律 (NSR)", rhythm[0], "#10B981"),
        ("室性期前收缩 (PVC)", rhythm[1], "#F59E0B"),
        ("室上性心律失常 (AF / AFl / AT / SVT)", rhythm[2], "#F97316"),
        ("室性心动过速 (VT)", crit[2], "#EF4444"),
        ("心室颤动 (VF)", crit[3], "#991B1B"),
    ]
    diag, conf, color = max(candidates, key=lambda x: x[1])
    if conf < 0.5:
        return "信号质量不足", conf, "#94A3B8"
    return diag, conf, color

# Unified class display
UNI_CLASSES = [
    ("正常窦性心律 (NSR)",                        lambda r, c: r[0], "#10B981"),
    ("室性期前收缩 (PVC)",                         lambda r, c: r[1], "#F59E0B"),
    ("室上性心律失常 (AF / AFl / AT / SVT)",           lambda r, c: r[2], "#F97316"),
    ("室性心动过速 (VT)",                         lambda r, c: c[2], "#EF4444"),
    ("心室颤动 (VF)",                              lambda r, c: c[3], "#991B1B"),
]

def bar_html(label, prob, color):
    p = min(float(prob), 1.0)
    w = p * 100
    c = color if p > 0.03 else "rgba(255,255,255,0.12)"
    return (
        f"<div style='display:flex;align-items:center;margin-bottom:4px;font-size:12px;'>"
        f"<span style='width:180px;color:#E2E8F0;'>{label}</span>"
        f"<div style='flex:1;background:rgba(0,0,0,0.3);height:7px;border-radius:3px;margin:0 8px;'>"
        f"<div style='width:{w}%;background:{c};height:100%;border-radius:3px;'></div>"
        f"</div>"
        f"<span style='width:40px;text-align:right;font-family:monospace;color:#94A3B8;'>{w:.0f}%</span>"
        f"</div>"
    )

# =========================================================
# Simulation Loop
# =========================================================
if st.sidebar.button("启动数据推流", use_container_width=True):
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    display_len, step_size, refresh_rate = 1000, 20, 0.08

    rhy_hist = deque(maxlen=5)
    cri_hist = deque(maxlen=5)
    risk_history = deque([0.0] * 10, maxlen=10)

    while True:
        current_pts = base_offset_pts + (sim_step * step_size)
        if current_pts >= len(data_source) - 1:
            break

        # ECG
        new_data = data_source[current_pts - step_size : current_pts]
        idx = (sim_step * step_size) % display_len
        ecg_buffer[idx : idx + step_size] = new_data
        gap = (idx + step_size + 25) % display_len
        if gap > (idx + step_size):
            ecg_buffer[idx + step_size : gap] = np.nan
        else:
            ecg_buffer[idx + step_size : display_len] = np.nan
            ecg_buffer[0 : gap] = np.nan

        df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': ecg_buffer})
        chart_placeholder.altair_chart(
            alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
                x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None),
                y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]),
                        axis=alt.Axis(title="mV", grid=True))
            ).properties(height=200).configure_view(strokeOpacity=0),
            use_container_width=True
        )

        cur_sec = current_pts / 250
        metric_time.metric("当前扫描线", f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}")

        # API
        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 150000) : current_pts]
            if len(full_win) < 150000:
                full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')

            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict",
                                     json={"ecg": full_win.tolist()}, timeout=5)
                if resp.status_code == 200:
                    d = resp.json()
                    rhy_hist.append(d.get('rhythm', [1, 0, 0]))
                    cri_hist.append(d.get('criticality', [1, 0, 0, 0]))
                    risk_history.append(d.get('risk_trajectory', 0.0))
            except Exception:
                pass

            if rhy_hist:
                rhythm = np.mean(list(rhy_hist), axis=0).tolist()
                crit = np.mean(list(cri_hist), axis=0).tolist()
            else:
                rhythm, crit = [1, 0, 0], [1, 0, 0, 0]

            # ---- 综合诊断 ----
            diag, conf, color = unified_diagnosis(rhythm, crit)

            diag_placeholder.markdown(
                f"<div style='background:#1E1E28;padding:20px 28px;border-radius:10px;"
                f"border-left:10px solid {color};box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
                f"<h1 style='margin:0;color:{color};font-size:26px;font-weight:700;'>{diag}</h1>"
                f"<p style='margin:6px 0 0;color:#94A3B8;font-size:14px;'>"
                f"置信度 {conf*100:.1f}%</p></div>",
                unsafe_allow_html=True
            )

            # Unified probability bars
            bars = ""
            for label, fn, c in UNI_CLASSES:
                bars += bar_html(label, fn(rhythm, crit), c)
            class_placeholder.markdown(
                f"<div style='background:#1E1E28;padding:12px 16px;border-radius:6px;'>{bars}</div>",
                unsafe_allow_html=True
            )

            # Risk trend
            traj_list = list(risk_history)
            t_axis = np.linspace(-len(traj_list) + 1, 0, len(traj_list))
            df_traj = pd.DataFrame({'Time': t_axis, 'Risk': traj_list})
            traj_placeholder.altair_chart(
                alt.Chart(df_traj).mark_area(
                    color=alt.Gradient(gradient='linear', stops=[
                        alt.GradientStop(color='rgba(239,68,68,0.05)', offset=0),
                        alt.GradientStop(color='rgba(239,68,68,0.6)', offset=1)
                    ]), line={'color': '#EF4444'}
                ).encode(
                    x=alt.X('Time:Q', axis=alt.Axis(title="历史", grid=False)),
                    y=alt.Y('Risk:Q', scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(title="风险", format='%'))
                ).properties(height=100).configure_view(strokeOpacity=0),
                use_container_width=True
            )

            # Log
            if crit[2] > 0.3 or crit[3] > 0.3 or (rhythm[2] > 0.5):
                ts = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                st.session_state.anomaly_logs.append({"time": ts, "title": diag})
                if len(st.session_state.anomaly_logs) > 20:
                    st.session_state.anomaly_logs = st.session_state.anomaly_logs[-20:]
                lines = []
                for e in reversed(st.session_state.anomaly_logs):
                    lines.append(
                        f"<div style='padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05);"
                        f"font-size:12px;color:#94A3B8;'>"
                        f"<span style='color:#F59E0B;font-family:monospace;'>{e['time']}</span> "
                        f"— {e['title']}</div>"
                    )
                log_placeholder.markdown(
                    f"<div style='max-height:240px;overflow-y:auto;'>{''.join(lines)}</div>",
                    unsafe_allow_html=True
                )

        sim_step += 1
        time.sleep(refresh_rate)
else:
    diag_placeholder.info("请选择病例并点击【启动数据推流】开始推演。")
    chart_placeholder.line_chart(np.full(1000, 0.0), height=200, use_container_width=True)
