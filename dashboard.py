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
    "201":   {"label": "案例四：阵发性心房颤动 (AFib)",
              "desc": "MIT-BIH 201 号记录，阵发性房颤发作，RR 间期绝对不齐，P 波消失，f 波显现。",
              "default_start": 10.0, "db": "mitdb"},
    "207":   {"label": "案例五：室性心动过速 / 室扑 (VT / VFl)",
              "desc": "MIT-BIH 207 号记录，短阵室速与心室扑动交替，宽 QRS 心动过速，高危节律样本。",
              "default_start": 12.0, "db": "mitdb"},
}

selected_id = st.sidebar.selectbox(
    "选择监测源", options=list(clinical_cases.keys()),
    format_func=lambda x: clinical_cases[x]["label"]
)
start_mins = st.sidebar.slider(
    "推演起始点 (分钟)", 10.0, 16.0,
    clinical_cases[selected_id]["default_start"], 0.5
)
st.sidebar.caption("前 10 分钟用于建立基线参考。上限 16 分钟（需预留 12 分钟缓冲）。")

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
        record = wfdb.rdrecord(rec_path, sampto=int(30 * 60 * 360))
        src_fs = 360
    elif db == 'vfdb':
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
# Layout
# =========================================================
status_placeholder = st.empty()
diag_placeholder = st.empty()

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### 实时心电波形 (250Hz 单导联)")
    chart_placeholder = st.empty()
    st.markdown("### 心律失常概率演变趋势")
    traj_placeholder = st.empty()
    st.markdown("### 病灶溯源热力图 (1D-CAM)")
    cam_placeholder = st.empty()

with col2:
    st.markdown("### 推演时间")
    metric_time = st.empty()
    st.markdown("### 预警诊断")
    class_placeholder = st.empty()

st.divider()
st.markdown("### 事件日志")
log_placeholder = st.empty()

# =========================================================
# Constants
# =========================================================
CLASS_NAMES = [
    "正常窦性心律", "室性早搏 (PVC)", "心房颤动 (AFib)",
    "心室颤动 (VF)", "室性心动过速 (VT)", "房速/室上速 (AT/SVT)"
]

CLASS_COLORS = [
    "#10B981", "#F59E0B", "#F97316",
    "#991B1B", "#EF4444", "#EAB308"
]

INFERENCE_MAP = {
    0: {"title": "未见明显节律异常", "color": "#10B981"},
    1: {"title": "室性期前收缩 (PVC) 风险", "color": "#F59E0B"},
    2: {"title": "心房颤动 (AFib) 风险", "color": "#F97316"},
    3: {"title": "心室颤动 (VF) 风险", "color": "#991B1B"},
    4: {"title": "室性心动过速 (VT) 风险", "color": "#EF4444"},
    5: {"title": "房速/室上速 (AT/SVT) 风险", "color": "#EAB308"},
}


def bar_html(label, prob, color):
    p = min(float(prob), 1.0)
    w = p * 100
    c = color if p > 0.03 else "rgba(255,255,255,0.12)"
    return (
        f"<div style='display:flex;align-items:center;margin-bottom:4px;font-size:12px;'>"
        f"<span style='width:170px;color:#E2E8F0;'>{label}</span>"
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

    prob_history = list(deque(maxlen=60))      # 60 帧概率历史 (~60s)
    pred_class_prev = 0

    while True:
        current_pts = base_offset_pts + (sim_step * step_size)
        if current_pts >= len(data_source) - 1:
            break

        # ECG buffer
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

        # API call (~1 Hz)
        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 180000) : current_pts]
            if len(full_win) < 180000:
                full_win = np.pad(full_win, (180000 - len(full_win), 0), 'constant')

            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict",
                                     json={"ecg": full_win.tolist()}, timeout=10)
                resp.raise_for_status()
                d = resp.json()
                pred_class = d["pred_class"]
                probs = d["probabilities"]
                cam_data = d.get("cam", [])
                prob_history.append(probs)
            except Exception:
                pred_class = pred_class_prev
                cam_data = []
        else:
            pred_class = pred_class_prev
            cam_data = []

        # ---- 预警诊断 ----
        diag_info = INFERENCE_MAP.get(pred_class, INFERENCE_MAP[0])
        diag_placeholder.markdown(
            f"<div style='background:#1E1E28;padding:20px 28px;border-radius:10px;"
            f"border-left:10px solid {diag_info['color']};box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
            f"<h1 style='margin:0;color:{diag_info['color']};font-size:26px;font-weight:700;'>{diag_info['title']}</h1>"
            f"<p style='margin:6px 0 0;color:#94A3B8;font-size:14px;'>未来 5 分钟预警</p></div>",
            unsafe_allow_html=True
        )

        # ---- 概率分布条 ----
        if prob_history:
            probs_now = prob_history[-1]
            bars = ""
            for i, name in enumerate(CLASS_NAMES):
                bars += bar_html(name, probs_now[i], CLASS_COLORS[i])
            class_placeholder.markdown(
                f"<div style='background:#1E1E28;padding:12px 16px;border-radius:6px;'>{bars}</div>",
                unsafe_allow_html=True
            )

        # ---- 概率演变趋势 (堆叠面积图) ----
        if len(prob_history) > 1:
            df_traj = pd.DataFrame(prob_history, columns=CLASS_NAMES)
            t_axis = np.linspace(-len(prob_history) + 1, 0, len(prob_history))
            df_traj['Time'] = t_axis

            df_long = df_traj.melt('Time', var_name='类别', value_name='概率')
            traj_placeholder.altair_chart(
                alt.Chart(df_long).mark_area(opacity=0.7).encode(
                    x=alt.X('Time:Q', axis=alt.Axis(title="时间 (步)", grid=False)),
                    y=alt.Y('概率:Q', scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(title="概率", format='%')),
                    color=alt.Color('类别:N',
                                    scale=alt.Scale(
                                        domain=CLASS_NAMES,
                                        range=CLASS_COLORS),
                                    legend=alt.Legend(orient='bottom', columns=3))
                ).properties(height=140).configure_view(strokeOpacity=0),
                use_container_width=True
            )

        # ---- 1D-CAM 热力图 ----
        if cam_data and len(cam_data) > 1:
            df_cam = pd.DataFrame({'TimeStep': np.arange(len(cam_data)), 'Attention': cam_data})
            cam_placeholder.altair_chart(
                alt.Chart(df_cam).mark_area(
                    color=alt.Gradient(gradient='linear', stops=[
                        alt.GradientStop(color='#000000', offset=0),
                        alt.GradientStop(color='#FF4444', offset=1)
                    ])
                ).encode(
                    x=alt.X('TimeStep:Q', axis=alt.Axis(labels=False, title="时序注意力分布")),
                    y=alt.Y('Attention:Q', scale=alt.Scale(domain=[0, 1.0]), axis=None)
                ).properties(height=50).configure_view(strokeOpacity=0),
                use_container_width=True
            )

        # ---- 事件日志 ----
        if pred_class != 0:
            ts = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
            diag_title = CLASS_NAMES[pred_class]
            # 仅在新事件或类别切换时才记录
            if not st.session_state.anomaly_logs or \
               st.session_state.anomaly_logs[-1].get("class") != pred_class:
                st.session_state.anomaly_logs.append({
                    "time": ts, "title": diag_title, "class": pred_class
                })
                if len(st.session_state.anomaly_logs) > 20:
                    st.session_state.anomaly_logs = st.session_state.anomaly_logs[-20:]

        if st.session_state.anomaly_logs:
            lines = []
            for e in reversed(st.session_state.anomaly_logs):
                c = CLASS_COLORS.get(e.get("class", 0), "#94A3B8")
                lines.append(
                    f"<div style='padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05);"
                    f"font-size:12px;color:#94A3B8;'>"
                    f"<span style='color:#F59E0B;font-family:monospace;'>{e['time']}</span> "
                    f"— <span style='color:{c};'>{e['title']}</span></div>"
                )
            log_placeholder.markdown(
                f"<div style='max-height:240px;overflow-y:auto;'>{''.join(lines)}</div>",
                unsafe_allow_html=True
            )
        else:
            log_placeholder.info("✅ 当前监测时段内未检测到异常节律。")

        pred_class_prev = pred_class
        sim_step += 1
        time.sleep(refresh_rate)
else:
    diag_placeholder.info("请选择病例并点击【启动数据推流】开始推演。")
    chart_placeholder.line_chart(np.full(1000, 0.0), height=200, use_container_width=True)
