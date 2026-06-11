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

from constants import CLASS_NAMES, CLASS_COLORS, INFERENCE_MAP

st.set_page_config(page_title="心电监护预警系统", layout="wide")

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("心电监护预警系统")

clinical_cases = {
    "100":   {"label": "案例一：正常窦性心律 (NSR)",
              "desc": "MIT-BIH 100 号记录，标准窦性心律，无明显心律失常事件，用于验证系统基线稳定性与误报抑制。",
              "default_start": 10.0, "db": "mitdb", "tier": "core"},
    "119":   {"label": "案例二：频发室性期前收缩 (PVC)",
              "desc": "MIT-BIH 119 号记录，存在大量室性异位搏动（二/三联律），PVC 负荷显著，考验当前节律头对室性异位的识别精度。",
              "default_start": 10.0, "db": "mitdb", "tier": "core"},
    "210":   {"label": "案例三：持续性心房颤动 (AFib)",
              "desc": "MIT-BIH 210 号记录，持续性房颤，RR 间期绝对不齐、P 波消失。当前节律头与未来倾向头均稳定输出 AFib（最强主张类）。",
              "default_start": 10.0, "db": "mitdb", "tier": "core"},
    "223":   {"label": "案例四：室性心动过速发作 (VT onset)",
              "desc": "MIT-BIH 223 号记录，约第 9 分钟由窦性节律转入室速短阵发作，演示系统对 Normal→VT 状态转换的检出能力。",
              "default_start": 8.0, "db": "mitdb", "tier": "core"},
    "209":   {"label": "案例五：房速 / 室上速 (AT / SVT)〔探索性〕",
              "desc": "MIT-BIH 209 号记录，阵发性房速 / 室上速。⚠ 该类训练样本极少、验证集为 0，属数据受限的探索性类别；模型当前倾向将其归为 VT/AFib，结果仅供能力边界参考，不作为有效主张。",
              "default_start": 10.0, "db": "mitdb", "tier": "exploratory"},
}

selected_id = st.sidebar.selectbox(
    "选择监测源", options=list(clinical_cases.keys()),
    format_func=lambda x: clinical_cases[x]["label"]
)
_ds = clinical_cases[selected_id]["default_start"]
start_mins = st.sidebar.slider(
    "推演起始点 (分钟)", min(8.0, _ds), 16.0, _ds, 0.5
)
st.sidebar.caption("起始点前的历史用于建立基线参考（模型回溯过去 10 分钟）。上限 16 分钟（需预留缓冲）。")

if st.sidebar.button("清空追踪日志", width="stretch"):
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
if clinical_cases[selected_id].get("tier") == "exploratory":
    st.warning(
        "⚠ 探索性类别：该心律类型训练/验证数据受限（验证集样本为 0），"
        "下方预测仅用于展示模型能力边界，**不构成有效诊断主张**。"
    )

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
# Class scheme imported from constants.py (single source of truth)
# =========================================================


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

if st.sidebar.button("启动数据推流", width="stretch"):
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    display_len, step_size, refresh_rate = 1000, 20, 0.08

    prob_history = deque(maxlen=60)      # 60 帧概率历史 (~60s), 环形缓冲
    pred_class_prev = 0
    cur_class = 0
    uncertainty_info = {
        "confidence": "unknown",
        "risk_score": 0.0,
        "risk_mean": 0.0,
        "risk_std": 0.0,
        "vote_count": 0,
        "ensemble_size": 3,
    }

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
                pred_class = d["pred_class"]               # future tendency
                cur_class = d.get("current_class", pred_class)  # current rhythm
                probs = d["probabilities"]
                cam_data = d.get("cam", [])
                uncertainty_info = {
                    "confidence": d.get("confidence", "unknown"),
                    "risk_score": float(d.get("risk_score", 1.0 - probs[0])),
                    "risk_mean": float(d.get("risk_mean", 1.0 - probs[0])),
                    "risk_std": float(d.get("risk_std", 0.0)),
                    "vote_count": int(d.get("future_vote_count", 0)),
                    "ensemble_size": int(d.get("ensemble_size", 1)),
                }
                prob_history.append(probs)
            except Exception:
                pred_class = pred_class_prev
                cam_data = []
        else:
            pred_class = pred_class_prev
            cam_data = []

        # ---- 预警诊断 (未来倾向) + 当前节律 ----
        diag_info = INFERENCE_MAP.get(pred_class, INFERENCE_MAP[0])
        cur_name = CLASS_NAMES[cur_class] if 0 <= cur_class < len(CLASS_NAMES) else "—"
        diag_placeholder.markdown(
            f"<div style='background:#1E1E28;padding:20px 28px;border-radius:10px;"
            f"border-left:10px solid {diag_info['color']};box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
            f"<h1 style='margin:0;color:{diag_info['color']};font-size:26px;font-weight:700;'>{diag_info['title']}</h1>"
            f"<p style='margin:6px 0 0;color:#94A3B8;font-size:14px;'>未来 2~5 分钟状态倾向预警 ｜ "
            f"当前节律：<span style='color:#CBD5E1;'>{cur_name}</span></p></div>",
            unsafe_allow_html=True
        )

        # ---- 概率分布条 ----
        if prob_history:
            probs_now = prob_history[-1]
            bars = ""
            for i, name in enumerate(CLASS_NAMES):
                bars += bar_html(name, probs_now[i], CLASS_COLORS[i])
            confidence_key = uncertainty_info["confidence"]
            confidence_label = {
                "high": "高",
                "medium": "中",
                "low": "低",
            }.get(confidence_key, "未知")
            confidence_color = {
                "high": "#10B981",
                "medium": "#F59E0B",
                "low": "#EF4444",
            }.get(confidence_key, "#94A3B8")
            uncertainty_html = (
                "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;"
                "margin-bottom:12px;font-size:12px;'>"
                f"<div><span style='color:#94A3B8;'>正式集成风险</span><br>"
                f"<b style='color:#E2E8F0;'>{uncertainty_info['risk_score']:.1%}</b></div>"
                f"<div><span style='color:#94A3B8;'>成员风险均值</span><br>"
                f"<b style='color:#E2E8F0;'>{uncertainty_info['risk_mean']:.1%}</b></div>"
                f"<div><span style='color:#94A3B8;'>模型分歧 σ</span><br>"
                f"<b style='color:#E2E8F0;'>{uncertainty_info['risk_std']:.3f}</b></div>"
                f"<div><span style='color:#94A3B8;'>置信等级</span><br>"
                f"<b style='color:{confidence_color};'>{confidence_label} "
                f"({uncertainty_info['vote_count']}/{uncertainty_info['ensemble_size']})</b></div>"
                "</div>"
                "<div style='color:#64748B;font-size:10px;margin-bottom:10px;'>"
                "置信等级仅表示三个模型的一致程度，不代表临床诊断置信区间。</div>"
            )
            class_placeholder.markdown(
                f"<div style='background:#1E1E28;padding:12px 16px;border-radius:6px;'>"
                f"{uncertainty_html}{bars}</div>",
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
                idx = e.get("class", 0)
                c = CLASS_COLORS[idx] if 0 <= idx < len(CLASS_COLORS) else "#94A3B8"
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
    chart_placeholder.line_chart(np.full(1000, 0.0), height=200, width="stretch")
