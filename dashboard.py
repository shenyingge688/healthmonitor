"""
Script: dashboard.py
Version: V10.0 (ICU Telemetry Foundation UI)
"""
import streamlit as st
import numpy as np
import pandas as pd
import requests
import wfdb
import time
import os
import altair as alt
from scipy import signal
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="V10 层级化临床终端", layout="wide")

clinical_cases = {
    "100": {"bed_label": "档案 100 (基线平稳)", "desc": "基础节律为正常窦性，血流动力学极度稳定。"},
    "119": {"bed_label": "档案 119 (室早负荷)", "desc": "包含室性早搏，观察临界激惹状态。"},
    "201": {"bed_label": "档案 201 (房颤演变)", "desc": "RR间期绝对不齐，典型房颤。"},
    "207": {"bed_label": "档案 207 (室速发作)", "desc": "极危：持续室性心动过速 (VT)。"},
    "209": {"bed_label": "档案 209 (房速/SVT)", "desc": "室上性激惹，快心率但不伴随室性失代偿。"},
    "PROSIM_01": {"bed_label": "外部硬件源 (ProSim)", "desc": "示波器直连。"}
}

st.sidebar.title("📡 V10 监测台")
selected_id = st.sidebar.selectbox("选择监测源", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
start_mins = st.sidebar.slider("设定起始点 (分钟)", 0.0, 15.0, 10.0, 0.1)

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    if rec_id == "PROSIM_01":
        if os.path.exists("prosim_custom_signal.npy"): return np.load("prosim_custom_signal.npy")
        else: return np.zeros(300000)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', 'mitdb', rec_id)
    if os.path.exists(rec_path + ".dat"): record = wfdb.rdrecord(rec_path, sampto=300000)
    else: record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    return signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), 250, 360)

data_source = load_sim_data(selected_id)

st.title(f"🏥 层级化临床动力学终端 - {clinical_cases[selected_id]['bed_label']}")
st.info(f"**【临床特征】** {clinical_cases[selected_id]['desc']}")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("### 🫀 1. 基础组织化节律 (Rhythm)")
    rhythm_box = st.empty()
with col2:
    st.markdown("### ⚠️ 2. 心室危急度 (Criticality)")
    crit_box = st.empty()
with col3:
    st.markdown("### 📉 3. 离散生存风险 (Hazard)")
    hazard_box = st.empty()

st.divider()
st.subheader("📡 实时心电波形流")
chart_placeholder = st.empty()

if st.sidebar.button("🔴 启动数据推流", use_container_width=True):
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    
    while True:
        current_pts = base_offset_pts + (sim_step * 20)
        if current_pts >= len(data_source) - 1: break

        new_data = data_source[current_pts - 20 : current_pts]
        idx = (sim_step * 20) % 1000
        ecg_buffer[idx : idx + 20] = new_data
        
        gap_end = (idx + 45) % 1000
        if gap_end > (idx + 20): ecg_buffer[idx + 20 : gap_end] = np.nan
        else: ecg_buffer[idx + 20 : 1000] = np.nan; ecg_buffer[0 : gap_end] = np.nan

        df_plot = pd.DataFrame({'Point': np.arange(1000), 'Signal': ecg_buffer})
        line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
            x=alt.X('Point:Q', scale=alt.Scale(domain=[0, 1000]), axis=None), 
            y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)"))
        ).properties(height=200).configure_view(strokeOpacity=0) 
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 150000) : current_pts]
            if len(full_win) < 150000: full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')
            
            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win.tolist()}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # 1. 渲染 Rhythm Panel
                    r_labels = ["正常窦性 (Normal)", "室性负荷 (PVC)", "心房颤动 (AFIB)", "室上性激惹 (SVT/AT)"]
                    r_idx = np.argmax(data['rhythm_probs'])
                    r_color = "#10B981" if r_idx == 0 else "#F59E0B" if r_idx == 1 else "#F97316"
                    rhythm_box.markdown(f"<div style='background:#1E1E28; padding:20px; border-radius:8px; border-left:6px solid {r_color};'><h3 style='margin:0;color:{r_color};'>{r_labels[r_idx]}</h3><p style='margin:5px 0 0 0;color:#94A3B8;'>当前主导节律置信度: <strong style='color:#E2E8F0;'>{data['rhythm_probs'][r_idx]*100:.1f}%</strong></p></div>", unsafe_allow_html=True)
                    
                    # 2. 渲染 Criticality Panel
                    c_labels = ["代偿稳态 (Stable)", "室性不稳定 (Instability)", "室性心动过速 (Sustained VT)", "极危崩溃 (VF)"]
                    c_idx = np.argmax(data['crit_probs'])
                    c_color = "#10B981" if c_idx == 0 else "#F59E0B" if c_idx == 1 else "#EF4444"
                    crit_box.markdown(f"<div style='background:#1E1E28; padding:20px; border-radius:8px; border-left:6px solid {c_color};'><h3 style='margin:0;color:{c_color};'>{c_labels[c_idx]}</h3><p style='margin:5px 0 0 0;color:#94A3B8;'>血流动力学危急度: <strong style='color:#E2E8F0;'>{data['crit_probs'][c_idx]*100:.1f}%</strong></p></div>", unsafe_allow_html=True)
                    
                    # 3. 渲染 Hazard Panel
                    hz = data['hazard_probs']
                    hz_color = "#EF4444" if hz[2] > 0.5 else "#10B981"
                    hazard_html = f"""
                    <div style='background:#1E1E28; padding:15px 20px; border-radius:8px; border-left:6px solid {hz_color};'>
                        <table style='width:100%; text-align:left; font-size:15px; color:#E2E8F0;'>
                            <tr style='color:#94A3B8; border-bottom:1px solid #333;'>
                                <th style='padding-bottom:5px;'>预测视界</th><th style='padding-bottom:5px;text-align:right;'>崩溃累积概率</th>
                            </tr>
                            <tr><td style='padding-top:8px;'>30 秒内</td><td style='padding-top:8px;text-align:right;color:{"#EF4444" if hz[0]>0.2 else "#10B981"};font-weight:bold;'>{hz[0]*100:.1f}%</td></tr>
                            <tr><td style='padding-top:5px;'>1 分钟内</td><td style='padding-top:5px;text-align:right;color:{"#EF4444" if hz[1]>0.3 else "#10B981"};font-weight:bold;'>{hz[1]*100:.1f}%</td></tr>
                            <tr><td style='padding-top:5px;'>5 分钟内</td><td style='padding-top:5px;text-align:right;color:{"#EF4444" if hz[2]>0.5 else "#10B981"};font-weight:bold;'>{hz[2]*100:.1f}%</td></tr>
                        </table>
                    </div>
                    """
                    hazard_box.markdown(hazard_html, unsafe_allow_html=True)
                    
            except Exception: pass

        sim_step += 1
        time.sleep(0.08)
else:
    chart_placeholder.info("👈 请在左侧侧边栏启动数据推流。")