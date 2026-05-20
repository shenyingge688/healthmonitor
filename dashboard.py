"""
Script: dashboard.py
Version: V10.4 (Master Clinical UI - Pure Signal Edition)
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

st.set_page_config(page_title="临床动力学终端", layout="wide")

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

st.sidebar.title("📡 V10 临床监测台")
clinical_cases = {
    "100": {"bed_label": "档案 100 (基线平稳)", "desc": "主要包含正常心搏 (N)，基线平稳。"},
    "119": {"bed_label": "档案 119 (室早负荷)", "desc": "包含室性早搏 (V) 波形。"},
    "201": {"bed_label": "档案 201 (房颤演变)", "desc": "RR间期绝对不齐，呈现房颤特征。"},
    "207": {"bed_label": "档案 207 (室速/室扑)", "desc": "短阵室性心动过速与室扑交替发作。"},
    "209": {"bed_label": "档案 209 (房性激惹)", "desc": "存在阵发性室上速/房速发作特征。"},
    "PROSIM_01": {"bed_label": "外部硬件源 (ProSim 200)", "desc": "示波器直连：硬件模拟生理电信号。"}
}

selected_id = st.sidebar.selectbox("选择监测源", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
start_mins = st.sidebar.slider("设定推演起始点 (分钟)", 10.0, 25.0, 10.0, 0.5)
st.sidebar.caption("⏳ **医学提示**: V10 引擎需截取前 10 分钟信号建立动力学基线 (Burn-in)，推演强制从第 10 分钟起步。")

if st.sidebar.button("🗑️ 清空追踪日志", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    if rec_id == "PROSIM_01":
        if os.path.exists("prosim_custom_signal.npy"):
            return np.load("prosim_custom_signal.npy")
        else:
            return np.zeros(300000)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', 'mitdb', rec_id)
    if os.path.exists(rec_path + ".dat"): record = wfdb.rdrecord(rec_path, sampto=300000)
    else: record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    return signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), 250, 360)

data_source = load_sim_data(selected_id)

st.title(f"📊 动态推演分析终端 - {clinical_cases[selected_id]['bed_label']}")
st.info(f"**【病史/体征】** {clinical_cases[selected_id]['desc']}")

warning_box = st.empty()
api_status_box = st.empty() 

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📡 实时心电波形")
    chart_placeholder = st.empty()
    st.markdown("##### 📈 恶化轨迹 (TTE Time-to-Alarm)")
    traj_placeholder = st.empty()
with col2:
    st.subheader("⏱️ 推演时间")
    metric_time = st.empty()
    st.subheader("🚨 综合演化指数")
    metric_risk = st.empty()

st.divider()
st.subheader("📑 并发事件日志")
log_placeholder = st.empty()

def build_prob_column_html(title, prob_list):
    html = f"<div style='flex: 1; min-width: 260px; margin-right: 15px;'>"
    html += f"<p style='margin:0 0 10px 0; font-weight: 600; color: #A0AEC0; font-size: 13px;'>{title}</p>"
    for item in prob_list:
        p_val = min(item['prob'], 1.0)
        w = p_val * 100
        c = item['color'] if p_val > 0.05 else "rgba(255,255,255,0.15)"
        text_c = "#E2E8F0" if p_val > 0.05 else "#64748B"
        font_w = "600" if p_val > 0.05 else "400"
        
        html += f"<div style='display:flex; align-items:center; margin-bottom:6px; color:{text_c}; font-size: 12px;'>"
        html += f"<span style='width:130px; font-weight:{font_w}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{item['label']}</span>"
        html += f"<div style='flex-grow:1; background:rgba(0,0,0,0.3); height:6px; margin:0 10px; border-radius:3px; overflow:hidden;'>"
        html += f"<div style='width:{w}%; background:{c}; height:100%; border-radius:3px; transition: width 0.3s ease;'></div>"
        html += "</div>"
        html += f"<span style='width:45px; text-align:right; font-family:monospace;'>{w:.1f}%</span>"
        html += "</div>"
    html += "</div>"
    return html

if st.sidebar.button("🔴 启动数据推流", use_container_width=True):
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    display_len, step_size, refresh_rate = 1000, 20, 0.08  
    
    rhythm_data = [1.0, 0.0, 0.0, 0.0]
    crit_data = [1.0, 0.0, 0.0, 0.0]
    hazard_data = [0.0, 0.0, 0.0]
    last_risk = 0.0
    risk_history = deque([0.0] * 10, maxlen=10)  # 最近 10 次真实风险值
    cached_warning_html = ""

    while True:
        current_pts = base_offset_pts + (sim_step * step_size)
        if current_pts >= len(data_source) - 1: break

        new_data = data_source[current_pts - step_size : current_pts]
        idx = (sim_step * step_size) % display_len
        ecg_buffer[idx : idx + step_size] = new_data
        
        gap_end = (idx + step_size + 25) % display_len
        if gap_end > (idx + step_size): ecg_buffer[idx + step_size : gap_end] = np.nan
        else: ecg_buffer[idx + step_size : display_len] = np.nan; ecg_buffer[0 : gap_end] = np.nan

        df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': ecg_buffer})
        line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
            x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None), 
            y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)", grid=True)) 
        ).properties(height=200).configure_view(strokeOpacity=0) 
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        cur_sec = current_pts / 250
        metric_time.metric("当前扫描线", f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}")

        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 150000) : current_pts]
            if len(full_win) < 150000: full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')
            
            try:
                # 🚀 已剥离 Z-score，直接输送原始带偏置的物理数据
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win.tolist()}, timeout=3)
                
                if resp.status_code == 200:
                    api_status_box.empty() 
                    data = resp.json()
                    
                    if 'rhythm' in data:
                        rhythm_data = data.get('rhythm', [1.0, 0.0, 0.0, 0.0])
                        crit_data = data.get('criticality', [1.0, 0.0, 0.0, 0.0])
                        hazard_data = data.get('hazard', [0.0, 0.0, 0.0])
                        last_risk = hazard_data[-1]
                        # 真实轨迹：记录每次 API 返回的 5 分钟风险值
                        current_risk = float(data.get('risk_trajectory', 0.0))
                        risk_history.append(current_risk)
                    else:
                        api_status_box.warning("⚠️ 检测到旧版 FastAPI 响应格式。请更新 backend 以获得完整三维输出！")

                    # 绘制最近 10 次风险历史
                    traj_list = list(risk_history)
                    time_axis = np.linspace(-len(traj_list) + 1, 0, len(traj_list))
                    df_traj = pd.DataFrame({'Time': time_axis, 'Risk': traj_list})
                    traj_chart = alt.Chart(df_traj).mark_area(
                        color=alt.Gradient(gradient='linear', stops=[
                            alt.GradientStop(color='rgba(239, 68, 68, 0.1)', offset=0),
                            alt.GradientStop(color='rgba(239, 68, 68, 0.8)', offset=1)
                        ]),
                        line={'color': '#EF4444'}
                    ).encode(
                        x=alt.X('Time:Q', axis=alt.Axis(title="API 调用历史 (次)", grid=True)),
                        y=alt.Y('Risk:Q', scale=alt.Scale(domain=[0, 1.0]), axis=alt.Axis(title="5分钟风险", format='%'))
                    ).properties(height=120).configure_view(strokeOpacity=0)
                    traj_placeholder.altair_chart(traj_chart, use_container_width=True)
                else:
                    api_status_box.error(f"❌ 引擎连接异常 (Status Code: {resp.status_code})")
            except Exception as e:
                api_status_box.error(f"❌ 推理引擎断开连接: {e}")
        
        metric_risk.metric("5分钟恶化概率", f"{last_risk * 100:.1f}%")

        rhy_list = [
            {"label": "正常稳态 (Normal)", "prob": rhythm_data[0], "color": "#10B981"},
            {"label": "室早负荷 (PVC)", "prob": rhythm_data[1], "color": "#F59E0B"},
            {"label": "房颤演变 (AFib)", "prob": rhythm_data[2], "color": "#F97316"},
            {"label": "房性激惹 (SVT/AT)", "prob": rhythm_data[3], "color": "#D946EF"}
        ]
        
        cri_list = [
            {"label": "安全界限 (Safe)", "prob": crit_data[0], "color": "#10B981"},
            {"label": "异位负荷 (PVC-Load)", "prob": crit_data[1], "color": "#F59E0B"},
            {"label": "室速威胁 (VT)", "prob": crit_data[2], "color": "#EF4444"},
            {"label": "心室颤动 (VF)", "prob": crit_data[3], "color": "#991B1B"}
        ]
        
        haz_list = [
            {"label": "30秒 崩溃预警", "prob": hazard_data[0], "color": "#FCA5A5"},
            {"label": "1分钟 崩溃预警", "prob": hazard_data[1], "color": "#F87171"},
            {"label": "5分钟 崩溃预警", "prob": hazard_data[2], "color": "#DC2626"}
        ]

        # —— 概率融合警告逻辑 ——
        # 核心思想：条件概率 P(崩溃) = P(危急度异常) × P(5分钟风险)
        # hazard 只有在 criticality 或 rhythm 有异常信号时才被放大
        p_rhythm_abnormal = 1.0 - rhythm_data[0]       # 非 Normal 的概率
        p_criticality_abnormal = 1.0 - crit_data[0]     # 非 Safe 的概率
        p_hazard_5m = hazard_data[2]                    # 5 分钟崩溃概率

        # 综合有效风险：条件概率融合
        # 公式：effective_risk = P_crit_abnormal × P_hazard
        # 这保证了正常档案即使 hazard 偶然偏高也不会触发误报
        effective_risk = p_criticality_abnormal * p_hazard_5m

        # 对极端高 hazard 但无背书的兜底（防止罕见漏报）
        if p_hazard_5m > 0.90 and effective_risk < 0.30:
            effective_risk = 0.35

        # 等级映射
        if effective_risk > 0.50:
            diag_title = "💀 系统性崩溃预兆"
            diag_desc = f"危急度异常概率 {p_criticality_abnormal*100:.0f}% × 5分钟风险 {p_hazard_5m*100:.0f}% → 综合风险极高。请立即确认生命体征！"
            accent_color = "#991B1B"
        elif effective_risk > 0.25:
            diag_title = "🚨 极危: 室速/室颤威胁"
            diag_desc = f"检测到高危急度信号与显著短期崩溃风险。综合风险评分: {effective_risk*100:.1f}%"
            accent_color = "#EF4444"
        elif effective_risk > 0.10:
            if rhythm_data[2] > 0.30:
                diag_title = "⚠️ 心房颤动 / 节律异常"
                diag_desc = "检测到心房异常激惹，需防范远期血栓与心衰风险。"
                accent_color = "#F97316"
            elif crit_data[1] > 0.30:
                diag_title = "⚠️ 显著室性异位搏动"
                diag_desc = "室早负荷升高，可能诱发更严重的心律失常。"
                accent_color = "#F59E0B"
            else:
                diag_title = "⚠️ 轻度异常信号"
                diag_desc = "节律或危急度出现轻微偏离，建议持续观察。"
                accent_color = "#F59E0B"
        elif effective_risk > 0.03:
            diag_title = "⚡ 轻微异位搏动"
            diag_desc = "偶发异位心搏，生命体征总体平稳，继续监测。"
            accent_color = "#3B82F6"
        else:
            diag_title = "未见急症指征"
            diag_desc = "生命体征平稳，未见明显血液动力学恶化趋势。"
            accent_color = "#10B981"

        dist_html = "<div style='display: flex; flex-wrap: wrap; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;'>"
        dist_html += build_prob_column_html("🎯 节律中枢 (Rhythm)", sorted(rhy_list, key=lambda x: x['prob'], reverse=True))
        dist_html += build_prob_column_html("🫀 危急评估 (Criticality)", sorted(cri_list, key=lambda x: x['prob'], reverse=True))
        dist_html += build_prob_column_html("⚠️ 生存预警 (Hazard)", haz_list)
        dist_html += "</div>"
            
        current_warning_html = f"""
        <div style="background-color: #1E1E28; padding: 20px 25px; border-radius: 8px; border-left: 8px solid {accent_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin:0; font-size: 20px; color: {accent_color};">{diag_title}</h2>
                    <h3 style="margin: 4px 0 0 0; font-weight: 400; font-size: 13px; color: #94A3B8;">{diag_desc}</h3>
                </div>
            </div>
            {dist_html}
        </div>
        """
        
        if current_warning_html != cached_warning_html:
            warning_box.markdown(current_warning_html, unsafe_allow_html=True)
            cached_warning_html = current_warning_html

        sim_step += 1
        time.sleep(refresh_rate)
else:
    warning_box.info("👈 请在左侧选择病例并点击【启动数据推流】开始推演。")
    chart_placeholder.line_chart(np.full(1000, 0.0), height=200, use_container_width=True)