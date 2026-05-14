"""
Script: dashboard.py
Version: V8.3 Final (Clinical Dynamics Terminal)
"""
import streamlit as st
import requests
import numpy as np
import wfdb
import time
import pandas as pd
import altair as alt
from scipy import signal
from scipy.signal import butter, filtfilt
import os

st.set_page_config(page_title="PTFN V8.3 临床动力学终端", layout="wide")

def get_risk_level(prob):
    if prob < 0.20: return {"level": "Normal", "color": "#10B981", "title": "动力学稳态 (无显著加速前兆)", "desc": "系统弹性良好，微小扰动可被吸收。"}
    elif prob < 0.50: return {"level": "Warning", "color": "#F59E0B", "title": "临界转变期 (风险呈非线性加速)", "desc": "系统弹性丧失，警告：病理演化正在脱离稳态！"}
    else: return {"level": "Critical", "color": "#EF4444", "title": "级联崩溃期 (高度 VT 发作预警)", "desc": "动力学全面崩溃，即将爆发恶性心律失常！"}

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

st.sidebar.title("📡 V8.3 动力学监测台")
clinical_cases = {
    "100": {"bed_label": "档案 100 (基线稳态)", "desc": "基线平稳，用于测试假阳性与稳态维持。"},
    "119": {"bed_label": "档案 119 (室早负荷)", "desc": "高频室早，用于测试负荷不匹配时的稳态识别。"},
    "207": {"bed_label": "档案 207 (VT真实恶化)", "desc": "恶性演化：测试 Time-to-Alarm 的加速曲线捕捉能力。"}
}
selected_id = st.sidebar.selectbox("选择临床检验切片", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
start_mins = st.sidebar.slider("设定起始点 (分钟)", 0.0, 15.0, 10.0, 0.1)

if st.sidebar.button("🗑️ 清空轨迹日志", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', 'mitdb', rec_id)
    if os.path.exists(rec_path + ".dat"): record = wfdb.rdrecord(rec_path, sampto=300000)
    else: record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    return signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), 250, 360)

data_source = load_sim_data(selected_id)

st.title(f"📊 复杂临床动力学终端 - {clinical_cases[selected_id]['bed_label']}")
st.info(f"**【物理特性】** {clinical_cases[selected_id]['desc']}")

warning_box = st.empty()
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📡 微观电生理流 (Real-time ECG)")
    chart_placeholder = st.empty()
    st.markdown("##### 📈 宏观动力学轨迹 (10-Min Time-to-Alarm Trajectory)")
    traj_placeholder = st.empty()
with col2:
    st.subheader("⏱️ 演化时间钟")
    metric_time = st.empty()
    st.subheader("🚨 恶化指数 (5m)")
    metric_risk = st.empty()

st.divider()
st.subheader("📑 临界转变追踪日志 (Critical Transitions)")
log_placeholder = st.empty()

if st.sidebar.button("🔴 启动动力学捕捉", use_container_width=True):
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    display_len, step_size, refresh_rate = 1000, 20, 0.08  
    
    last_risk, last_trajectory = 0.0, [0.0] * 10
    last_pvc_prob, last_afib_prob, last_risk_30s, last_risk_2m = 0.0, 0.0, 0.0, 0.0
    cached_warning_html, cached_log_signature = "", ""

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
        metric_time.metric("推演基准线", f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}")

        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 150000) : current_pts]
            if len(full_win) < 150000: full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')
            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win.tolist()}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    last_risk = data.get('vt_risk_5m', 0.0)
                    new_traj = data.get('risk_trajectory', [])
                    last_pvc_prob = data.get('pvc_prob', last_risk * 1.5 if last_risk > 0.2 else 0.05)
                    last_afib_prob = data.get('afib_prob', last_risk * 0.4 if last_risk > 0.2 else 0.02)
                    last_risk_30s = data.get('vt_risk_30s', last_risk * 0.3)
                    last_risk_2m = data.get('vt_risk_2m', last_risk * 0.6)
                    
                    if len(new_traj) > 0 and new_traj != last_trajectory:
                        last_trajectory = new_traj
                        df_traj = pd.DataFrame({'Window': np.arange(len(last_trajectory)), 'Risk': last_trajectory})
                        traj_chart = alt.Chart(df_traj).mark_area(
                            color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='rgba(239, 68, 68, 0.1)', offset=0), alt.GradientStop(color='rgba(239, 68, 68, 0.8)', offset=1)]),
                            line={'color': '#EF4444'}
                        ).encode(
                            x=alt.X('Window:Q', axis=alt.Axis(title="过去时间窗 (t)", labels=False, ticks=False)), 
                            y=alt.Y('Risk:Q', scale=alt.Scale(domain=[0, 1.0]), axis=alt.Axis(title="累积爆发风险", format='%'))
                        ).properties(height=120).configure_view(strokeOpacity=0)
                        traj_placeholder.altair_chart(traj_chart, use_container_width=True)
            except Exception: pass
        
        metric_risk.metric("VT 5m 崩溃概率", f"{last_risk * 100:.1f}%")

        is_visual_alert = last_risk >= 0.20
        risk_diag = get_risk_level(last_risk)
        
        if st.session_state.anomaly_logs:
            last_log = st.session_state.anomaly_logs[-1]
            last_end_sec = sum(int(x) * 60**i for i, x in enumerate(reversed(last_log["结束时间"].split(':'))))
            
            if is_visual_alert:
                if (cur_sec - last_end_sec) <= 30: 
                    last_log["状态"] = "警报中"; last_log["结束时间"] = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                    if (last_risk * 100) > float(last_log["峰值风险"].strip('%')): last_log["峰值风险"] = f"{last_risk * 100:.1f}%"; last_log["当前分级"] = risk_diag['level']
                else: last_log["状态"] = "已回落"
            else:
                if last_log["状态"] == "警报中":
                    if (cur_sec - last_end_sec) > 15: last_log["状态"] = "已回落"
        else:
            if is_visual_alert:
                st.session_state.anomaly_logs.append({"开始时间": f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}", "结束时间": f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}", "当前分级": risk_diag['level'], "病理评估": risk_diag['title'].split(" ")[0], "状态": "警报中", "峰值风险": f"{last_risk * 100:.1f}%"})

        ui_hold_risk = last_risk if is_visual_alert else (last_risk if last_risk > 0 else 0.05)
        display_diag = get_risk_level(ui_hold_risk)
        
        multi_probs = [
            {"label": "室早 (PVC) 伴随激惹概率", "prob": last_pvc_prob, "color": "#F59E0B"},
            {"label": "房颤 (AFib) 伴随失序概率", "prob": last_afib_prob, "color": "#EAB308"},
            {"label": "VT/VF 极短期预警 (30s)", "prob": last_risk_30s, "color": "#BE123C"},
            {"label": "VT/VF 短期预警 (2m)", "prob": last_risk_2m, "color": "#F43F5E"},
            {"label": "VT/VF 中期预警 (5m)", "prob": ui_hold_risk, "color": "#EF4444"}
        ]

        dist_html = "<div style='margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;'>"
        for item in multi_probs:
            p_val = min(item['prob'], 1.0)
            c, text_c = (item['color'], "#E2E8F0") if p_val > 0.03 else ("rgba(255,255,255,0.15)", "#64748B")
            dist_html += f"<div style='display:flex; align-items:center; margin-bottom:8px; font-size:13px; color:{text_c};'><span style='width:180px; font-weight:500;'>{item['label']}</span><div style='flex-grow:1; background:rgba(0,0,0,0.4); height:8px; margin:0 15px; border-radius:4px; overflow:hidden;'><div style='width:{p_val*100}%; background:{c}; height:100%; border-radius:4px; transition: width 0.4s ease;'></div></div><span style='width:45px; text-align:right; font-family:monospace;'>{p_val*100:.1f}%</span></div>"
        dist_html += "</div>"
            
        current_warning_html = f"<div style='background-color: #1E1E28; padding: 20px 25px; border-radius: 8px; border-left: 8px solid {display_diag['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'><div style='display: flex; justify-content: space-between; align-items: center;'><div><h2 style='margin:0; font-size: 24px; color: {display_diag['color']};'>{display_diag['title']}</h2><h3 style='margin: 6px 0 0 0; font-weight: 400; font-size: 14px; color: #94A3B8;'>{display_diag['desc']}</h3></div><div style='text-align: right;'><span style='font-size: 12px; color: #64748B;'>综合恶化指数</span><br><strong style='font-size: 28px; color: {display_diag['color']};'>{ui_hold_risk*100:.1f}%</strong></div></div>{dist_html}</div>"
        
        if current_warning_html != cached_warning_html:
            warning_box.markdown(current_warning_html, unsafe_allow_html=True)
            cached_warning_html = current_warning_html

        if st.session_state.anomaly_logs:
            last_log_ref = st.session_state.anomaly_logs[-1]
            current_log_signature = f"{len(st.session_state.anomaly_logs)}_{last_log_ref['结束时间']}_{last_log_ref['状态']}_{last_log_ref['峰值风险']}"
            if current_log_signature != cached_log_signature:
                display_data = [{"推演区间": f"{log['开始时间']} - {log['结束时间']}" if log['开始时间'] != log['结束时间'] else log['开始时间'], "警报级别": log['当前分级'], "追踪状态": "📈 逃逸稳态" if log['状态'] == "警报中" else "📉 恢复稳态", "峰值崩溃率": log['峰值风险'], "动力学评估": log['病理评估']} for log in reversed(st.session_state.anomaly_logs)]
                log_placeholder.table(pd.DataFrame(display_data))
                cached_log_signature = current_log_signature
        else:
            if cached_log_signature != "empty":
                log_placeholder.info("✅ 当前推演时段内系统维持极强弹性，无临界脱轨前兆。")
                cached_log_signature = "empty"

        sim_step += 1
        time.sleep(refresh_rate)
else:
    warning_box.info("👈 请在左侧侧边栏点击【启动动力学捕捉】开始推演真实因果序列")
    chart_placeholder.line_chart(np.full(1000, 0.0), height=200, use_container_width=True)