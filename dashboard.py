import streamlit as st
import requests
import numpy as np
import wfdb
import time
import pandas as pd
import altair as alt
from scipy import signal
from scipy.signal import butter, filtfilt

clinical_cases = {
    "100": {"bed_label": "01号监测源 (基线稳态)", "desc": "✅ 稳态信号：周期性去极化波形规整，时序特征处于常态区间。"},
    "119": {"bed_label": "02号监测源 (局部形态偏离)", "desc": "⚠️ 形态畸变：观测到形态偏离基线，呈现典型异位搏动特征。"},
    "201": {"bed_label": "03号监测源 (节律随机变异)", "desc": "🔍 节律失序：RR间期绝对不齐，基线伴随高频扰动特征。"},
    "208": {"bed_label": "04号监测源 (演变趋势偏离)", "desc": "🚨 持续畸变趋势：检测到隐匿性波形畸变前驱，未来信号失稳风险极高。"},
    "203": {"bed_label": "05号监测源 (极端形态解构)", "desc": "🛑 信号严重偏移：波形形态发生极端解构，电生理机制发生严重偏离。"}
}

st.set_page_config(page_title="时序信号分析中心", layout="wide")

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

st.sidebar.title("📡 信号监测控制台")
selected_id = st.sidebar.selectbox("选择监测源", list(clinical_cases.keys()))
case = clinical_cases[selected_id]

st.sidebar.divider()
st.sidebar.subheader("⏳ 时间轴与记录控制")
start_mins = st.sidebar.slider("设定起始点 (分钟)", 10.0, 15.0, 10.0, 0.1)
base_offset_pts = int(start_mins * 60 * 250)

if st.sidebar.button("🗑️ 清空异常记录", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.toast("日志已清空")

if st.sidebar.button("🧹 清除数据缓存", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.title(f"📊 动态分析终端 - {case['bed_label']}")
st.markdown("##### ⚡ **核心引擎：5 分钟形态失稳超前预警系统**")

status_container = st.empty()
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
metric_time, metric_risk, metric_status, metric_info = col1.empty(), col2.empty(), col3.empty(), col4.empty()

st.subheader("📡 信号示波器 (锁定坐标系)")
chart_placeholder = st.empty()

st.divider()
st.subheader("📑 形态偏移事件日志")
log_placeholder = st.empty()

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    raw_ecg = record.p_signal[:, 0]
    clean_ecg = clean_ecg_signal(raw_ecg, fs=360)
    ecg_250hz = signal.resample(clean_ecg, int(len(clean_ecg) * (250 / 360)))
    return ecg_250hz

data_source = load_sim_data(selected_id)

if st.sidebar.button("🔴 启动超前特征监测", use_container_width=True):
    st.sidebar.success("监测引擎已激活")
    
    if 'sim_step' not in st.session_state or 'last_offset' not in st.session_state or st.session_state.last_offset != base_offset_pts:
        st.session_state.sim_step = 0
        st.session_state.last_offset = base_offset_pts
    
    display_len = 1250 
    
    while True:
        current_pts = base_offset_pts + (st.session_state.sim_step * 50)
        if current_pts >= len(data_source) - 1:
            st.warning("已到达记录末尾。")
            break
            
        current_view = data_source[current_pts - display_len : current_pts]
        full_win_for_ai = data_source[current_pts - 150000 : current_pts]

        df_plot = pd.DataFrame({'Time': np.arange(len(current_view)), 'Signal': current_view})
        line_chart = alt.Chart(df_plot).mark_line(strokeWidth=2, color='#00FF41', interpolate='linear').encode(
            x=alt.X('Time', axis=None), 
            y=alt.Y('Signal', scale=alt.Scale(domain=[-4, 4]), axis=alt.Axis(title="幅度"))
        ).properties(width='container', height=300).configure_view(strokeOpacity=0)
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        total_seconds = current_pts / 250
        time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
        metric_time.metric("当前时间点", time_str)

        if st.session_state.sim_step % 30 == 0:
            try:
                resp = requests.post("http://127.0.0.1:8001/api/predict", json={"ecg": full_win_for_ai.tolist()}).json()
                raw_prob, alert_level = resp['future_risk_prob'], resp['alert_level']
                
                # 🎯 视觉映射：因为 0.60 以下全是 Normal，前端按照 0.60 进行底噪平滑
                display_prob = (raw_prob / 0.60) * 0.20 if alert_level == "Normal" else raw_prob

                metric_risk.metric("形态失稳概率", f"{display_prob*100:.1f}%")
                metric_status.metric("特征分级", alert_level)
                
                if alert_level != "Normal":
                    last_time = st.session_state.anomaly_logs[-1]['时间'] if st.session_state.anomaly_logs else ""
                    if last_time != time_str:
                        st.session_state.anomaly_logs.append({
                            "时间": time_str, "分级": alert_level, "形态描述": "时序特征偏离基线", "概率": f"{display_prob*100:.1f}%"
                        })

                if st.session_state.anomaly_logs:
                    log_placeholder.table(pd.DataFrame(st.session_state.anomaly_logs).iloc[::-1])

                msg, bg = {
                    "High": ("🚨 **高度形态畸变预警**", "#FF0000"),
                    "Warning": ("⚠️ **中度形态偏移提示**", "#FF8C00"),
                    "Normal": ("✅ **信号基线平稳**", "#00A36C")
                }[alert_level]
                
                metric_info.markdown(msg)
                status_container.markdown(f'<div style="background-color:{bg}; padding:15px; border-radius:10px; color:white; text-align:center;"><h2>预警状态：{alert_level.upper()}</h2></div>', unsafe_allow_html=True)
            except Exception: pass

        st.session_state.sim_step += 1
        time.sleep(0.05)