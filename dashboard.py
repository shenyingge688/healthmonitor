"""
模块名称：AI 时序信号超前预警控制台 (Clinical Dashboard)
模块功能：
    1. 基于 Streamlit 构建交互式医疗 UI。
    2. 模拟真实心电监护仪的“扫描覆盖(Sweep)”推流模式。
    3. 异步向后端请求 AI 推理结果，并动态渲染超前预警看板。
    4. 解析后端传回的 1D-CAM 数据，绘制过去 10 分钟的“病灶溯源热力图”。
    5. 实现临床级事件状态机，自动合并连续警报并追踪峰值危险概率。
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

st.set_page_config(page_title="AI 心电超前预警中心", layout="wide")

# ==========================================
# 模块 1：临床场景定义与缓存初始化
# ==========================================
clinical_cases = {
    "100": {"bed_label": "01号监测源 (基线稳态)", "desc": "✅ 稳态信号：周期性去极化波形规整，时序特征处于常态区间。"},
    "119": {"bed_label": "02号监测源 (局部形态偏离)", "desc": "⚠️ 形态畸变：观测到形态偏离基线，呈现典型异位搏动特征。"},
    "201": {"bed_label": "03号监测源 (节律随机变异)", "desc": "🔍 节律失序：RR间期绝对不齐，基线伴随高频扰动特征。"},
    "208": {"bed_label": "04号监测源 (演变趋势偏离)", "desc": "🚨 持续畸变趋势：检测到隐匿性波形畸变前驱，未来信号失稳风险极高。"},
    "203": {"bed_label": "05号监测源 (极端形态解构)", "desc": "🛑 信号严重偏移：波形形态发生极端解构，电生理机制发生严重偏离。"}
}

# 状态管理器初始化
if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []
if 'ecg_buffer' not in st.session_state:
    st.session_state.ecg_buffer = np.full(1000, np.nan) # 模拟真实监护仪画布缓存

# ==========================================
# 模块 2：侧边栏与数据加载
# ==========================================
st.sidebar.title("📡 信号监测控制台")
selected_id = st.sidebar.selectbox("选择监测源", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
case = clinical_cases[selected_id]

st.sidebar.divider()
st.sidebar.subheader("⏳ 时间轴控制")
start_mins = st.sidebar.slider("设定起始点 (分钟)", 10.0, 15.0, 10.0, 0.1)
base_offset_pts = int(start_mins * 60 * 250)

if st.sidebar.button("🗑️ 清空异常记录", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    """离线读取 MIT-BIH 原始数据，滤波并重采样至 250Hz"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'mitdb')
    rec_path = os.path.join(data_dir, rec_id)

    record = wfdb.rdrecord(rec_path, sampto=300000) if os.path.exists(rec_path + ".dat") else wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    raw_ecg = record.p_signal[:, 0]
    return signal.resample_poly(clean_ecg_signal(raw_ecg, fs=360), 250, 360) 

data_source = load_sim_data(selected_id)

# ==========================================
# 模块 3：UI 核心视窗布局
# ==========================================
st.title(f"📊 动态分析终端 - {case['bed_label']}")
st.info(f"**【当前病患体征】** {case['desc']}")

st.markdown("### 🎯 核心中枢：未来 5 分钟形态失稳超前预警")
warning_box = st.empty()  

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📡 实时心电监护仪 (临床扫描模式)")
    chart_placeholder = st.empty()
    st.markdown("##### 🔍 10 分钟 AI 病灶溯源热力图 (Grad-CAM)")
    cam_placeholder = st.empty() # 新增：用于展示 XAI 解释性热力图
with col2:
    st.subheader("⏱️ 监测状态")
    metric_time = st.empty()
    metric_risk = st.empty()
    metric_status = st.empty()

st.divider()
st.subheader("📑 临床级异常事件追踪日志")
log_placeholder = st.empty()

# ==========================================
# 模块 4：推流引擎与主事件循环
# ==========================================
if st.sidebar.button("🔴 启动 AI 预警引擎", use_container_width=True):
    st.sidebar.success("预警引擎已激活！")
    
    st.session_state.sim_step = 0
    st.session_state.ecg_buffer = np.full(1000, np.nan)
    
    display_len = 1000  
    step_size = 20       # 降低步长，提高渲染细腻度
    refresh_rate = 0.08  # 黄金休眠间隔，平衡浏览器负载实现流畅推流
    
    display_prob = 0.0
    alert_level = "Normal"
    cam_data = []

    while True:
        current_pts = base_offset_pts + (st.session_state.sim_step * step_size)
        if current_pts >= len(data_source) - 1:
            st.warning("已到达记录末尾。")
            break

        # 1. 扫描模式环形缓冲更新
        new_data = data_source[current_pts - step_size : current_pts]
        idx = (st.session_state.sim_step * step_size) % display_len
        st.session_state.ecg_buffer[idx : idx + step_size] = new_data
        
        gap_end = (idx + step_size + 25) % display_len
        if gap_end > (idx + step_size):
            st.session_state.ecg_buffer[idx + step_size : gap_end] = np.nan
        else:
            st.session_state.ecg_buffer[idx + step_size : display_len] = np.nan
            st.session_state.ecg_buffer[0 : gap_end] = np.nan

        # 2. 锁定物理刻度渲染心电图
        df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': st.session_state.ecg_buffer})
        line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
            x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None), 
            y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)", grid=True)) 
        ).properties(height=250).configure_view(strokeOpacity=0) 
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        # 3. 时间计算
        total_seconds = current_pts / 250
        time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
        metric_time.metric("当前时间点", time_str)

        # 4. 后端 AI 接口请求 (控制请求频率防止网络拥塞)
        if st.session_state.sim_step % 12 == 0:
            full_win_for_ai = data_source[current_pts - 150000 : current_pts]
            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win_for_ai.tolist()})
                resp.raise_for_status()
                data = resp.json()
                # 接收后端的平滑预测与 CAM 热力图
                display_prob = data['future_risk_prob'] 
                alert_level = data['alert_level']
                cam_data = data.get('cam_heatmap', [])
            except requests.exceptions.ConnectionError:
                st.error("⚠️ 致命错误：无法连接到后端 AI 服务！请确认是否在另一个终端运行了 `python main.py`")
                break
            except Exception:
                pass

        # 5. XAI 溯源热力图渲染
        if cam_data:
            df_cam = pd.DataFrame({'TimeStep': np.arange(len(cam_data)), 'Attention': cam_data})
            cam_chart = alt.Chart(df_cam).mark_area(
                color=alt.Gradient(gradient='linear', stops=[
                    alt.GradientStop(color='#000000', offset=0), 
                    alt.GradientStop(color='#FF0000', offset=1)
                ])
            ).encode(
                x=alt.X('TimeStep:Q', scale=alt.Scale(domain=[0, len(cam_data)]), axis=alt.Axis(labels=False, title="过去 10 分钟溯源轴")),
                y=alt.Y('Attention:Q', scale=alt.Scale(domain=[0, 1.0]), axis=None)
            ).properties(height=60).configure_view(strokeOpacity=0)
            cam_placeholder.altair_chart(cam_chart, use_container_width=True)

        # 6. 中央预警看板动态更新
        if alert_level == "High":
            bg_color, status_title = "#FF0000", "🚨 高危畸变预警：预测未来 5 分钟内将发生严重形态失稳！"
        elif alert_level == "Warning":
            bg_color, status_title = "#FF8C00", "⚠️ 中度偏离提示：预测未来 5 分钟内可能出现形态失稳。"
        else:
            # 视觉压缩：仅在 Normal 时段映射显示，稳定安全期视觉
            # 将 [0, 0.60) 映射到 [0, 0.20)，避免界面出现 50% 仍显示绿色的误导
            display_prob = (display_prob / 0.60) * 0.20 
            bg_color, status_title = "#00A36C", "✅ 信号基线平稳：未见未来失稳风险。"

        warning_html = f"""
        <div style="background-color: {bg_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; color: white;">{status_title}</h2>
            <h4 style="margin-top: 10px; color: white; font-weight: normal;">AI 综合风险推断: {display_prob*100:.1f}%</h4>
        </div>
        """
        warning_box.markdown(warning_html, unsafe_allow_html=True)
        metric_risk.metric("综合概率", f"{display_prob*100:.1f}%")
        metric_status.metric("AI 评级", alert_level)

        # 7. 临床级事件区间折叠追踪日志
        if alert_level != "Normal":
            is_new_event = True
            if st.session_state.anomaly_logs:
                last_log = st.session_state.anomaly_logs[-1]
                if last_log["状态"] == "活跃" and last_log["分级"] == alert_level:
                    is_new_event = False
                    last_log["结束时间"] = time_str
                    current_max = float(last_log["峰值概率"].strip('%'))
                    if (display_prob * 100) > current_max:
                        last_log["峰值概率"] = f"{display_prob * 100:.1f}%"
                elif last_log["状态"] == "活跃" and last_log["分级"] != alert_level:
                    last_log["状态"] = "已结束"

            if is_new_event:
                st.session_state.anomaly_logs.append({
                    "开始时间": time_str, "结束时间": time_str,
                    "分级": alert_level, "形态描述": "超前时序特征偏离基线",
                    "峰值概率": f"{display_prob * 100:.1f}%", "状态": "活跃"
                })
        else:
            if st.session_state.anomaly_logs and st.session_state.anomaly_logs[-1]["状态"] == "活跃":
                st.session_state.anomaly_logs[-1]["状态"] = "已结束"

        if st.session_state.anomaly_logs:
            display_data = [{
                "时间区间": f"{log['开始时间']} 至 {log['结束时间']}" if log['开始时间'] != log['结束时间'] else f"{log['开始时间']} (瞬时)",
                "预警分级": log['分级'],
                "事件状态": "🔴 持续警报中" if log['状态'] == "活跃" else "⚪ 已解除",
                "期间最高峰值": log['峰值概率'],
                "临床推断": log['形态描述']
            } for log in reversed(st.session_state.anomaly_logs)]
            log_placeholder.table(pd.DataFrame(display_data))
        else:
            log_placeholder.info("✅ 当前监测时段内暂无形态畸变记录。")

        st.session_state.sim_step += 1
        time.sleep(refresh_rate)