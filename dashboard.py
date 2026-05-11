"""
HealthMonitor  - 前端实时渲染与控制台模块 
功能描述：
1. 丝滑推流： while True 内部推流架构，彻底杜绝 st.rerun 导致的全屏闪烁。
2. 渲染阻断 (Memoization)：在 while 循环中，若预警状态未改变，强行阻断 HTML 与 CAM 的重绘，极大降低 CPU 负载。
3. 临床级防抖状态机：完美集成 30 秒事件聚合与 15 秒解除缓冲期。
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

st.set_page_config(page_title="HealthMonitor 心电预警终端", layout="wide")

# ==========================================
# 核心配置与状态字典
# ==========================================
INFERENCE_MAP = {
    0: {"level": "Normal", "color": "#10B981", "title": "信号平稳，未见明显节律异常"},
    1: {"level": "Warning", "color": "#F59E0B", "title": "提示频发室性早搏 (PVC) 风险"},
    2: {"level": "High", "color": "#F97316", "title": "检测到心房颤动 (AFib) 特征"},
    3: {"level": "Critical", "color": "#EF4444", "title": "警惕心室颤动 (VF) 风险"},
    4: {"level": "Critical", "color": "#F43F5E", "title": "发现室速/室扑 (VT/VFl) 特征"},
    5: {"level": "Warning", "color": "#EAB308", "title": "心房扑动/房速 (AT/AFl) 特征"}
}
CLASS_NAMES = ["正常稳态", "室早", "房颤", "室颤", "室速/扑", "房扑/速"]

if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []

# ==========================================
# 侧边栏与数据加载
# ==========================================
st.sidebar.title("📡 信号监测控制台")
clinical_cases = {
    "100": {"bed_label": "档案 100 (基线稳态)", "desc": "稳态记录：主要包含正常心搏 (N)，基线平稳。"},
    "119": {"bed_label": "档案 119 (室早特征)", "desc": "形态畸变：包含大量室性早搏 (V) 畸变波形。"},
    "201": {"bed_label": "档案 201 (房颤演变)", "desc": "节律失序：RR间期绝对不齐，呈现典型房颤特征。"},
    "207": {"bed_label": "档案 207 (室速/室扑)", "desc": "极危恶性：短阵室速与室扑交替发作。"},
    "209": {"bed_label": "档案 209 (房性激惹)", "desc": "室上性激惹：阵发性室上速/房速发作特征。"},
    "PROSIM_01": {"bed_label": "外部硬件源 (ProSim)", "desc": "示波器直连：硬件模拟病患生理电信号。"}
}
selected_id = st.sidebar.selectbox("选择监测源", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
start_mins = st.sidebar.slider("设定起始点 (分钟)", 0.0, 15.0, 10.0, 0.1)

if st.sidebar.button("🗑️ 清空异常记录", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()
st.sidebar.caption("提示：改变左侧选项或刷新页面即可停止引擎。")

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
            return np.zeros(150000)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', 'mitdb', rec_id)
    if os.path.exists(rec_path + ".dat"):
        record = wfdb.rdrecord(rec_path, sampto=300000)
    else:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
    return signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), 250, 360)

data_source = load_sim_data(selected_id)

# ==========================================
# 主界面布局 (UI 占位符定义)
# ==========================================
st.title(f"📊 动态分析终端 - {clinical_cases[selected_id]['bed_label']}")
st.info(f"**【档案特征】** {clinical_cases[selected_id]['desc']}")

# 1. 顶部预警面板
warning_box = st.empty()

# 2. 波形展示区域
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("📡 实时推流模拟")
    chart_placeholder = st.empty()
    st.markdown("##### 🔍 10 分钟 病灶溯源热力图 (1D-CAM)")
    cam_placeholder = st.empty()
with col2:
    st.subheader("⏱️ 监测状态")
    metric_time = st.empty()

st.divider()
st.subheader("📑 异常事件追踪日志")
log_placeholder = st.empty()

# ==========================================
# 核心推流引擎 (结合 V3.0 的 while True 架构)
# ==========================================
if st.sidebar.button("🔴 启动预警引擎", use_container_width=True):
    st.sidebar.success("预警引擎已激活！推流进行中...")
    
    # 局部状态初始化 (仅在循环内存活，最大化性能)
    sim_step = 0
    base_offset_pts = int(start_mins * 60 * 250)
    ecg_buffer = np.full(1000, np.nan)
    
    display_len = 1000  
    step_size = 20       # 恢复 V3.0 的细致步长
    refresh_rate = 0.08  # 恢复 V3.0 的 12.5 FPS 黄金渲染帧率
    
    # 推断数据缓存
    last_class = 0
    last_prob = 0.0
    last_all_probs = [0] * 6
    last_cam = []
    ui_hold = {'class': 0, 'prob': 0.0, 'all_probs': [0]*6}
    
    # Memoization 阻断缓存
    cached_warning_html = ""
    cached_log_signature = ""

    # 🚀 进入丝滑渲染循环
    while True:
        current_pts = base_offset_pts + (sim_step * step_size)
        if current_pts >= len(data_source) - 1:
            st.warning("已到达记录末尾。")
            break

        # -------------------------------------
        # 1. 扫描模式环形缓冲更新 & 波形渲染
        # -------------------------------------
        new_data = data_source[current_pts - step_size : current_pts]
        idx = (sim_step * step_size) % display_len
        ecg_buffer[idx : idx + step_size] = new_data
        
        gap_end = (idx + step_size + 25) % display_len
        if gap_end > (idx + step_size):
            ecg_buffer[idx + step_size : gap_end] = np.nan
        else:
            ecg_buffer[idx + step_size : display_len] = np.nan
            ecg_buffer[0 : gap_end] = np.nan

        df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': ecg_buffer})
        line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
            x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None), 
            y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)", grid=True)) 
        ).properties(height=250).configure_view(strokeOpacity=0) 
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        cur_sec = current_pts / 250
        time_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
        metric_time.metric("当前时间点", time_str)

        # -------------------------------------
        # 2. 后端请求 (每隔 12 帧调用一次)
        # -------------------------------------
        if sim_step % 12 == 0:
            full_win = data_source[max(0, current_pts - 150000) : current_pts]
            if len(full_win) < 150000:
                full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')
            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win.tolist()}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    last_class = data['pred_class']
                    last_prob = data['future_risk_prob']
                    last_all_probs = data.get('all_probs', [0]*6)
                    
                    # 仅在获取到新 CAM 时渲染，防止闪烁
                    new_cam = data.get('cam_heatmap', [])
                    if new_cam != last_cam:
                        last_cam = new_cam
                        df_cam = pd.DataFrame({'T': np.arange(len(last_cam)), 'A': last_cam})
                        cam_chart = alt.Chart(df_cam).mark_area(
                            color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#000000', offset=0), alt.GradientStop(color='#FF0000', offset=1)])
                        ).encode(x=alt.X('T:Q', axis=None), y=alt.Y('A:Q', scale=alt.Scale(domain=[0, 1]), axis=None)).properties(height=60).configure_view(strokeOpacity=0)
                        cam_placeholder.altair_chart(cam_chart, use_container_width=True)
            except Exception:
                pass

        # -------------------------------------
        # 3. 临床级防抖状态机 (15s缓冲 & 30s聚合)
        # -------------------------------------
        is_new_event = True
        is_visual_alert = False
        
        if st.session_state.anomaly_logs:
            last_log = st.session_state.anomaly_logs[-1]
            last_end_sec = sum(int(x) * 60**i for i, x in enumerate(reversed(last_log["结束时间"].split(':'))))
            
            if last_class != 0:
                is_visual_alert = True
                if last_log["预警类型"] == last_class:
                    if (cur_sec - last_end_sec) <= 30: # 30s 内聚合
                        is_new_event = False
                        last_log["状态"] = "活跃"
                        last_log["结束时间"] = time_str
                        # 记录期间最高峰值
                        current_max = float(last_log["峰值概率"].strip('%'))
                        if (last_prob * 100) > current_max:
                            last_log["峰值概率"] = f"{last_prob * 100:.1f}%"
                    else:
                        last_log["状态"] = "已结束"
                else:
                    last_log["状态"] = "已结束"
            else:
                is_new_event = False 
                if last_log["状态"] == "活跃":
                    if (cur_sec - last_end_sec) <= 15: # 15s 解除缓冲期
                        is_visual_alert = True
                    else:
                        last_log["状态"] = "已结束"
        else:
            if last_class != 0:
                is_visual_alert = True

        if is_new_event and last_class != 0:
            st.session_state.anomaly_logs.append({
                "开始时间": time_str, "结束时间": time_str, "预警类型": last_class, 
                "分级": INFERENCE_MAP[last_class]["level"],
                "形态描述": INFERENCE_MAP[last_class]["title"].split("：")[0], 
                "状态": "活跃", "峰值概率": f"{last_prob * 100:.1f}%"
            })

        # -------------------------------------
        # 4. 预警面板与日志的阻断渲染 (Memoization)
        # -------------------------------------
        if is_visual_alert and last_class != 0:
            ui_hold.update({'class': last_class, 'prob': last_prob, 'all_probs': last_all_probs})
            
        target = ui_hold if is_visual_alert else {'class': last_class, 'prob': last_prob, 'all_probs': last_all_probs}
        diag = INFERENCE_MAP[target['class']]
        display_p = target['prob'] if target['class'] != 0 else target['prob'] * 0.2
        
        # 动态构建 HTML
        dist_html = ""
        if target['class'] != 0:
            prob_dict = {i: p for i, p in enumerate(target['all_probs'])}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            dist_html += "<div style='margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>"
            for c_i, p in sorted_probs[:3]:
                if p > 0.05:
                    w = min(p * 100, 100)
                    c = diag["color"] if c_i == target['class'] else "rgba(255,255,255,0.2)"
                    dist_html += f"<div style='display:flex; align-items:center; margin-bottom:4px; font-size:12px; color:#A0AEC0;'><span style='width:80px;'>{CLASS_NAMES[c_i]}</span><div style='flex-grow:1; background:rgba(0,0,0,0.3); height:6px; margin:0 10px; border-radius:3px;'><div style='width:{w}%; background:{c}; height:100%; border-radius:3px;'></div></div><span>{p*100:.1f}%</span></div>"
            dist_html += "</div>"
            
        current_warning_html = f"""
        <div style="background-color: #1E1E28; padding: 15px 20px; border-radius: 8px; border-left: 6px solid {diag['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <h2 style="margin:0; font-size: 20px; color: {diag['color']};">{diag['title']}</h2>
            <h3 style="margin: 5px 0 0 0; font-weight: 400; font-size: 14px; color: #E2E8F0;">主要事件推断概率：<strong style="font-size: 16px;">{display_p * 100:.1f}%</strong></h3>
            {dist_html}
        </div>
        """
        
        # 【核心性能提升】仅在 HTML 发生实质性改变时重绘，消除全屏白光频闪
        if current_warning_html != cached_warning_html:
            warning_box.markdown(current_warning_html, unsafe_allow_html=True)
            cached_warning_html = current_warning_html

        # 日志渲染阻断
        if st.session_state.anomaly_logs:
            last_log_ref = st.session_state.anomaly_logs[-1]
            current_log_signature = f"{len(st.session_state.anomaly_logs)}_{last_log_ref['结束时间']}_{last_log_ref['状态']}_{last_log_ref['峰值概率']}"
            
            if current_log_signature != cached_log_signature:
                display_data = [{
                    "区间": f"{log['开始时间']} - {log['结束时间']}" if log['开始时间'] != log['结束时间'] else log['开始时间'],
                    "级别": log['分级'],
                    "状态": "🔴 警报中" if log['状态'] == "活跃" else "⚪ 解除",
                    "峰值": log['峰值概率'],
                    "描述": log['形态描述']
                } for log in reversed(st.session_state.anomaly_logs)]
                log_placeholder.table(pd.DataFrame(display_data))
                cached_log_signature = current_log_signature
        else:
            if cached_log_signature != "empty":
                log_placeholder.info("✅ 当前监测时段内暂无形态畸变记录。")
                cached_log_signature = "empty"

        sim_step += 1
        time.sleep(refresh_rate)
else:
    # 未启动引擎时的静态占位提示
    warning_box.info("👈 请在左侧侧边栏点击【启动预警引擎】开始推流分析")
    chart_placeholder.line_chart(np.full(1000, 0.0), height=250, use_container_width=True)