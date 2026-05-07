"""
HealthMonitor V4.5 - 前端实时渲染与控制台模块 (布局优化版)
功能描述：
1. 预留占位：引擎未启动时显示空白坐标系与待机预警面板。
2. 布局重排：将状态监控与日志追踪移至波形图下方。
3. 稳健渲染：保持 2.5 FPS 采样率，彻底消除 DOM 重绘导致的频闪。
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

# --- 核心配置 ---
INFERENCE_MAP = {
    0: {"level": "Normal", "color": "#10B981", "title": "信号平稳，未见明显节律异常"},
    1: {"level": "Warning", "color": "#F59E0B", "title": "提示频发室性早搏 (PVC) 风险"},
    2: {"level": "High", "color": "#F97316", "title": "检测到心房颤动 (AFib) 特征"},
    3: {"level": "Critical", "color": "#EF4444", "title": "警惕心室颤动 (VF) 风险"},
    4: {"level": "Critical", "color": "#F43F5E", "title": "发现室速/室扑 (VT/VFl) 特征"},
    5: {"level": "Warning", "color": "#EAB308", "title": "心房扑动/房速 (AT/AFl) 特征"}
}
CLASS_NAMES = ["正常稳态", "室早", "房颤", "室颤", "室速/扑", "房扑/速"]

# --- 【修复】缓存与视觉锁初始化 ---
SESSION_DEFAULTS = {
    'anomaly_logs': [],                 
    'ecg_buffer': np.full(1000, 0.0),
    'engine_running': False,            
    'sim_step': 0,                      
    'base_offset_pts': int(10.0 * 60 * 250), # 修复点：加入基准时间的初始化
    'selected_case_id': None,
    'last_pred': {'class': 0, 'prob': 0.0, 'all_probs': [0]*6, 'cam': []},
    'ui_hold': {'class': 0, 'prob': 0.0, 'all_probs': [0]*6}
}

for key, default_val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val if not isinstance(default_val, np.ndarray) else default_val.copy()

# --- 侧边栏 ---
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

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🗑️ 清空日志", use_container_width=True):
        st.session_state.anomaly_logs = []
        st.session_state.last_pred = {'class': 0, 'prob': 0.0, 'all_probs': [0]*6, 'cam': []}
        st.session_state.ui_hold = {'class': 0, 'prob': 0.0, 'all_probs': [0]*6}
        st.rerun()
with col_btn2:
    if st.button("🔄 重置波形", use_container_width=True):
        st.session_state.ecg_buffer = np.full(1000, 0.0)
        st.rerun()

# --- 数据加载与预处理 ---
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

if not st.session_state.engine_running or st.session_state.selected_case_id != selected_id:
    data_source = load_sim_data(selected_id)
    st.session_state.selected_case_id = selected_id
else:
    data_source = load_sim_data(st.session_state.selected_case_id)

# --- 主界面布局 ---
st.title(f"📊 动态分析终端 - {clinical_cases[selected_id]['bed_label']}")

# 1. 顶部预警面板 (占位定义)
warning_box = st.empty()

# 2. 波形展示区域 (占位定义)
st.subheader("📡 实时推流模拟")
chart_placeholder = st.empty()
cam_placeholder = st.empty()

st.divider()

# 3. 监控状态与日志区域 (移动到下方)
col_status, col_log = st.columns([1, 2])
with col_status:
    st.subheader("⏱️ 监测状态")
    metric_time = st.empty()
with col_log:
    st.subheader("📑 异常事件追踪日志")
    log_placeholder = st.empty()

# --- 渲染逻辑 ---
def draw_ui_elements(is_active=False):
    """绘制所有 UI 元素，支持待机占位"""
    s = st.session_state
    
    # A. 绘制预警面板
    def render_warning_html(diag, prob, all_p, cls_idx):
        accent = diag["color"]
        title = diag["title"]
        # 如果是正常状态，概率显示虚化
        display_p = prob if cls_idx != 0 else prob * 0.2
        
        dist_html = ""
        if cls_idx != 0:
            prob_dict = {i: p for i, p in enumerate(all_p)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            dist_html += "<div style='margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>"
            for c_i, p in sorted_probs[:3]:
                if p > 0.05:
                    width = min(p * 100, 100)
                    color = accent if c_i == cls_idx else "rgba(255,255,255,0.2)"
                    dist_html += f"<div style='display:flex; align-items:center; margin-bottom:4px; font-size:12px; color:#A0AEC0;'>"
                    dist_html += f"<span style='width:80px;'>{CLASS_NAMES[c_i]}</span>"
                    dist_html += f"<div style='flex-grow:1; background:rgba(0,0,0,0.3); height:6px; margin:0 10px; border-radius:3px;'><div style='width:{width}%; background:{color}; height:100%; border-radius:3px;'></div></div>"
                    dist_html += f"<span>{p*100:.1f}%</span></div>"
            dist_html += "</div>"
            
        return f"""
        <div style="background-color: #1E1E28; padding: 15px 20px; border-radius: 8px; border-left: 6px solid {accent}; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <h2 style="margin:0; font-size: 20px; color: {accent};">{title}</h2>
            <h3 style="margin: 5px 0 0 0; font-weight: 400; font-size: 14px; color: #E2E8F0;">
                事件推断概率：<strong style="font-size: 16px;">{display_p * 100:.1f}%</strong>
            </h3>
            {dist_html}
        </div>
        """

    # 逻辑裁决：是否由于 15s 缓冲期需要维持 UI 警报
    is_visual_alert = False
    current_sec = (s.base_offset_pts + s.sim_step * 100) / 250
    if s.anomaly_logs and s.anomaly_logs[-1]["状态"] == "活跃":
        last_end_sec = sum(int(x) * 60**i for i, x in enumerate(reversed(s.anomaly_logs[-1]["结束时间"].split(':'))))
        if s.last_pred['class'] != 0 or (current_sec - last_end_sec) <= 15:
            is_visual_alert = True
            s.ui_hold.update({'class': s.last_pred['class'], 'prob': s.last_pred['prob'], 'all_probs': s.last_pred['all_probs']}) if s.last_pred['class'] != 0 else None

    # 渲染面板
    target = s.ui_hold if is_visual_alert else s.last_pred
    warning_box.markdown(render_warning_html(INFERENCE_MAP[target['class']], target['prob'], target['all_probs'], target['class']), unsafe_allow_html=True)

    # B. 绘制波形图 (即便暂停也显示 buffer)
    df_plot = pd.DataFrame({'Point': np.arange(1000), 'Signal': s.ecg_buffer})
    line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
        x=alt.X('Point:Q', scale=alt.Scale(domain=[0, 1000]), axis=None),
        y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)", grid=True))
    ).properties(height=250).configure_view(strokeOpacity=0)
    chart_placeholder.altair_chart(line_chart, use_container_width=True)

    # C. 绘制溯源热力图 (CAM)
    if s.last_pred['cam']:
        df_cam = pd.DataFrame({'T': np.arange(len(s.last_pred['cam'])), 'A': s.last_pred['cam']})
        cam_c = alt.Chart(df_cam).mark_area(
            color=alt.Gradient(gradient='linear', stops=[
                alt.GradientStop(color='#000000', offset=0),
                alt.GradientStop(color='#FF0000', offset=1)
            ])
        ).encode(
            x=alt.X('T:Q', axis=None), y=alt.Y('A:Q', scale=alt.Scale(domain=[0, 1]), axis=None)
        ).properties(height=60).configure_view(strokeOpacity=0)
        cam_placeholder.altair_chart(cam_c, use_container_width=True)
    else:
        cam_placeholder.info("⏳ 引擎未启动或正在等待首批 10 分钟特征溯源计算...")

    # D. 绘制状态与日志
    time_str = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d}"
    metric_time.metric("档案时间点", time_str)
    
    if s.anomaly_logs:
        display_data = [{
            "区间": f"{log['开始时间']} - {log['结束时间']}" if log['开始时间'] != log['结束时间'] else log['开始时间'],
            "级别": log['分级'],
            "状态": "🔴 警报中" if log['状态'] == "活跃" else "⚪ 解除",
            "描述": log['形态描述']
        } for log in reversed(s.anomaly_logs)]
        log_placeholder.table(pd.DataFrame(display_data))
    else:
        log_placeholder.info("✅ 暂无形态畸变记录。")

# --- 模拟循环函数 ---
def run_simulation_frame():
    s = st.session_state
    current_pts = int(start_mins * 60 * 250) + (s.sim_step * 100)
    s.base_offset_pts = int(start_mins * 60 * 250)
    
    if current_pts >= len(data_source) - 1:
        st.warning("已到达记录末尾。")
        s.engine_running = False
        return

    # 推流数据更新
    new_data = data_source[current_pts - 100 : current_pts]
    s.ecg_buffer = np.roll(s.ecg_buffer, -100)
    s.ecg_buffer[-100:] = new_data
    
    # 后端推断 (每 1.2 秒调用一次)
    if s.sim_step % 3 == 0:
        try:
            full_win = data_source[max(0, current_pts-150000):current_pts]
            if len(full_win) < 150000:
                full_win = np.pad(full_win, (150000 - len(full_win), 0), 'constant')
            
            resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win.tolist()}, headers={"X-Device-ID": selected_id}, timeout=3)
            data = resp.json()
            s.last_pred.update({
                'class': data['pred_class'], 'prob': data['future_risk_prob'], 
                'all_probs': data.get('all_probs', [0]*6), 'cam': data.get('cam_heatmap', [])
            })
        except: pass

    # 状态机更新逻辑
    cur_sec = current_pts / 250
    t_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
    pc = s.last_pred['class']
    
    is_new = True
    if s.anomaly_logs:
        last_log = s.anomaly_logs[-1]
        last_end_sec = sum(int(x) * 60**i for i, x in enumerate(reversed(last_log["结束时间"].split(':'))))
        
        if pc != 0:
            if last_log["预警类型"] == pc:
                if (cur_sec - last_end_sec) <= 30:
                    is_new = False
                    last_log["状态"] = "活跃"
                    last_log["结束时间"] = t_str
                else:
                    last_log["状态"] = "已结束"
            else:
                last_log["状态"] = "已结束"
        else:
            is_new = False 
            if last_log["状态"] == "活跃":
                if (cur_sec - last_end_sec) > 15:
                    last_log["状态"] = "已结束"

    if is_new and pc != 0:
        s.anomaly_logs.append({
            "开始时间": t_str, "结束时间": t_str, "预警类型": pc, "分级": INFERENCE_MAP[pc]["level"],
            "形态描述": INFERENCE_MAP[pc]["title"].split("：")[0], "状态": "活跃", "峰值概率": f"{s.last_pred['prob'] * 100:.1f}%"
        })
    
    # 统一 UI 渲染
    draw_ui_elements(is_active=True)
    s.sim_step += 1

# --- 侧边栏控制 ---
st.sidebar.divider()
col_ctrl1, col_ctrl2 = st.sidebar.columns(2)
with col_ctrl1:
    if st.button("🔴 启动引擎", use_container_width=True):
        st.session_state.engine_running = True
        # 重置部分运行状态，保持起点对齐
        st.session_state.sim_step = 0
        st.session_state.base_offset_pts = int(start_mins * 60 * 250)
with col_ctrl2:
    if st.button("⏹️ 暂停", use_container_width=True):
        st.session_state.engine_running = False

# --- 主执行入口 ---
if st.session_state.engine_running:
    run_simulation_frame()
    time.sleep(0.4) # 降频 2.5 FPS 彻底消除闪烁
    st.rerun()
else:
    # 非运行状态执行占位渲染
    draw_ui_elements(is_active=False)