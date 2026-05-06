"""
HealthMonitor V3.0 - 前端实时渲染与控制台模块
功能描述：基于 Streamlit 框架构建的监护仪 UI，负责动态展示心电波形、
        后端分类预测结果、1D-CAM 热力图以及记录长程异常事件日志。
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
# [系统初始化] 测试病例与全局状态设定
# ==========================================
clinical_cases = {
    "100": {"bed_label": "档案 100 (基线稳态)", "desc": "稳态记录：主要包含正常心搏 (N)，基线平稳。"},
    "119": {"bed_label": "档案 119 (室早特征)", "desc": "形态畸变：包含大量室性早搏 (V) 畸变波形。"},
    "201": {"bed_label": "档案 201 (房颤演变)", "desc": "节律失序：RR间期绝对不齐，呈现典型房颤特征。"},
    "207": {"bed_label": "档案 207 (室速/室扑)", "desc": "极危恶性：短阵室速与室扑交替发作。"},
    "209": {"bed_label": "档案 209 (房性激惹)", "desc": "室上性激惹：阵发性室上速/房速发作特征。"},
    "PROSIM_01": {"bed_label": "外部硬件源 (ProSim)", "desc": "示波器直连：硬件模拟病患生理电信号。"}
}

# 初始化会话状态，保证重绘时数据不丢失
SESSION_DEFAULTS = {
    'anomaly_logs': [],                 # 异常事件追踪日志
    'ecg_buffer': np.full(1000, np.nan),# 长度为1000的心电渲染缓冲区
    'engine_running': False,            # 引擎运行状态
    'sim_step': 0,                      # 模拟时间步进
    'base_offset_pts': 0,               # 信号读取的初始偏移量
    'selected_case_id': None            # 当前锁定的档案ID
}

for key, default_val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val if not isinstance(default_val, np.ndarray) else default_val.copy()

# ==========================================
# [侧边栏] 控制面板布局
# ==========================================
st.sidebar.title("📡 信号监测控制台")

selected_id = st.sidebar.selectbox(
    "选择监测源",
    options=list(clinical_cases.keys()),
    format_func=lambda x: clinical_cases[x]["bed_label"]
)
case = clinical_cases[selected_id]

st.sidebar.divider()
st.sidebar.subheader("⏳ 时间轴控制")
start_mins = st.sidebar.slider("设定起始点 (分钟)", 10.0, 15.0, 10.0, 0.1)
base_offset_pts = int(start_mins * 60 * 250)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🗑️ 清空日志", use_container_width=True):
        st.session_state.anomaly_logs = []
        st.rerun()
with col_btn2:
    if st.button("🔄 重置波形", use_container_width=True):
        st.session_state.ecg_buffer = np.full(1000, np.nan)
        st.rerun()

# ==========================================
# [数据处理] 信号读取与滤波
# ==========================================
def clean_ecg_signal(data, fs=360):
    """使用 0.5Hz~45Hz 巴特沃斯带通滤波器去除基线漂移和肌电干扰"""
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    """读取指定档案数据，清洗并进行 360Hz 到 250Hz 的重采样"""
    if rec_id == "PROSIM_01":
        if os.path.exists("prosim_custom_signal.npy"):
            return np.load("prosim_custom_signal.npy")
        else:
            st.sidebar.error("未找到 prosim_custom_signal.npy")
            return np.zeros(150000)
            
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, 'data', 'mitdb', rec_id)
    
    if os.path.exists(rec_path + ".dat"):
        record = wfdb.rdrecord(rec_path, sampto=300000)
    else:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
        
    return signal.resample_poly(clean_ecg_signal(record.p_signal[:, 0], fs=360), 250, 360)

# 若档案切换或未启动，则提前预加载数据
if not st.session_state.engine_running or st.session_state.selected_case_id != selected_id:
    data_source = load_sim_data(selected_id)
    st.session_state.selected_case_id = selected_id

# ==========================================
# [主视窗] 布局与 UI 挂载点
# ==========================================
st.title(f"📊 动态分析终端 - {case['bed_label']}")
st.info(f"**【档案特征】** {case['desc']}")
st.markdown("### 🎯 未来 5 分钟异常分布推断")

warning_box = st.empty()
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

# 定义类别映射字典，用于动态更改报警级别和 UI 颜色
INFERENCE_MAP = {
    0: {"level": "Normal", "color": "#10B981", "title": "信号平稳，未见明显节律异常"},
    1: {"level": "Warning", "color": "#F59E0B", "title": "提示频发室性早搏 (PVC) 风险"},
    2: {"level": "High", "color": "#F97316", "title": "检测到心房颤动 (AFib) 特征"},
    3: {"level": "Critical", "color": "#EF4444", "title": "警惕心室颤动 (VF) 风险"},
    4: {"level": "Critical", "color": "#F43F5E", "title": "发现室速/室扑 (VT/VFl) 特征"},
    5: {"level": "Warning", "color": "#EAB308", "title": "心房扑动/房速 (AT/AFl) 特征"}
}
CLASS_NAMES = ["正常稳态", "室早", "房颤", "室颤", "室速/扑", "房扑/速"]

# 渲染控制参数
display_len = 1000  # 屏幕上的波形显示长度
step_size = 20      # 每一帧往前推进的数据点数

# ==========================================
# [主循环] 单帧执行器
# ==========================================
def run_simulation_frame():
    """每次触发重绘时执行一帧物理状态更新"""
    s = st.session_state
    data_source = load_sim_data(selected_id)
    current_pts = base_offset_pts + (s.sim_step * step_size)
    
    # 溢出保护
    if current_pts >= len(data_source) - 1:
        st.warning("已到达记录末尾。")
        s.engine_running = False
        return
        
    # --- 1. 更新滚动波形 (FIFO) ---
    new_data = data_source[current_pts - step_size : current_pts]
    # 利用 numpy.roll 整体左移缓冲区，并将新数据拼接到尾部
    s.ecg_buffer = np.roll(s.ecg_buffer, -step_size)
    s.ecg_buffer[-step_size:] = new_data
    
    # 渲染当前帧波形
    df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': s.ecg_buffer})
    line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
        x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None),
        y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="幅度 (mV)", grid=True))
    ).properties(height=250).configure_view(strokeOpacity=0)
    chart_placeholder.altair_chart(line_chart, use_container_width=True)
    
    # 时间轴计算
    total_seconds = current_pts / 250
    time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
    metric_time.metric("档案时间点", time_str)
    
    # --- 2. 触发后端预测 ---
    display_prob = 0.0
    pred_class = 0
    all_probs = [0] * 6
    cam_data = []
    
    # 降频调用策略：每 12 帧 (约一秒) 请求一次推理服务
    if s.sim_step % 12 == 0:
        full_win_for_ai = data_source[current_pts - 150000 : current_pts]
        # 长度防呆机制，不足窗口长度则在头部补零
        if len(full_win_for_ai) < 150000:
            full_win_for_ai = np.pad(full_win_for_ai, (150000 - len(full_win_for_ai), 0), 'constant')
            
        try:
            # 向后端传递唯一标识 X-Device-ID 以使用专属平滑队列
            resp = requests.post(
                "http://127.0.0.1:8000/api/predict",
                json={"ecg": full_win_for_ai.tolist()},
                headers={"X-Device-ID": selected_id},
                timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
            pred_class = data['pred_class']
            display_prob = data['future_risk_prob']
            all_probs = data.get('all_probs', [0]*6)
            cam_data = data.get('cam_heatmap', [])
        except Exception as e:
            st.toast(f"⚠️ 后端网络异常: {e}")

    # --- 3. 渲染 CAM 注意力热力图 ---
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

    # --- 4. 生成临床预警卡片与分布条 ---
    diag_info = INFERENCE_MAP.get(pred_class, INFERENCE_MAP[0])
    accent_color = diag_info["color"]
    status_title = diag_info["title"]
    alert_level = diag_info["level"]
    
    prob_dict = {i: prob for i, prob in enumerate(all_probs)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    distribution_html = ""
    if pred_class != 0:
        distribution_html += "<div style='margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; font-size: 14px;'>"
        for cls_idx, prob in sorted_probs[:3]:
            if prob > 0.05:
                bar_width = min(prob * 100, 100)
                bar_color = accent_color if cls_idx == pred_class else "rgba(255,255,255,0.3)"
                font_weight = "600" if cls_idx == pred_class else "400"
                text_color = "#E2E8F0" if cls_idx == pred_class else "#A0AEC0"
                distribution_html += f"<div style='display: flex; align-items: center; margin-bottom: 8px; color: {text_color};'>"
                distribution_html += f"<span style='width: 120px; font-weight: {font_weight};'>{CLASS_NAMES[cls_idx]}</span>"
                distribution_html += f"<div style='flex-grow: 1; background: rgba(0,0,0,0.3); border-radius: 4px; margin: 0 12px; height: 8px;'><div style='width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 4px;'></div></div>"
                distribution_html += f"<span style='width: 50px; text-align: right;'>{prob*100:.1f}%</span></div>"
        distribution_html += "</div>"
    else:
        # 若为正常，降低背景置信度以防引起错觉
        display_prob = display_prob * 0.20

    warning_box.markdown(f"""
    <div style="background-color: #1E1E28; padding: 20px 24px; border-radius: 8px; border-left: 6px solid {accent_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <div style="text-align: left;">
            <h2 style="margin:0; font-size: 22px; color: {accent_color}; letter-spacing: 0.5px;">{status_title}</h2>
            <h3 style="margin: 8px 0 0 0; font-weight: 400; font-size: 16px; color: #E2E8F0;">主要事件推断概率：<strong style="font-size: 18px;">{display_prob * 100:.1f}%</strong></h3>
        </div>
        {distribution_html}
    </div>
    """, unsafe_allow_html=True)

    # --- 5. 异常日志状态机更新 ---
    is_new_event = True
    if s.anomaly_logs:
        last_log = s.anomaly_logs[-1]
        if last_log["状态"] == "活跃" and last_log["预警类型"] == pred_class:
            is_new_event = False
            last_log["结束时间"] = time_str
            # 更新该连续事件期间的最大概率峰值
            try:
                current_max = float(str(last_log["峰值概率"]).replace('%', ''))
                if (display_prob * 100) > current_max:
                    last_log["峰值概率"] = f"{display_prob * 100:.1f}%"
            except (ValueError, AttributeError):
                last_log["峰值概率"] = f"{display_prob * 100:.1f}%"
        elif last_log["状态"] == "活跃" and last_log["预警类型"] != pred_class:
            last_log["状态"] = "已结束"

    # 捕捉新事件的产生
    if is_new_event and pred_class != 0:
        s.anomaly_logs.append({
            "开始时间": time_str, "结束时间": time_str, "预警类型": pred_class,
            "分级": alert_level, "形态描述": status_title.split("：")[0].strip(),
            "峰值概率": f"{display_prob * 100:.1f}%", "状态": "活跃"
        })
    elif pred_class == 0 and s.anomaly_logs and s.anomaly_logs[-1]["状态"] == "活跃":
        s.anomaly_logs[-1]["状态"] = "已结束"

    # 渲染日志表
    if s.anomaly_logs:
        display_data = [{
            "区间": f"{log['开始时间']} 至 {log['结束时间']}" if log['开始时间'] != log['结束时间'] else log['开始时间'],
            "级别": log['分级'],
            "状态": "🔴 警报中" if log['状态'] == "活跃" else "⚪ 解除",
            "峰值": log['峰值概率'],
            "描述": log['形态描述']
        } for log in reversed(s.anomaly_logs)]
        log_placeholder.table(pd.DataFrame(display_data))
    else:
        log_placeholder.info("✅ 当前监测时段内暂无形态畸变记录。")

    s.sim_step += 1

# ==========================================
# [引擎开关] 控制逻辑
# ==========================================
st.sidebar.divider()
col_ctrl1, col_ctrl2 = st.sidebar.columns(2)
with col_ctrl1:
    if st.button("🔴 启动引擎", use_container_width=True):
        st.session_state.engine_running = True
        st.session_state.sim_step = 0
        st.session_state.base_offset_pts = base_offset_pts
with col_ctrl2:
    if st.button("⏹️ 暂停", use_container_width=True):
        st.session_state.engine_running = False

# 依靠 st.rerun() 实现合规的循环推流
if st.session_state.engine_running:
    run_simulation_frame()
    time.sleep(0.08)
    st.rerun()