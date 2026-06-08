"""
->AI -> (Clinical Dashboard)
->
    1. -> Streamlit -> UI->
    2. ->(Sweep)->
    3. -> AI ->
    4. -> 1D-CAM -> 10 ->
    5. ->
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

st.set_page_config(page_title="->", layout="wide")

# ==========================================
# -> 1-> (->)
# ==========================================
clinical_cases = {
    "100": {"bed_label": "-> 100 (->)", "desc": "-> -> (N)-> Class 0->"},
    "119": {"bed_label": "-> 119 (->)", "desc": "-> -> (V) -> Class 1->"},
    "201": {"bed_label": "-> 201 (->)", "desc": "-> ->RR-> Class 2->"},
    "207": {"bed_label": "-> 207 (->/->)", "desc": "-> -> Class 4->"},
    "209": {"bed_label": "-> 209 (->)", "desc": "-> ->/-> Class 5->"},
    
   
    "PROSIM_01": {"bed_label": "-> (ProSim 200)", "desc": "-> ->"}
}

# ->
if 'anomaly_logs' not in st.session_state:
    st.session_state.anomaly_logs = []
if 'ecg_buffer' not in st.session_state:
    st.session_state.ecg_buffer = np.full(1000, np.nan) # ->

# ==========================================
# -> 2->
# ==========================================
st.sidebar.title("-> ->")
selected_id = st.sidebar.selectbox("->", options=list(clinical_cases.keys()), format_func=lambda x: clinical_cases[x]["bed_label"])
case = clinical_cases[selected_id]

st.sidebar.divider()
st.sidebar.subheader("-> ->")
start_mins = st.sidebar.slider("-> (->)", 10.0, 15.0, 10.0, 0.1)
base_offset_pts = int(start_mins * 60 * 250)

if st.sidebar.button("-> ->", use_container_width=True):
    st.session_state.anomaly_logs = []
    st.rerun()

def clean_ecg_signal(data, fs=360):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    return filtfilt(b, a, data)

@st.cache_data
def load_sim_data(rec_id):
    """-> MIT-BIH ->"""
    
    # =========================================
    # ->
    # =========================================
    if rec_id == "PROSIM_01":
        # -> (-> .npy ->)
        return np.load("prosim_custom_signal.npy")

    # =========================================
    # -> MIT-BIH ->
    # =========================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'mitdb')
    rec_path = os.path.join(data_dir, rec_id)
    
    if os.path.exists(rec_path + ".dat"):
        record = wfdb.rdrecord(rec_path, sampto=300000)
    else:
        record = wfdb.rdrecord(rec_id, pn_dir='mitdb', sampto=300000)
        
    raw_ecg = record.p_signal[:, 0]
    return signal.resample_poly(clean_ecg_signal(raw_ecg, fs=360), 250, 360)
data_source = load_sim_data(selected_id)

# ==========================================
# -> 3->UI ->
# ==========================================
st.title(f"-> -> - {case['bed_label']}")
st.info(f"**->** {case['desc']}")

st.markdown("### -> -> 5 ->")
warning_box = st.empty()  

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("-> -> ")
    chart_placeholder = st.empty()
    st.markdown("##### -> 10 -> -> (Grad-CAM)")
    cam_placeholder = st.empty() # -> XAI ->
with col2:
    st.subheader("-> ->")
    metric_time = st.empty()
    metric_risk = st.empty()
    metric_status = st.empty()

st.divider()
st.subheader("-> ->")
log_placeholder = st.empty()
# ->AI-> (->/->)
INFERENCE_MAP = {
    0: {"level": "Normal", "color": "#10B981", "title": "->"},
    1: {"level": "Warning", "color": "#F59E0B", "title": "-> (PVC) ->"},
    2: {"level": "High", "color": "#F97316", "title": "-> (AFib) ->"},
    3: {"level": "Critical", "color": "#EF4444", "title": "-> (VF)->"},
    4: {"level": "Critical", "color": "#F43F5E", "title": "->/-> (VT/VFl) ->"},
    5: {"level": "Warning", "color": "#EAB308", "title": "->/-> (AT/AFl) ->"}
}

CLASS_NAMES = ["-> (Normal)", "-> (PVC)", "-> (AFib)", "-> (VF)", "->/-> (VT/VFl)", "->/-> (AT/AFl)"]
# ==========================================
# -> 4->
# ==========================================
if st.sidebar.button("-> ->", use_container_width=True):
    st.sidebar.success("->")
    
    st.session_state.sim_step = 0
    st.session_state.ecg_buffer = np.full(1000, np.nan)
    
    display_len = 1000  
    step_size = 20       # ->
    refresh_rate = 0.08  # ->
    
    display_prob = 0.0
    alert_level = "Normal"
    cam_data = []

    while True:
        current_pts = base_offset_pts + (st.session_state.sim_step * step_size)
        if current_pts >= len(data_source) - 1:
            st.warning("->")
            break

        # 1. ->
        new_data = data_source[current_pts - step_size : current_pts]
        idx = (st.session_state.sim_step * step_size) % display_len
        st.session_state.ecg_buffer[idx : idx + step_size] = new_data
        
        gap_end = (idx + step_size + 25) % display_len
        if gap_end > (idx + step_size):
            st.session_state.ecg_buffer[idx + step_size : gap_end] = np.nan
        else:
            st.session_state.ecg_buffer[idx + step_size : display_len] = np.nan
            st.session_state.ecg_buffer[0 : gap_end] = np.nan

        # 2. ->
        df_plot = pd.DataFrame({'Point': np.arange(display_len), 'Signal': st.session_state.ecg_buffer})
        line_chart = alt.Chart(df_plot).mark_line(color='#00FF41', strokeWidth=1.5).encode(
            x=alt.X('Point:Q', scale=alt.Scale(domain=[0, display_len]), axis=None), 
            y=alt.Y('Signal:Q', scale=alt.Scale(domain=[-3.5, 3.5]), axis=alt.Axis(title="-> (mV)", grid=True)) 
        ).properties(height=250).configure_view(strokeOpacity=0) 
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

        # 3. ->
        total_seconds = current_pts / 250
        time_str = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"
        metric_time.metric("->", time_str)

        # 4. -> (->)
        if st.session_state.sim_step % 12 == 0:
            full_win_for_ai = data_source[current_pts - 150000 : current_pts]
            try:
                resp = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": full_win_for_ai.tolist()})
                resp.raise_for_status()
                data = resp.json()
                
                # ->
                pred_class = data['pred_class']
                display_prob = data['future_risk_prob']
                all_probs = data.get('all_probs', [0]*6)
                cam_data = data.get('cam_heatmap', [])
            except Exception:
                pass

        # 5. XAI ->
        if cam_data:
            df_cam = pd.DataFrame({'TimeStep': np.arange(len(cam_data)), 'Attention': cam_data})
            cam_chart = alt.Chart(df_cam).mark_area(
                color=alt.Gradient(gradient='linear', stops=[
                    alt.GradientStop(color='#000000', offset=0), 
                    alt.GradientStop(color='#FF0000', offset=1)
                ])
            ).encode(
                x=alt.X('TimeStep:Q', scale=alt.Scale(domain=[0, len(cam_data)]), axis=alt.Axis(labels=False, title="-> 10 ->")),
                y=alt.Y('Attention:Q', scale=alt.Scale(domain=[0, 1.0]), axis=None)
            ).properties(height=60).configure_view(strokeOpacity=0)
            cam_placeholder.altair_chart(cam_chart, use_container_width=True)

        # 6. ->  ---
        diag_info = INFERENCE_MAP.get(pred_class, INFERENCE_MAP[0])
        accent_color = diag_info["color"]
        status_title = diag_info["title"]
        
        prob_dict = {i: prob for i, prob in enumerate(all_probs)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        distribution_html = ""
        if pred_class != 0: 
            # -> Streamlit ->
            distribution_html += "<div style='margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; font-size: 14px;'>"
            distribution_html += "<p style='margin:0 0 10px 0; font-weight: 500; color: #A0AEC0;'>-> -></p>"
            
            for cls_idx, prob in sorted_probs[:3]:
                if prob > 0.05: 
                    cls_name = CLASS_NAMES[cls_idx]
                    bar_width = min(prob * 100, 100)
                    bar_color = accent_color if cls_idx == pred_class else "rgba(255,255,255,0.3)"
                    font_weight = "600" if cls_idx == pred_class else "400"
                    text_color = "#E2E8F0" if cls_idx == pred_class else "#A0AEC0"
                    
                    distribution_html += f"<div style='display: flex; align-items: center; margin-bottom: 8px; color: {text_color};'>"
                    distribution_html += f"<span style='width: 140px; font-weight: {font_weight};'>{cls_name}</span>"
                    distribution_html += f"<div style='flex-grow: 1; background: rgba(0,0,0,0.3); border-radius: 4px; margin: 0 12px; height: 8px;'>"
                    distribution_html += f"<div style='width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 4px;'></div>"
                    distribution_html += f"</div>"
                    distribution_html += f"<span style='width: 50px; text-align: right;'>{prob*100:.1f}%</span>"
                    distribution_html += f"</div>"
            distribution_html += "</div>"
        else:
            display_prob = display_prob * 0.20 # ->

        # -> + ->
        warning_html = f"""
        <div style="background-color: #1E1E28; padding: 20px 24px; border-radius: 8px; border-left: 6px solid {accent_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <div style="text-align: left;">
                <h2 style="margin:0; font-size: 22px; color: {accent_color}; letter-spacing: 0.5px;">{status_title}</h2>
                <h3 style="margin: 8px 0 0 0; font-weight: 400; font-size: 16px; color: #E2E8F0;">-><strong style="font-size: 18px;">{display_prob * 100:.1f}%</strong></h3>
            </div>
            {distribution_html}
        </div>
        """
        warning_box.markdown(warning_html, unsafe_allow_html=True)
        
        # 7. -> (->)
        is_new_event = True
        if st.session_state.anomaly_logs:
            last_log = st.session_state.anomaly_logs[-1]
            if last_log["->"] == "->" and last_log["->"] == pred_class:
                is_new_event = False
                last_log["->"] = time_str
                current_max = float(last_log["->"].strip('%'))
                if (display_prob * 100) > current_max:
                    last_log["->"] = f"{display_prob * 100:.1f}%"
            elif last_log["->"] == "->" and last_log["->"] != pred_class:
                last_log["->"] = "->"

        if is_new_event and pred_class != 0:
            st.session_state.anomaly_logs.append({
                "->": time_str,
                "->": time_str,
                "->": pred_class,
                "->": alert_level,
                "->": status_title.split("->")[0].replace("->", "").replace("->", "").strip(),
                "->": f"{display_prob * 100:.1f}%",
                "->": "->"
            })
        elif pred_class == 0 and st.session_state.anomaly_logs and st.session_state.anomaly_logs[-1]["->"] == "->":
            st.session_state.anomaly_logs[-1]["->"] = "->"

        if st.session_state.anomaly_logs:
            display_data = [{
                "->": f"{log['->']} -> {log['->']}" if log['->'] != log['->'] else f"{log['->']} (->)",
                "->": log['->'],
                "->": "-> ->" if log['->'] == "->" else "-> ->",
                "->": log['->'],
                "->": log['->']
            } for log in reversed(st.session_state.anomaly_logs)]
            log_placeholder.table(pd.DataFrame(display_data))
        else:
            log_placeholder.info("-> ->")

        st.session_state.sim_step += 1
        time.sleep(refresh_rate)