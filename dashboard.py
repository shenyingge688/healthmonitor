# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import requests
import wfdb  # 专门用于读取 PhysioNet 真实医疗数据的库

st.set_page_config(page_title="多模态监测系统", layout="wide")

# ================= 真实数据加载引擎 (带缓存防卡顿) =================
@st.cache_data
def load_mitbih_data():
    try:
        st.sidebar.info("⏳ 首次运行：正在从 PhysioNet 云端下载 200 号患者真实心电记录...")
        # 读取 200 号记录 (包含了极其典型的室性心律失常)
        record = wfdb.rdrecord('200', pn_dir='mitdb')
        st.sidebar.success("✅ 真实临床数据加载完成！")
        return record.p_signal[:, 0]  # 返回第一导联的真实心电波形
    except Exception as e:
        st.sidebar.error(f"下载失败，请检查网络: {e}")
        return None

# ================= 侧边栏：全局中枢 =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cardiogram.png", width=60)
    st.title("多模态监测台")
    
    st.markdown("---")
    st.subheader("⚙️ 设备控制")
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        
    if st.button("▶️ 启动 AI 实时监测" if not st.session_state.is_running else "⏹️ 停止监测", use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running

    st.markdown("---")
    patient_id = st.text_input("当前患者 ID", "真实早搏患者_Record200")

# ================= 顶部：AI 实时诊断引擎输出 =================
st.markdown("### 🧠 AI 实时诊断结论 (基于真实数据推理)")
ai_status_placeholder = st.empty()

# ================= 中间：实时波形 =================
st.markdown("---")
col_waves, col_trends = st.columns([2, 1])

with col_waves:
    st.markdown("#### 🫀 实时心电图 (MIT-BIH 真实数据)")
    ecg_chart = st.empty()
    st.markdown("#### 💪 皮肤电活动 (模拟数据)")
    eda_chart = st.empty()

with col_trends:
    st.markdown("#### 📋 历史疾病分布")
    pie_chart_placeholder = st.empty()
    
    if 'disease_counts' not in st.session_state:
        st.session_state.disease_counts = {
            "Normal": 1, "Arrhythmia": 0, "Sleep Apnea": 0, "Stress Overload": 0, "Autonomic Disorder": 0
        }

# ================= 实时联网推理引擎 =================
if st.session_state.is_running:
    API_URL = "http://127.0.0.1:8000/api/predict"
    
    # 提前加载真实的 ECG 数据
    real_ecg = load_mitbih_data()
    data_index = 0  # 类似磁带的播放指针
    
    while st.session_state.is_running:
        # 1. 像放电影一样，每次截取 750 个点 (约 3 秒波形)
        if real_ecg is not None:
            end_index = data_index + 750
            if end_index > len(real_ecg):
                data_index = 0  # 播到头了就循环播放
                end_index = 750
            ecg_window = real_ecg[data_index:end_index].tolist()
            data_index += 40  # 每次往前走 40 个点，产生波形向左流动的动画效果
        else:
            # 如果没网下载失败，退化为随机噪点
            ecg_window = (np.random.randn(750) * 0.1 + 0.6).tolist()
            
        # 注意：由于 MIT-BIH 只有心电图，皮肤电和血氧我们依然用代码模拟
        ppg_window = (np.random.randn(750) * 0.1 + 0.5).tolist()
        eda_window = (np.random.randn(750) * 0.02 + 0.95).tolist()
        
        payload = {"ecg": ecg_window, "ppg": ppg_window, "eda": eda_window}
        
        try:
            # 2. 发给后端大模型
            response = requests.post(API_URL, json=payload, timeout=2)
            
            if response.status_code == 200:
                ai_result = response.json().get("prediction", {})
                disease = ai_result.get("disease", "未知")
                confidence = ai_result.get("confidence", 0.0)
                alert_level = ai_result.get("alert_level", "Normal")
                hr = ai_result.get("hr", 0.0)
                
                # 记录疾病次数画饼图
                if disease in st.session_state.disease_counts:
                    st.session_state.disease_counts[disease] += 1
                
                color = "green" if alert_level == "Normal" else "orange" if alert_level == "Warning" else "red"
                
                # 3. 渲染顶部的诊断结果
                ai_status_placeholder.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; border: 2px solid {color}; background-color: rgba(255,255,255,0.05);">
                    <h4 style="margin:0; color:{color};">当前诊断: <strong>{disease}</strong> (置信度: {confidence*100:.1f}%)</h4>
                    <p style="margin:5px 0 0 0;">系统计算心率: {hr} BPM | 预警等级: {alert_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                ai_status_placeholder.error(f"后端报错: {response.status_code}")
                
        except requests.exceptions.RequestException:
            ai_status_placeholder.error("🚨 无法连接到后端！请确认你的另一个终端里 `py -m uvicorn main:app --reload` 正在运行。")
            st.session_state.is_running = False
            break
        
        # 4. 画出波形图 (把 750 个点完整画出来，展现心跳细节)
        ecg_chart.line_chart(ecg_window, height=180)
        eda_chart.line_chart(eda_window[-150:], height=120, color="#2ca02c")
        
        # 5. 更新饼图
        pie_df = pd.DataFrame(list(st.session_state.disease_counts.items()), columns=['疾病', '次数'])
        fig = px.pie(pie_df, values='次数', names='疾病', hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        pie_chart_placeholder.plotly_chart(fig, use_container_width=True, key=str(time.time()))
        
        time.sleep(0.1) # 刷新频率，让心电图跑得更流畅