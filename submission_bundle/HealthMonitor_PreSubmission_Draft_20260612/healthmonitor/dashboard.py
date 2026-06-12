"""Competition dashboard for the ECG arrhythmia health-warning system."""
from __future__ import annotations

import json
import os
import inspect
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from .constants import CLASS_COLORS, CLASS_NAMES, TARGET_FS
from .demo_replay import DEMO_CASES, load_case_signal
from .paths import DEMO_DIR, EVIDENCE_DIR, PROJECT_ROOT


API_URL = os.environ.get("ECG_API_URL", "http://127.0.0.1:8000/api/predict")
METRIC_REGISTRY = PROJECT_ROOT / "artifacts" / "metric_registry.json"
DISPLAY_CASES = ["119", "100", "201"]
BOUNDARY_CASES = ["223", "209"]
TAB_MONITOR = "监测预警"
TAB_REPLAY = "精选病例"
TAB_EVIDENCE = "验证证据"


CASE_LABELS = {
    "119": "MIT-BIH 119 | PVC正式报警",
    "100": "MIT-BIH 100 | 正常无报警",
    "201": "MIT-BIH 201 | AFib辅助研究",
}

RISK_MESSAGES = {
    "normal": "当前未见持续风险升高，建议继续观察。",
    "fluctuation": "检测到短时风险波动，请保持静息并检查电极接触。",
    "alert": "未来5分钟整体心律失常风险持续升高，建议由专业人员复核或就医评估。",
    "poor_signal": "当前信号质量不足，请检查电极、导联或采集设备。",
}
SAFETY_MESSAGE = (
    "如伴胸痛、晕厥、呼吸困难或持续明显心悸，请立即联系急救或前往医院。"
)


st.set_page_config(
    page_title="ECG健康预警系统",
    page_icon="ECG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #182230;
      --muted: #667085;
      --line: #d9dee7;
      --paper: #ffffff;
      --wash: #f4f6f8;
      --teal: #087f78;
      --amber: #b86b00;
      --red: #b42318;
      --blue: #2563eb;
    }
    .stApp { background: var(--wash); color: var(--ink); }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    [data-testid="stSidebar"] { background: #14212b; }
    [data-testid="stSidebar"] * { color: #eef6f6; }
    [data-testid="stMetric"] {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.7rem 0.85rem;
      box-shadow: 0 8px 20px rgba(16, 24, 40, 0.04);
    }
    .hero {
      background: #ffffff;
      border: 1px solid var(--line);
      border-left: 6px solid var(--teal);
      border-radius: 8px;
      padding: 1rem 1.15rem;
      margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.75rem; }
    .hero p { margin: .35rem 0 0; color: var(--muted); max-width: 900px; }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: .18rem .55rem;
      margin: 0 .3rem .25rem 0;
      font-size: .76rem;
      font-weight: 700;
      background: #e6f4f1;
      color: #087f78;
      border: 1px solid #bfe2dc;
    }
    .badge.research { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
    .badge.boundary { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: .9rem 1rem;
      min-height: 108px;
      box-shadow: 0 8px 20px rgba(16, 24, 40, 0.035);
    }
    .panel-label { color: var(--muted); font-size: .78rem; font-weight: 700; }
    .panel-value { color: var(--ink); font-size: 1.22rem; font-weight: 750; margin-top: .22rem; }
    .panel-note { color: var(--muted); font-size: .78rem; margin-top: .35rem; line-height: 1.45; }
    .risk-normal { border-left: 6px solid #087f78; }
    .risk-fluctuation { border-left: 6px solid #d28700; }
    .risk-alert { border-left: 6px solid #b42318; }
    .small-note { color: var(--muted); font-size: .84rem; line-height: 1.55; }
    .footer-note {
      margin-top: 1rem;
      padding: .8rem 1rem;
      background: #fff7e8;
      border: 1px solid #f1d7a8;
      border-radius: 8px;
      color: #694100;
      font-size: .86rem;
    }
    .timeline-row {
      border-left: 3px solid #087f78;
      padding: .12rem 0 .6rem .75rem;
      margin-left: .2rem;
      color: #344054;
      font-size: .86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_json_file(path_str, mtime_ns):
    with Path(path_str).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest():
    path = DEMO_DIR / "replay_manifest.json"
    return cached_json_file(str(path), path.stat().st_mtime_ns)


def load_registry():
    return cached_json_file(str(METRIC_REGISTRY), METRIC_REGISTRY.stat().st_mtime_ns)


def load_replay(case_id):
    path = DEMO_DIR / f"mitdb_{case_id}.json"
    return cached_json_file(str(path), path.stat().st_mtime_ns)


@st.cache_data
def load_signal(case_id):
    signal, _, _ = load_case_signal(DEMO_CASES[case_id], str(PROJECT_ROOT))
    return signal


def load_json(path):
    json_path = Path(path)
    return cached_json_file(str(json_path), json_path.stat().st_mtime_ns)


def pct(value, digits=1):
    return f"{100 * float(value):.{digits}f}%"


def tier_badge(tier):
    if tier == "formal_success":
        return "<span class='badge'>正式冻结策略</span>"
    if tier == "pilot_success":
        return "<span class='badge research'>辅助研究能力</span>"
    return "<span class='badge boundary'>探索性边界</span>"


def quality_text(quality):
    return {"good": "良好", "limited": "受限", "poor": "不足"}.get(
        quality.get("level"), "未知"
    )


def status_title(status):
    return {
        "normal": "无正式报警",
        "fluctuation": "短时风险波动",
        "alert": "正式报警已触发",
    }.get(status, "状态待核验")


def probability_chart(probabilities):
    frame = pd.DataFrame({
        "类别": CLASS_NAMES,
        "概率": probabilities,
        "颜色": CLASS_COLORS,
    })
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("概率:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("类别:N", sort=None, title=None),
            color=alt.Color("颜色:N", scale=None, legend=None),
            tooltip=["类别", alt.Tooltip("概率:Q", format=".1%")],
        )
        .properties(height=190, background="#ffffff")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#475467", titleColor="#344054", gridColor="#eaecf0")
    )


def waveform_chart(case_id, timestamp_sec):
    signal = load_signal(case_id)
    center = int(timestamp_sec * TARGET_FS)
    start = max(0, center - 4 * TARGET_FS)
    end = min(len(signal), center + 4 * TARGET_FS)
    segment = signal[start:end]
    frame = pd.DataFrame({
        "相对时间": (np.arange(len(segment)) + start - center) / TARGET_FS,
        "心电信号": segment,
    })
    return (
        alt.Chart(frame)
        .mark_line(color="#087f78", strokeWidth=1.2)
        .encode(
            x=alt.X("相对时间:Q", title="相对当前推理点（秒）"),
            y=alt.Y("心电信号:Q", title="幅值"),
            tooltip=[
                alt.Tooltip("相对时间:Q", format=".2f"),
                alt.Tooltip("心电信号:Q", format=".3f"),
            ],
        )
        .properties(height=225, background="#ffffff")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#475467", titleColor="#344054", gridColor="#eaecf0")
    )


def risk_timeline_chart(replay, selected_index=None):
    rows = replay["rows"]
    frame = pd.DataFrame({
        "时间（分钟）": [row["timestamp_min"] for row in rows],
        "原始风险": [row["overall_risk_raw"] for row in rows],
        "EWMA风险": [row["policy"]["ewma_risk"] for row in rows],
    })
    long = frame.melt("时间（分钟）", var_name="序列", value_name="风险")
    base = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.1)
        .encode(
            x=alt.X("时间（分钟）:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("风险:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color(
                "序列:N",
                scale=alt.Scale(
                    domain=["原始风险", "EWMA风险"],
                    range=["#98a2b3", "#087f78"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("时间（分钟）:Q", format=".2f"),
                "序列:N",
                alt.Tooltip("风险:Q", format=".1%"),
            ],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"阈值": [0.10]}))
        .mark_rule(color="#b42318", strokeDash=[5, 4])
        .encode(y="阈值:Q")
    )
    chart = base + threshold
    if selected_index is not None:
        selected_time = rows[selected_index]["timestamp_min"]
        chart += (
            alt.Chart(pd.DataFrame({"时间（分钟）": [selected_time]}))
            .mark_rule(color="#182230", strokeWidth=1)
            .encode(x="时间（分钟）:Q")
        )
    return (
        chart.properties(height=225, background="#ffffff")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#475467", titleColor="#344054", gridColor="#eaecf0")
        .configure_legend(labelColor="#475467", titleColor="#344054")
    )


def cam_chart(row):
    cam = np.asarray(row.get("cam", []), dtype=float)
    frame = pd.DataFrame({"历史窗口": np.arange(len(cam)), "关注强度": cam})
    return (
        alt.Chart(frame)
        .mark_area(color="#087f78", opacity=0.55)
        .encode(
            x=alt.X("历史窗口:Q", title="过去10分钟的30秒窗口"),
            y=alt.Y("关注强度:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            tooltip=["历史窗口", alt.Tooltip("关注强度:Q", format=".3f")],
        )
        .properties(height=120, background="#ffffff")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#475467", titleColor="#344054", gridColor="#eaecf0")
    )


def event_log(rows, selected_index):
    events = []
    for row in rows[: selected_index + 1]:
        transition = row["policy"].get("transition")
        if transition in {"alarm_started", "alarm_cleared"}:
            events.append({
                "time": row["timestamp_min"],
                "text": "正式报警开始" if transition == "alarm_started" else "正式报警解除",
            })
    if not events:
        st.caption("所选时间之前没有正式报警状态变化。")
        return
    for event in reversed(events[-6:]):
        st.markdown(
            f"<div class='timeline-row'><b>{event['time']:.2f} min</b> | {event['text']}</div>",
            unsafe_allow_html=True,
        )


def live_api_check(case_id, row):
    signal = load_signal(case_id)
    current_pt = int(row["timestamp_sec"] * TARGET_FS)
    payload_signal = signal[max(0, current_pt - 180000):current_pt]
    if len(payload_signal) < 150000:
        payload_signal = np.pad(payload_signal, (150000 - len(payload_signal), 0), "constant")
    response = requests.post(
        API_URL,
        json={"ecg": payload_signal.astype(float).tolist()},
        timeout=30,
    )
    response.raise_for_status()
    live = response.json()
    max_error = float(
        np.max(np.abs(np.asarray(live["probabilities"]) - np.asarray(row["probabilities_fut"])))
    )
    return live, max_error


def streamlit_accepts(func, parameter):
    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def render_tabs(labels, default_tab):
    if streamlit_accepts(st.tabs, "default"):
        return st.tabs(labels, default=default_tab)
    return st.tabs(labels)


def render_dataframe(frame):
    if streamlit_accepts(st.dataframe, "width"):
        return st.dataframe(frame, width="stretch", hide_index=True)
    return st.dataframe(frame, use_container_width=True, hide_index=True)


manifest = load_manifest()
registry = load_registry()
default_indices = manifest.get("default_indices", {"119": 4, "100": 17, "201": 10})

st.sidebar.markdown("## ECG健康预警")
selected_case = st.sidebar.selectbox(
    "精选病例",
    options=DISPLAY_CASES,
    index=0,
    format_func=lambda case_id: CASE_LABELS[case_id],
)
replay = load_replay(selected_case)
max_index = len(replay["rows"]) - 1
selected_index = st.sidebar.slider(
    "冻结回放推理点",
    min_value=0,
    max_value=max_index,
    value=min(max_index, int(default_indices.get(selected_case, 0))),
)
st.sidebar.caption("每个点间隔10秒。正式报警由冻结策略按时间顺序计算。")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='small-note'>系统定位：未来5分钟整体心律失常风险预警。"
    "AFib仅作为外部研究支持的辅助方向展示，不接入正式报警。</div>",
    unsafe_allow_html=True,
)

row = replay["rows"][selected_index]
quality = row["signal_quality"]
policy = row["policy"]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TAB_MONITOR


def keep_replay_tab():
    st.session_state["active_tab"] = TAB_REPLAY
    st.session_state["api_expanded"] = True

st.markdown(
    """
    <div class="hero">
      <h1>ECG心律失常健康预警系统</h1>
      <p>基于过去10分钟单导联ECG，对未来5分钟整体心律失常风险进行分层提示，并展示冻结证据与能力边界。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

monitor_tab, replay_tab, evidence_tab = render_tabs(
    [TAB_MONITOR, TAB_REPLAY, TAB_EVIDENCE],
    st.session_state.get("active_tab", TAB_MONITOR),
)

with monitor_tab:
    st.markdown(tier_badge(replay["tier"]), unsafe_allow_html=True)
    st.subheader(f"{replay['short_title']} | {row['timestamp_min']:.2f} min")

    status = policy["status"]
    status_class = {
        "normal": "risk-normal",
        "fluctuation": "risk-fluctuation",
        "alert": "risk-alert",
    }.get(status, "risk-normal")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f"<div class='panel'><div class='panel-label'>正式报警状态</div>"
            f"<div class='panel-value'>{status_title(status)}</div>"
            f"<div class='panel-note'>阈值 10% | 连续 {policy['consecutive_count']} 点</div></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='panel'><div class='panel-label'>整体风险</div>"
            f"<div class='panel-value'>{pct(row['overall_risk_raw'])}</div>"
            f"<div class='panel-note'>EWMA {pct(policy['ewma_risk'])}</div></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"<div class='panel'><div class='panel-label'>信号质量</div>"
            f"<div class='panel-value'>{pct(quality['score'], 0)}</div>"
            f"<div class='panel-note'>{quality_text(quality)} | 历史窗口已满足</div></div>",
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f"<div class='panel'><div class='panel-label'>模型投票数</div>"
            f"<div class='panel-value'>{row['future_vote_count']}/3</div>"
            f"<div class='panel-note'>风险离散度 {row['risk_std']:.3f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='panel {status_class}' style='margin-top:1rem;'>"
        f"<div class='panel-label'>处置提示</div>"
        f"<div class='panel-value'>{status_title(status)}</div>"
        f"<div class='panel-note'>{RISK_MESSAGES[status]}</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("#### 单导联ECG波形")
        st.altair_chart(
            waveform_chart(selected_case, row["timestamp_sec"]),
            use_container_width=True,
        )
        st.markdown("#### 风险时间线")
        st.altair_chart(
            risk_timeline_chart(replay, selected_index),
            use_container_width=True,
        )
    with right:
        st.markdown("#### 未来窗口类别分布")
        st.altair_chart(
            probability_chart(row["probabilities_fut"]),
            use_container_width=True,
        )
        if selected_case == "201":
            st.metric("AFib方向分数", pct(row["afib_direction_score"]))
            st.caption("外部研究支持：LTAFDB 83记录主分析、84记录敏感性分析。")
        st.markdown("#### 报警事件")
        event_log(replay["rows"], selected_index)

    st.markdown("#### 关注窗口")
    st.altair_chart(cam_chart(row), use_container_width=True)
    st.markdown(
        f"<div class='footer-note'><b>安全提示：</b>{SAFETY_MESSAGE}</div>",
        unsafe_allow_html=True,
    )

with replay_tab:
    st.markdown(tier_badge(replay["tier"]), unsafe_allow_html=True)
    st.subheader(replay["title"])
    st.write(replay["claim"])
    st.caption(replay["boundary"])

    summary = replay["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("冻结推理点", summary["points"])
    c2.metric("正式报警点", summary["official_alarm_points"])
    c3.metric("当前目标Argmax", f"{summary['current_target_argmax_points']}/{summary['points']}")
    c4.metric("未来目标Argmax", f"{summary['future_target_argmax_points']}/{summary['points']}")

    st.altair_chart(
        risk_timeline_chart(replay, selected_index),
        use_container_width=True,
    )

    with st.expander("实时API核验", expanded=st.session_state.get("api_expanded", False)):
        st.caption("API不可用时会明确报错；冻结回放仍可独立运行。")
        if st.button("调用实时API", type="secondary", on_click=keep_replay_tab):
            try:
                live, max_error = live_api_check(selected_case, row)
                st.success(f"API调用成功，与冻结未来概率最大绝对误差 {max_error:.3e}。")
                st.json({
                    "input_status": live.get("input_status"),
                    "signal_quality": live.get("signal_quality"),
                    "overall_risk_raw": live.get("overall_risk_raw"),
                    "afib_direction_score": live.get("afib_direction_score"),
                    "future_vote_count": live.get("future_vote_count"),
                    "risk_std": live.get("risk_std"),
                    "policy_config_version": live.get("policy_config_version"),
                })
            except Exception as exc:
                st.error(f"API核验失败：{type(exc).__name__}: {exc}")

with evidence_tab:
    formal = registry["formal_capability"]
    afib = registry["afib_auxiliary_research"]
    st.subheader("能力分层")
    capability = pd.DataFrame([
        {
            "层级": "正式能力",
            "内容": "未来5分钟整体心律失常风险",
            "证据": "锁定测试 + 事件级报警评估",
            "前端行为": "接入正式EWMA+k=2报警",
        },
        {
            "层级": "辅助研究能力",
            "内容": "AFib方向分数",
            "证据": "LTAFDB 83记录主分析 + 84记录敏感性分析",
            "前端行为": "仅在AFib病例和证据页展示",
        },
        {
            "层级": "探索性边界",
            "内容": "VT/VF、AT/SVT专项方向",
            "证据": "未进入正式锁定测试",
            "前端行为": "不作为现场主病例展示",
        },
    ])
    render_dataframe(capability)

    st.markdown("#### 正式整体报警")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("锁定测试事件召回", pct(formal["locked_test_event_recall"]))
    o2.metric("事件起点召回", pct(formal["locked_test_incident_event_recall"]))
    o3.metric("中位提前量", f"{formal['locked_test_median_lead_time_sec']:.0f}s")
    o4.metric(
        "误报episode/患者小时",
        f"{formal['locked_test_false_alert_episodes_per_patient_hour']:.2f}",
    )

    st.markdown("#### AFib外部研究支持")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("外部AUROC", f"{afib['dominant_auroc']:.3f}")
    a2.metric("AFib事件召回", pct(afib["event_recall"]))
    a3.metric("中位提前量", f"{afib['median_lead_time_sec']:.0f}s")
    a4.metric(
        "误报episode/记录小时",
        f"{afib['false_alert_episodes_per_patient_hour']:.3f}",
    )
    st.caption(afib["scope"])

    with st.expander("能力边界：223与209", expanded=False):
        for case_id in BOUNDARY_CASES:
            boundary = load_replay(case_id)
            summary = boundary["summary"]
            st.markdown(f"**{boundary['short_title']}**")
            st.write(boundary["boundary"])
            st.caption(
                f"正式报警点 {summary['official_alarm_points']}/{summary['points']}；"
                f"未来目标Argmax {summary['future_target_argmax_points']}/{summary['points']}。"
            )

    st.markdown(
        f"<div class='footer-note'><b>系统边界：</b>模型不替代临床诊断。{SAFETY_MESSAGE}</div>",
        unsafe_allow_html=True,
    )
