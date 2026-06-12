"""Generate pre-submission report and project manual DOCX/PDF drafts."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from healthmonitor.paths import ARTIFACTS_DIR, DOCS_DIR, PROJECT_ROOT


REPORT_DOCX = DOCS_DIR / "ECG心律失常健康预警系统竞赛报告_待提交草稿.docx"
MANUAL_DOCX = DOCS_DIR / "ECG心律失常健康预警系统项目说明书_待提交草稿.docx"


def load_registry():
    return json.loads((ARTIFACTS_DIR / "metric_registry.json").read_text(encoding="utf-8"))


def font(size=28, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_box(draw, xy, text, fill="#ffffff", outline="#1f4d78", text_fill="#182230", width=3):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=width)
    lines = text.split("\n")
    f = font(25, bold=True)
    line_h = 32
    total = len(lines) * line_h
    y = y1 + (y2 - y1 - total) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=f)
        draw.text((x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2, y), line, font=f, fill=text_fill)
        y += line_h


def arrow(draw, start, end, fill="#475467", width=5):
    draw.line([start, end], fill=fill, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    pts = [
        end,
        (end[0] - size * math.cos(ang - 0.45), end[1] - size * math.sin(ang - 0.45)),
        (end[0] - size * math.cos(ang + 0.45), end[1] - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def make_figures(registry):
    fig_dir = ARTIFACTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    big_font = font(34, True)
    mid_font = font(24, True)
    small_font = font(20)

    def base(path, title):
        img = Image.new("RGB", (1600, 900), "#f6f8fb")
        d = ImageDraw.Draw(img)
        d.text((60, 42), title, font=big_font, fill="#0b2545")
        return img, d

    img, d = base(fig_dir / "architecture.png", "系统架构")
    boxes = [
        ((90, 170, 360, 310), "公开ECG\nMIT-BIH/LTAFDB"),
        ((460, 170, 730, 310), "信号预处理\n重采样/滤波"),
        ((830, 170, 1100, 310), "三模型集成\n当前头+未来头"),
        ((1200, 170, 1470, 310), "正式报警\nEWMA+k=2"),
        ((460, 500, 730, 640), "冻结演示资产\nJSON+NPZ"),
        ((830, 500, 1100, 640), "指标注册表\n报告/前端共用"),
        ((1200, 500, 1470, 640), "前端三页\n预警/病例/证据"),
    ]
    for xy, text in boxes:
        draw_box(d, xy, text)
    for s, e in [((360, 240), (460, 240)), ((730, 240), (830, 240)), ((1100, 240), (1200, 240)), ((730, 570), (830, 570)), ((1100, 570), (1200, 570)), ((965, 310), (965, 500))]:
        arrow(d, s, e)
    img.save(fig_dir / "architecture.png")

    img, d = base(fig_dir / "data_window.png", "数据窗口协议")
    d.line([(160, 430), (1440, 430)], fill="#667085", width=8)
    d.rounded_rectangle((160, 340, 1040, 520), radius=18, fill="#e6f4f1", outline="#087f78", width=4)
    d.rounded_rectangle((1040, 340, 1440, 520), radius=18, fill="#fff7e8", outline="#b86b00", width=4)
    d.text((420, 400), "过去10分钟历史输入", font=mid_font, fill="#075e59")
    d.text((1120, 400), "未来5分钟预警目标", font=mid_font, fill="#7a4b00")
    for x in range(160, 1041, 110):
        d.line([(x, 320), (x, 540)], fill="#ffffff", width=2)
    d.text((185, 555), "30秒窗口，15秒步长，共39个历史窗口", font=small_font, fill="#344054")
    img.save(fig_dir / "data_window.png")

    img, d = base(fig_dir / "model_structure.png", "三模型集成结构")
    draw_box(d, (120, 240, 400, 380), "30秒ECG\n+ RR特征", fill="#ffffff")
    draw_box(d, (520, 120, 800, 250), "成员模型 A\nseed0", fill="#eff6ff", outline="#2563eb")
    draw_box(d, (520, 320, 800, 450), "成员模型 B\nseed1", fill="#eff6ff", outline="#2563eb")
    draw_box(d, (520, 520, 800, 650), "成员模型 C\nseed4", fill="#eff6ff", outline="#2563eb")
    draw_box(d, (950, 260, 1230, 430), "Logits平均\n概率分布", fill="#f4f6f9")
    draw_box(d, (1320, 260, 1500, 430), "风险/投票\n离散度", fill="#e6f4f1", outline="#087f78")
    for y in [185, 385, 585]:
        arrow(d, (400, 310), (520, y))
        arrow(d, (800, y), (950, 345))
    arrow(d, (1230, 345), (1320, 345))
    img.save(fig_dir / "model_structure.png")

    img, d = base(fig_dir / "alarm_logic.png", "正式报警逻辑")
    steps = [
        ("未来Normal概率", "取 1 - P(Normal)"),
        ("风险平滑", "EWMA α=0.65"),
        ("阈值判断", "风险 ≥ 10%"),
        ("连续确认", "连续2点触发"),
        ("状态输出", "正常/波动/报警"),
    ]
    x = 90
    for title, body in steps:
        draw_box(d, (x, 300, x + 240, 470), f"{title}\n{body}", fill="#ffffff")
        if x < 1170:
            arrow(d, (x + 240, 385), (x + 310, 385))
        x += 310
    img.save(fig_dir / "alarm_logic.png")

    # Risk timeline from frozen PVC demo.
    replay = json.loads((ARTIFACTS_DIR / "demo" / "mitdb_119.json").read_text(encoding="utf-8"))
    rows = replay["rows"]
    img, d = base(fig_dir / "pvc_risk_timeline.png", "MIT-BIH 119 冻结风险时间线")
    plot = (140, 170, 1460, 720)
    d.rectangle(plot, fill="#ffffff", outline="#d0d5dd", width=2)
    xs = [r["timestamp_min"] for r in rows]
    ys = [r["overall_risk_raw"] for r in rows]
    ew = [r["policy"]["ewma_risk"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    def pt(x, y):
        px = plot[0] + (x - xmin) / (xmax - xmin) * (plot[2] - plot[0])
        py = plot[3] - y * (plot[3] - plot[1])
        return px, py
    for frac in [0, .25, .5, .75, 1.0]:
        y = plot[3] - frac * (plot[3] - plot[1])
        d.line([(plot[0], y), (plot[2], y)], fill="#eaecf0", width=1)
        d.text((70, y - 12), f"{frac:.0%}", font=small_font, fill="#475467")
    d.line([pt(xmin, 0.10), pt(xmax, 0.10)], fill="#b42318", width=4)
    d.line([pt(xs[i], ys[i]) for i in range(len(xs))], fill="#98a2b3", width=5)
    d.line([pt(xs[i], ew[i]) for i in range(len(xs))], fill="#087f78", width=5)
    d.text((180, 745), "灰色：原始整体风险；绿色：EWMA；红线：10%正式阈值", font=small_font, fill="#344054")
    img.save(fig_dir / "pvc_risk_timeline.png")

    # Three simple UI snapshot figures using registry values.
    for key, title, filename in [
        ("pvc_alarm", "监测预警页：PVC正式报警", "ui_monitoring.png"),
        ("normal_no_alarm", "精选病例页：正常无报警", "ui_cases.png"),
        ("afib_auxiliary", "验证证据页：AFib辅助研究", "ui_evidence.png"),
    ]:
        case = registry["demo_cases"][key]
        img, d = base(fig_dir / filename, title)
        draw_box(d, (110, 170, 480, 320), f"病例\n{case['short_title']}", fill="#ffffff")
        draw_box(d, (550, 170, 850, 320), f"整体风险\n{case['overall_risk_raw']:.1%}", fill="#fff7e8", outline="#b86b00")
        draw_box(d, (920, 170, 1220, 320), f"信号质量\n{case['signal_quality_score']:.0%}", fill="#e6f4f1", outline="#087f78")
        draw_box(d, (1290, 170, 1500, 320), f"投票\n{case['future_vote_count']}/3", fill="#eff6ff", outline="#2563eb")
        d.text((120, 410), "冻结演示资产可独立运行；实时API核验为可选功能。", font=mid_font, fill="#344054")
        img.save(fig_dir / filename)


def set_run_font(run, name, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_paragraph(doc, text="", style=None, first_line=False):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        set_run_font(r, "宋体", 12)
    p.paragraph_format.line_spacing = Pt(20)
    p.paragraph_format.space_after = Pt(3)
    if first_line:
        p.paragraph_format.first_line_indent = Pt(24)
    return p


def add_mixed_paragraph(doc, parts, first_line=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = Pt(20)
    p.paragraph_format.space_after = Pt(3)
    if first_line:
        p.paragraph_format.first_line_indent = Pt(24)
    for text, latin in parts:
        r = p.add_run(text)
        set_run_font(r, "Times New Roman" if latin else "宋体", 12)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    set_run_font(r, "宋体", 14 if level == 1 else 12.5, bold=True, color="1F4D78")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = Pt(20)
    return p


def add_caption(doc, kind, number, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} {number}  {text}")
    set_run_font(r, "宋体", 10.5, bold=False)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    return p


def style_table(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = Pt(16)
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    set_run_font(run, "宋体", 9.5)
    for cell in table.rows[0].cells:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E8EEF5")
        cell._tc.get_or_add_tcPr().append(shading)
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True


def add_table(doc, caption_no, caption, headers, rows):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"表 {caption_no}  {caption}")
    set_run_font(r, "宋体", 10.5)
    table = doc.add_table(rows=1, cols=len(headers))
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    style_table(table)
    return table


def setup_doc(doc, report=True):
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.7)
    sec.right_margin = Cm(2.7)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal.font.size = Pt(12)
    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = styles[name]
        style.font.name = "宋体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("第  页")
    set_run_font(run, "宋体", 9)


def add_picture(doc, path, number, caption, width=6.1):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, "图", number, caption)


def build_report(registry):
    doc = Document()
    setup_doc(doc, report=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ECG心律失常健康预警系统\n竞赛报告（待提交草稿）")
    set_run_font(r, "宋体", 20, bold=True)
    add_paragraph(doc, "作品ID：{{作品ID待填写}}    学生类型：{{学生类型待填写}}    赛道：{{赛道待填写}}")
    add_paragraph(doc, "封面占位符未补齐，本文件仅作为待提交草稿，不作为最终提交版。")
    doc.add_page_break()

    add_heading(doc, "目录", 1)
    add_paragraph(doc, "（请在 Word 中更新自动目录字段后提交。）")
    doc.add_page_break()

    add_heading(doc, "一、作品概述", 1)
    add_mixed_paragraph(doc, [("本系统面向单导联 ECG 连续监测场景，正式能力限定为", False), ("未来5分钟整体心律失常风险预警", False), ("。系统不输出临床诊断结论，不替代医生判读；AFib 仅作为外部研究支持的辅助方向展示。", False)], True)
    add_table(doc, 1, "能力分层与宣传边界", ["层级", "内容", "前端行为", "边界"], [
        ["正式能力", "未来5分钟整体心律失常风险", "接入正式报警", "不宣称类别诊断"],
        ["辅助研究", "AFib方向分数", "病例与证据页展示", "不接入正式报警"],
        ["探索边界", "VT/VF、AT/SVT方向", "折叠说明", "不作为现场成功病例"],
    ])
    add_picture(doc, ARTIFACTS_DIR / "figures" / "architecture.png", 1, "系统整体架构")

    add_heading(doc, "二、系统设计与实现", 1)
    add_mixed_paragraph(doc, [("系统将过去10分钟 ECG 划分为 39 个 30 秒窗口，并以 15 秒步长形成轨迹输入；未来5分钟标签用于训练预警目标。", False)], True)
    add_picture(doc, ARTIFACTS_DIR / "figures" / "data_window.png", 2, "历史窗口与未来预警目标")
    add_mixed_paragraph(doc, [("模型采用三成员集成，输出当前节律参考头和未来趋势头。正式风险定义为 ", False), ("1 - P(future Normal)", True), ("，并同时给出模型投票数与风险离散度。", False)], True)
    add_picture(doc, ARTIFACTS_DIR / "figures" / "model_structure.png", 3, "三模型集成与输出结构")
    add_picture(doc, ARTIFACTS_DIR / "figures" / "alarm_logic.png", 4, "正式报警策略")

    formal = registry["formal_capability"]
    add_table(doc, 2, "正式冻结策略参数", ["项目", "取值"], [
        ["风险定义", formal["score"]],
        ["阈值", f"{formal['threshold']:.2f}"],
        ["EWMA α", f"{formal['ewma_alpha']:.2f}"],
        ["连续确认", f"{formal['consecutive_k']} 点"],
        ["对外版本", registry["policy_config_version"]],
    ])

    add_heading(doc, "三、测试与结果", 1)
    add_table(doc, 3, "锁定测试事件级表现", ["指标", "结果"], [
        ["事件召回", f"{formal['locked_test_event_recall']:.1%}"],
        ["事件起点召回", f"{formal['locked_test_incident_event_recall']:.1%}"],
        ["中位提前量", f"{formal['locked_test_median_lead_time_sec']:.0f}s"],
        ["误报episode/患者小时", f"{formal['locked_test_false_alert_episodes_per_patient_hour']:.2f}"],
    ])
    add_table(doc, 4, "PVC验证表现", ["指标", "结果"], [
        ["PVC召回", f"{formal['validation_future_pvc_recall']:.1%}"],
        ["PVC AUROC", f"{formal['validation_future_pvc_auroc']:.3f}"],
        ["PVC AP", f"{formal['validation_future_pvc_ap']:.3f}"],
    ])
    afib = registry["afib_auxiliary_research"]
    add_table(doc, 5, "AFib外部研究支持", ["指标", "结果"], [
        ["主分析记录数", afib["records_primary"]],
        ["敏感性分析记录数", afib["records_sensitivity"]],
        ["AUROC", f"{afib['dominant_auroc']:.3f}"],
        ["事件召回", f"{afib['event_recall']:.1%}"],
        ["中位提前量", f"{afib['median_lead_time_sec']:.0f}s"],
    ])
    add_picture(doc, ARTIFACTS_DIR / "figures" / "pvc_risk_timeline.png", 5, "PVC病例风险时间线")
    add_picture(doc, ARTIFACTS_DIR / "explainability" / "pvc_explainability_mitdb_119.png", 6, "PVC解释性关注图")

    add_heading(doc, "四、前端演示与病例", 1)
    cases = registry["demo_cases"]
    add_table(doc, 6, "精选冻结病例", ["病例", "默认点", "核心结果"], [
        ["MIT-BIH 119", cases["pvc_alarm"]["display_index"], f"PVC {cases['pvc_alarm']['probability_pvc']:.3f}，风险 {cases['pvc_alarm']['overall_risk_raw']:.3f}，报警触发"],
        ["MIT-BIH 100", cases["normal_no_alarm"]["display_index"], f"Normal {cases['normal_no_alarm']['probability_normal']:.3f}，风险 {cases['normal_no_alarm']['overall_risk_raw']:.3f}，无报警"],
        ["MIT-BIH 201", cases["afib_auxiliary"]["display_index"], f"AFib方向 {cases['afib_auxiliary']['afib_direction_score']:.3f}，未来AFib {cases['afib_auxiliary']['probability_afib']:.3f}"],
    ])
    add_picture(doc, ARTIFACTS_DIR / "figures" / "ui_monitoring.png", 7, "前端监测预警页")
    add_picture(doc, ARTIFACTS_DIR / "figures" / "ui_cases.png", 8, "前端精选病例页")
    add_picture(doc, ARTIFACTS_DIR / "figures" / "ui_evidence.png", 9, "前端验证证据页")

    add_heading(doc, "五、创新与局限", 1)
    add_paragraph(doc, "创新点包括未来窗口预警建模、三模型集成、正式报警策略与冻结演示资产解耦、指标注册表统一驱动报告和前端。", first_line=True)
    add_paragraph(doc, "局限性包括长尾类别证据不足、单导联输入对信号质量敏感、外部AFib结果仍为研究支持而非临床有效性证明。", first_line=True)
    add_table(doc, 7, "验收与复现材料", ["材料", "路径"], [
        ["指标注册表", "artifacts/metric_registry.json"],
        ["正式模型", "models/official_ensemble/"],
        ["冻结演示资产", "artifacts/demo/"],
        ["关键证据", "artifacts/evidence/"],
    ])

    add_heading(doc, "六、总结", 1)
    add_paragraph(doc, "本系统已冻结正式策略和演示资产，可在无完整原始数据库的情况下启动前端并复现实例展示。提交前仍需补齐封面占位符，并在 Word 中更新目录、页码和交叉引用字段。", first_line=True)

    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] Moody G B, Mark R G. The impact of the MIT-BIH Arrhythmia Database[J]. IEEE Engineering in Medicine and Biology Magazine, 2001, 20(3): 45-50.",
        "[2] Goldberger A L, Amaral L A N, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet[J]. Circulation, 2000, 101(23): e215-e220.",
        "[3] Wagner P, Strodthoff N, Bousseljot R D, et al. PTB-XL, a large publicly available electrocardiography dataset[J]. Scientific Data, 2020, 7: 154.",
        "[4] Selvaraju R R, Cogswell M, Das A, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization[C]. ICCV, 2017.",
        "[5] Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks[C]. ICML, 2017.",
    ]
    for ref in refs:
        add_paragraph(doc, ref)
    doc.save(REPORT_DOCX)


def build_manual(registry):
    doc = Document()
    setup_doc(doc, report=False)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ECG心律失常健康预警系统项目说明书\n待提交草稿")
    set_run_font(r, "宋体", 18, bold=True)
    add_paragraph(doc, "封面占位符未补齐前，本说明书与提交包均按草稿状态管理。")
    add_heading(doc, "一、项目定位", 1)
    add_paragraph(doc, "正式能力为未来5分钟整体心律失常风险预警；禁止宣传为临床诊断、单类别确诊或急救替代工具。", first_line=True)
    add_heading(doc, "二、最终目录与用途", 1)
    add_table(doc, 1, "保留目录说明", ["目录", "用途"], [
        ["healthmonitor/", "核心模型、信号、策略、回放和API模块"],
        ["scripts/", "数据、训练、评估和外部验证脚本"],
        ["tests/", "策略、服务推理和冻结回放验收"],
        ["models/official_ensemble/", "正式三模型权重"],
        ["artifacts/", "演示、证据、策略、图源和解释性材料"],
        ["docs/", "报告与说明书"],
        ["submission_bundle/", "待提交草稿包"],
    ])
    add_heading(doc, "三、数据与模型流程", 1)
    add_paragraph(doc, "数据流程包括公开数据库下载、信号清洗、轨迹样本构建、三模型训练、锁定评估、外部AFib研究评估和冻结演示生成。", first_line=True)
    add_picture(doc, ARTIFACTS_DIR / "figures" / "architecture.png", 1, "开发与运行流程")
    add_heading(doc, "四、API字段与前端数据流", 1)
    add_table(doc, 2, "关键API字段", ["字段", "说明"], [
        ["probabilities", "未来5分钟类别概率，保持旧字段兼容"],
        ["overall_risk_raw", "整体风险，等于 1 - P(future Normal)"],
        ["signal_quality", "信号质量分数、等级和原因"],
        ["future_vote_count", "三模型未来类别投票数"],
        ["risk_std", "三模型风险离散度"],
        ["policy_config_version", registry["policy_config_version"]],
    ])
    add_heading(doc, "五、安装与启动", 1)
    add_paragraph(doc, "安装依赖：pip install -r requirements.txt")
    add_paragraph(doc, "启动前端：powershell -ExecutionPolicy Bypass -File start_demo.ps1")
    add_paragraph(doc, "可选API：python -m uvicorn healthmonitor.main:app --host 127.0.0.1 --port 8000")
    add_heading(doc, "六、训练、评估与测试命令", 1)
    commands = [
        "python scripts/data/plan_dataset_v7.py",
        "python scripts/training/train_trajectory.py --help",
        "python scripts/evaluation/final_validation_analysis.py --help",
        "python scripts/external_validation/evaluate_ltafdb_external.py --help",
        "python tests/test_monitoring_policy.py",
        "python tests/test_ensemble_serving.py",
        "python tests/test_demo_replay_consistency.py",
    ]
    for cmd in commands:
        add_paragraph(doc, cmd)
    add_heading(doc, "七、常见错误", 1)
    add_table(doc, 3, "故障处理", ["问题", "处理"], [
        ["模型缺失", "检查 models/official_ensemble/seed*/arrhythmia_warning_best.pth"],
        ["API异常", "查看HTTP 422/500 detail，不沿用旧预测"],
        ["数据路径", "使用 HEALTHMONITOR_DATA_DIR 覆盖公开数据库位置"],
        ["低质量信号", "返回明确错误，提示检查电极或采集设备"],
    ])
    add_heading(doc, "八、指标映射与维护规则", 1)
    add_paragraph(doc, "报告和前端中的关键数字均来自 artifacts/metric_registry.json。修改证据文件后必须重新运行 build_metric_registry.py，并重新跑三项验收测试。", first_line=True)
    doc.save(MANUAL_DOCX)


def convert_with_soffice(docx_path):
    out_dir = docx_path.parent / "rendered"
    out_dir.mkdir(exist_ok=True)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)], check=False)
    return docx_path.with_suffix(".pdf")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    make_figures(registry)
    build_report(registry)
    build_manual(registry)
    print(REPORT_DOCX)
    print(MANUAL_DOCX)


if __name__ == "__main__":
    main()
