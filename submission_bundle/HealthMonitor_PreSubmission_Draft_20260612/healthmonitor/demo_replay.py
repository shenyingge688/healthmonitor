"""Shared case definitions and ECG loading for frozen competition replays."""
import os
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal
from scipy.signal import butter, filtfilt

from .constants import TARGET_FS
from .paths import DATA_DIR, DEMO_DIR


DEMO_CASES = {
    "100": {
        "record_id": "100",
        "db": "mitdb",
        "split": "demo_holdout",
        "tier": "formal_success",
        "title": "正常基线与误报抑制",
        "short_title": "MIT-BIH 100 · 正常基线",
        "start_min": 10.0,
        "end_min": 13.0,
        "target_class": 0,
        "claim": "所选冻结片段用于展示正常基线和正式报警的误报抑制。",
        "boundary": "仅代表该冻结片段，不外推为所有正常心电均不报警。",
    },
    "119": {
        "record_id": "119",
        "db": "mitdb",
        "split": "demo_holdout",
        "tier": "formal_success",
        "title": "PVC稳定识别与模型关注窗口",
        "short_title": "MIT-BIH 119 · PVC",
        "start_min": 10.0,
        "end_min": 13.0,
        "target_class": 1,
        "claim": "所选冻结片段用于展示PVC输出及模型关注窗口与室性标注的对应关系。",
        "boundary": "类别输出是模型参考，不替代临床节律判读。",
    },
    "201": {
        "record_id": "201",
        "db": "mitdb",
        "split": "demo_holdout",
        "tier": "pilot_success",
        "title": "AFib辅助方向提示",
        "short_title": "MIT-BIH 201 · AFib辅助方向",
        "start_min": 26.5,
        "end_min": 29.5,
        "target_class": 2,
        "claim": "26.5至29.5分钟覆盖AFib标注起始后的片段，用于展示AFib辅助方向分数。",
        "boundary": "LTAFDB 83记录主分析、84记录敏感性分析的外部研究支持；尚未接入正式报警。",
    },
    "223": {
        "record_id": "223",
        "db": "mitdb",
        "split": "demo_holdout",
        "tier": "boundary",
        "title": "短阵VT未稳定检出",
        "short_title": "MIT-BIH 223 · VT能力边界",
        "start_min": 9.3,
        "end_min": 10.8,
        "target_class": 4,
        "claim": "标注在约9.64分钟出现短阵VT，当前模型未形成稳定VT输出。",
        "boundary": "该病例只用于展示能力边界，不能作为VT检出成功案例。",
    },
    "209": {
        "record_id": "209",
        "db": "mitdb",
        "split": "demo_holdout",
        "tier": "boundary",
        "title": "AT/SVT易与VT混淆",
        "short_title": "MIT-BIH 209 · AT/SVT能力边界",
        "start_min": 9.0,
        "end_min": 14.8,
        "target_class": 5,
        "claim": "所选片段含多次SVTA标注，当前模型常将其归入其他快速心律类别。",
        "boundary": "AT/SVT数据与验证证据不足，只展示局限，不形成有效识别主张。",
    },
}


def clean_ecg_signal(data, fs):
    nyquist = 0.5 * fs
    b, a = butter(4, [0.5 / nyquist, 45.0 / nyquist], btype="band")
    return filtfilt(b, a, data)


def load_case_signal(case, base_dir):
    frozen_path = DEMO_DIR / "signals" / f'{case["db"]}_{case["record_id"]}_signal.npz'
    if frozen_path.exists():
        with np.load(frozen_path) as data:
            source_fs = (
                float(data["source_fs"][0])
                if "source_fs" in data.files
                else float(TARGET_FS)
            )
            return (
                data["signal"].astype(np.float32),
                source_fs,
                str(frozen_path),
            )
    record_path = os.path.join(
        str(Path(os.environ.get("HEALTHMONITOR_DATA_DIR", DATA_DIR))),
        case["db"],
        case["record_id"],
    )
    record = wfdb.rdrecord(record_path, sampto=int(30 * 60 * 360))
    src_fs = float(getattr(record, "fs", 360) or 360)
    raw = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    cleaned = clean_ecg_signal(raw, src_fs)
    if int(src_fs) != TARGET_FS:
        cleaned = signal.resample_poly(
            cleaned,
            TARGET_FS,
            int(src_fs),
        )
    return cleaned.astype(np.float32), src_fs, record_path


def rhythm_annotations(record_path, src_fs, start_sec, end_sec):
    annotation = wfdb.rdann(record_path, "atr")
    rows = []
    for sample, note in zip(annotation.sample, annotation.aux_note):
        timestamp_sec = float(sample) / float(src_fs)
        cleaned_note = str(note).replace("\x00", "").strip()
        if cleaned_note and start_sec <= timestamp_sec <= end_sec:
            rows.append({
                "timestamp_sec": timestamp_sec,
                "timestamp_min": timestamp_sec / 60.0,
                "note": cleaned_note,
            })
    return rows


def count_ventricular_annotations(record_path, src_fs, start_sec, end_sec):
    annotation = wfdb.rdann(record_path, "atr")
    count = 0
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        timestamp_sec = float(sample) / float(src_fs)
        if start_sec <= timestamp_sec <= end_sec and symbol in {"V", "E"}:
            count += 1
    return count
