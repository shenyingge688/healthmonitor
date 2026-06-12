"""
Run frozen Round5 ensemble inference on a frozen LTAFDB cohort.

The external protocol is written and hashed before inference. No threshold,
lead, record, or model selection is performed from external model outputs.
Long recordings use cached 30-second window encodings; a direct-forward
equivalence check guards the optimized path.
"""
import argparse
import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wfdb
from scipy import signal
from scipy.spatial import cKDTree

from build_dataset_factory import (
    HISTORY_SEC,
    N_WINDOWS,
    OVERLAP_STRIDE_SEC,
    PREDICT_SEC,
    PTS_PER_WIN,
    TARGET_FS,
    WINDOW_SEC,
    _poincare_features,
    _sample_entropy,
    clean_ecg_signal,
    extract_rr_features,
)
from healthmonitor.dl_model import ArrhythmiaWarningNet
from prepare_ltafdb_external import (
    AF_MERGE_GAP_SEC,
    AF_MIN_EPISODE_SEC,
    normalize_rhythm_note,
)


CLASS_KEYS = ("normal", "pvc", "afib", "vf", "vt", "at_svt")
EVALUATION_STRIDE_SEC = 10
QUALITY_STD_MIN_MV = 1e-4
RR_LEGACY_EQUIVALENCE_ATOL = 5e-7
PROTOCOL_VERSION = "ltafdb-external-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_PROTOCOL_KEYS = (
    "database",
    "database_version",
    "checkpoints",
    "input",
    "preprocessing",
    "labeling",
    "frozen_policies",
    "numerical_execution",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data):
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def checkpoint_specs():
    base = Path(__file__).resolve().parent / "models_v7_stage1" / "seeds"
    return [
        {
            "seed": seed,
            "path": str(
                (
                    base
                    / f"seed{seed}"
                    / "arrhythmia_warning_best.pth"
                ).resolve()
            ),
        }
        for seed in (0, 1, 4)
    ]


def build_protocol(
    manifest_path,
    selected_leads_path,
    decision_path,
    window_batch_size,
    trajectory_batch_size,
    rr_workers,
):
    specs = checkpoint_specs()
    for spec in specs:
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        spec["sha256"] = sha256_file(path)
    manifest = pd.read_csv(manifest_path, dtype={"record_id": str})
    leads = pd.read_csv(selected_leads_path, dtype={"record_id": str})
    quality_pass = leads["record_quality_pass"].astype(bool)
    source_exception = (
        leads["source_integrity_exception"].astype(bool)
        if "source_integrity_exception" in leads
        else pd.Series(False, index=leads.index)
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_date": "2026-06-11",
        "database": "ltafdb",
        "database_version": "1.0.0",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_leads_path": str(Path(selected_leads_path).resolve()),
        "selected_leads_sha256": sha256_file(selected_leads_path),
        "validation_policy_decision_path": str(Path(decision_path).resolve()),
        "validation_policy_decision_sha256": sha256_file(decision_path),
        "checkpoints": specs,
        "record_selection_uses_model_performance": False,
        "lead_selection_uses_model_performance": False,
        "external_data_used_for_training": False,
        "evaluation_cohort": {
            "record_count": int(len(manifest)),
            "fixed_quality_pass_records": int(quality_pass.sum()),
            "source_integrity_exception_records": int(source_exception.sum()),
            "primary_analysis": "fixed signal-quality pass records only",
            "sensitivity_analysis": "all frozen evaluation-eligible records",
        },
        "input": {
            "history_sec": HISTORY_SEC,
            "prediction_horizon_sec": PREDICT_SEC,
            "window_sec": WINDOW_SEC,
            "window_overlap_stride_sec": OVERLAP_STRIDE_SEC,
            "trajectory_window_count": N_WINDOWS,
            "evaluation_stride_sec": EVALUATION_STRIDE_SEC,
            "source_lead_rule": (
                "lead 0 if fixed signal-quality gates pass; lead 1 only as "
                "a quality fallback"
            ),
        },
        "preprocessing": {
            "source_physical_units": "mV",
            "bandpass_hz": [0.5, 45.0],
            "butterworth_order": 4,
            "resampling": "scipy.signal.resample_poly",
            "target_fs_hz": TARGET_FS,
            "per_window_normalization": "z-score with denominator std + 1e-8",
            "rr_features": "same definitions as build_dataset_factory.py",
        },
        "labeling": {
            "af_gap_merge_sec": AF_MERGE_GAP_SEC,
            "minimum_af_episode_sec": AF_MIN_EPISODE_SEC,
            "current_window_sec": 30,
            "future_distribution_window_sec": PREDICT_SEC,
            "class_order": list(CLASS_KEYS),
        },
        "frozen_policies": {
            "baseline": "future six-class argmax",
            "afib_directional_score": (
                "P(AFib) / max(P(AFib) + P(Normal), 1e-8)"
            ),
            "threshold": 0.10,
            "smoothing": "causal EWMA",
            "smoothing_alpha": 0.65,
            "consecutive_k": 1,
        },
        "signal_quality_analysis": {
            "primary": "all inference time points",
            "sensitivity": (
                "histories where all 39 input windows have finite values and "
                f"standard deviation >= {QUALITY_STD_MIN_MV} mV"
            ),
        },
        "numerical_execution": {
            "window_batch_size": int(window_batch_size),
            "trajectory_batch_size": int(trajectory_batch_size),
            "rr_trajectory_workers": int(rr_workers),
            "cached_direct_probability_tolerance": 1e-3,
            "cached_direct_afib_score_tolerance": 1e-3,
            "cached_direct_argmax_match_required": True,
        },
        "external_threshold_tuning_allowed": False,
        "external_model_selection_allowed": False,
    }


def freeze_protocol(
    output_dir,
    manifest_path,
    selected_leads_path,
    decision_path,
    window_batch_size,
    trajectory_batch_size,
    rr_workers,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "external_evaluation_protocol.json"
    expected = canonical_json(
        build_protocol(
            manifest_path,
            selected_leads_path,
            decision_path,
            window_batch_size,
            trajectory_batch_size,
            rr_workers,
        )
    )
    if protocol_path.exists():
        current = protocol_path.read_text(encoding="utf-8")
        if current != expected:
            raise RuntimeError(
                "existing external protocol differs from current frozen inputs"
            )
    else:
        protocol_path.write_text(expected, encoding="utf-8")
    return protocol_path, sha256_file(protocol_path)


def validate_reuse_protocol(current, source):
    mismatches = [
        key
        for key in INFERENCE_PROTOCOL_KEYS
        if current.get(key) != source.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "reuse inference protocol differs in numerical fields: "
            f"{mismatches}"
        )


def reuse_record_inference(
    reuse_dir,
    record,
    selected_lead_index,
    record_quality_pass,
    source_integrity_exception,
    analysis_role,
):
    csv_path = Path(reuse_dir) / "per_record" / f"{record}.csv"
    metadata_path = Path(reuse_dir) / "per_record" / f"{record}.json"
    if not csv_path.exists() or not metadata_path.exists():
        return None

    frame = pd.read_csv(csv_path, dtype={"record_id": str})
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = payload.get("record_metadata", payload)
    required = {
        "patient_id",
        "record_id",
        "window_id",
        "timestamp_sec",
        "selected_lead_index",
        "prob_fut_normal",
        "prob_fut_afib",
        "afib_vs_normal",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"reuse record {record} is missing columns: {missing}")
    if len(frame) != int(metadata["rows"]):
        raise RuntimeError(f"reuse record {record} row count mismatch")
    if frame.duplicated(["patient_id", "window_id"]).any():
        raise RuntimeError(f"reuse record {record} contains duplicate rows")
    if len(frame) > 1 and not np.allclose(
        frame["timestamp_sec"].diff().dropna(),
        EVALUATION_STRIDE_SEC,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(f"reuse record {record} has an invalid time axis")
    source_leads = frame["selected_lead_index"].dropna().astype(int).unique()
    if (
        len(source_leads) != 1
        or int(source_leads[0]) != int(selected_lead_index)
    ):
        raise RuntimeError(f"reuse record {record} selected lead mismatch")

    frame["record_quality_pass"] = bool(record_quality_pass)
    frame["source_integrity_exception"] = bool(source_integrity_exception)
    frame["analysis_role"] = str(analysis_role)
    metadata = {
        **metadata,
        "record_quality_pass": bool(record_quality_pass),
        "source_integrity_exception": bool(source_integrity_exception),
        "analysis_role": str(analysis_role),
        "reuse_source_csv": str(csv_path.resolve()),
        "reuse_source_csv_sha256": sha256_file(csv_path),
    }
    return frame, metadata


def rhythm_markers(annotation, sig_len):
    markers = []
    for sample, aux_note in zip(annotation.sample, annotation.aux_note):
        rhythm = normalize_rhythm_note(aux_note)
        if rhythm is not None:
            markers.append((min(max(int(sample), 0), sig_len), rhythm))
    markers.sort(key=lambda item: item[0])
    deduplicated = []
    for sample, rhythm in markers:
        if deduplicated and deduplicated[-1][0] == sample:
            deduplicated[-1] = (sample, rhythm)
        else:
            deduplicated.append((sample, rhythm))
    return deduplicated


def cleaned_af_episodes(markers, sig_len, fs):
    raw = []
    for index, (start, rhythm) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else sig_len
        if rhythm == "AFIB" and end > start:
            raw.append([start, end])
    merge_gap = int(round(AF_MERGE_GAP_SEC * fs))
    merged = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    minimum = int(round(AF_MIN_EPISODE_SEC * fs))
    return [(start, end) for start, end in merged if end - start >= minimum]


def rhythm_class(rhythm):
    note = str(rhythm).upper()
    if "VFIB" in note or "VFL" in note:
        return 3
    if "VT" in note:
        return 4
    if "AFIB" in note:
        return 0
    if "AFL" in note or "SVT" in note or "AT" in note:
        return 5
    return 0


def build_labels(annotation, source_len, source_fs, target_len):
    markers = rhythm_markers(annotation, source_len)
    scale = TARGET_FS / float(source_fs)
    labels = np.zeros(target_len, dtype=np.int8)

    for index, (source_start, rhythm) in enumerate(markers):
        source_end = (
            markers[index + 1][0] if index + 1 < len(markers) else source_len
        )
        class_index = rhythm_class(rhythm)
        if class_index:
            start = min(int(source_start * scale), target_len)
            end = min(int(source_end * scale), target_len)
            labels[start:end] = class_index

    for source_sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in ("V", "E"):
            continue
        sample = min(int(int(source_sample) * scale), target_len)
        start = max(0, sample - 1125)
        end = min(target_len, sample + 1125)
        region = labels[start:end]
        region[region == 0] = 1

    episodes = cleaned_af_episodes(markers, source_len, source_fs)
    target_episodes = []
    for source_start, source_end in episodes:
        start = min(int(source_start * scale), target_len)
        end = min(int(source_end * scale), target_len)
        labels[start:end] = 2
        target_episodes.append((start, end))
    return labels, target_episodes


def preprocess_record(record_path, lead_index):
    header = wfdb.rdheader(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr", pn_dir=None)
    if str(header.fmt[lead_index]) != "16":
        raise ValueError(f"{record_path.name}: unsupported format")
    dat_path = record_path.with_suffix(".dat")
    raw = np.memmap(dat_path, dtype="<i2", mode="r").reshape(
        int(header.sig_len),
        int(header.n_sig),
    )
    digital = np.asarray(raw[:, int(lead_index)], dtype=np.float64)
    physical = (
        digital - float(header.baseline[lead_index])
    ) / float(header.adc_gain[lead_index])
    filtered = clean_ecg_signal(physical, fs=float(header.fs))
    if int(round(float(header.fs))) == TARGET_FS:
        ecg = filtered.astype(np.float32)
    else:
        ecg = signal.resample_poly(
            filtered,
            TARGET_FS,
            int(round(float(header.fs))),
        ).astype(np.float32)
    if not np.isfinite(ecg).all():
        raise ValueError(f"{record_path.name}: non-finite resampled ECG")
    labels, af_episodes = build_labels(
        annotation,
        int(header.sig_len),
        float(header.fs),
        len(ecg),
    )
    del raw, digital, physical, filtered
    return ecg, labels, af_episodes, header


def fast_sample_entropy(rr, m=2, r_factor=0.2):
    rr = np.asarray(rr, dtype=np.float64)
    n_values = len(rr)
    if n_values < m + 2:
        return 0.0
    radius = r_factor * np.std(rr)
    if radius < 1e-10:
        return 0.0

    def phi(template_len):
        templates = np.lib.stride_tricks.sliding_window_view(
            rr,
            template_len,
        )[:-1]
        if len(templates) < 2:
            return 0
        strict_radius = np.nextafter(radius, -np.inf)
        pairs = cKDTree(templates).query_pairs(
            strict_radius,
            p=np.inf,
            output_type="ndarray",
        )
        return int(2 * len(pairs))

    count_a = phi(m + 1)
    count_b = phi(m)
    return (
        float(-np.log(count_a / count_b) / 2.0)
        if count_b >= 1 and count_a > 0
        else 0.0
    )


def fast_trajectory_features(all_rr):
    if len(all_rr) < 10:
        return np.zeros(5, dtype=np.float32)
    rr = np.asarray(all_rr, dtype=np.float64)
    sd1, sd2, sd_ratio = _poincare_features(rr)
    cv_rr = float(np.std(rr) / max(np.mean(rr), 1e-8))
    return np.array(
        [fast_sample_entropy(rr), sd1, sd2, sd_ratio, cv_rr],
        dtype=np.float32,
    )


def normalized_windows(ecg, starts, batch_indices):
    windows = []
    for index in batch_indices:
        start = int(starts[int(index)])
        raw = ecg[start : start + PTS_PER_WIN]
        normalized = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
        windows.append(normalized.astype(np.float32))
    return np.stack(windows)[:, np.newaxis, :]


def build_record_features(ecg, maximum_trajectories=None, rr_workers=1):
    stride = EVALUATION_STRIDE_SEC * TARGET_FS
    maximum_start = len(ecg) - (HISTORY_SEC + PREDICT_SEC) * TARGET_FS
    if maximum_start <= 0:
        raise ValueError("record is too short for external trajectories")
    trajectory_starts = np.arange(0, maximum_start, stride, dtype=np.int64)
    if maximum_trajectories is not None:
        trajectory_starts = trajectory_starts[: int(maximum_trajectories)]
    offsets = (
        np.arange(N_WINDOWS, dtype=np.int64)
        * OVERLAP_STRIDE_SEC
        * TARGET_FS
    )
    all_starts = trajectory_starts[:, None] + offsets[None, :]
    unique_starts, inverse = np.unique(all_starts, return_inverse=True)
    trajectory_indices = inverse.reshape(len(trajectory_starts), N_WINDOWS)

    feature4 = np.zeros((len(unique_starts), 4), dtype=np.float32)
    window_std = np.zeros(len(unique_starts), dtype=np.float32)
    intervals = []
    for index, start in enumerate(unique_starts):
        raw = ecg[int(start) : int(start) + PTS_PER_WIN]
        window_std[index] = float(np.std(raw))
        features, rr = extract_rr_features(raw, fs=TARGET_FS)
        feature4[index] = features
        intervals.append(rr)

    def trajectory_features(window_indices):
        available = [
            intervals[int(window_index)]
            for window_index in window_indices
            if len(intervals[int(window_index)]) > 0
        ]
        concatenated = (
            np.concatenate(available)
            if available
            else np.array([], dtype=np.float32)
        )
        return fast_trajectory_features(concatenated)

    workers = max(int(rr_workers), 1)
    if workers == 1:
        trajectory_rows = list(map(trajectory_features, trajectory_indices))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            trajectory_rows = list(
                executor.map(trajectory_features, trajectory_indices)
            )
    rr_last = np.zeros((len(trajectory_starts), 9), dtype=np.float32)
    rr_last[:, :4] = feature4[trajectory_indices[:, -1]]
    rr_last[:, 4:] = np.stack(trajectory_rows)

    for index, window_indices in enumerate(trajectory_indices[:3]):
        available = [
            intervals[int(window_index)]
            for window_index in window_indices
            if len(intervals[int(window_index)]) > 0
        ]
        concatenated = (
            np.concatenate(available)
            if available
            else np.array([], dtype=np.float32)
        )
        trajectory = rr_last[index, 4:]
        if index < 3 and len(concatenated) > 0:
            legacy = np.array(
                [
                    _sample_entropy(concatenated),
                    *_poincare_features(concatenated),
                    float(
                        np.std(concatenated)
                        / max(np.mean(concatenated), 1e-8)
                    ),
                ],
                dtype=np.float32,
            )
            if not np.allclose(
                legacy,
                trajectory,
                rtol=0.0,
                atol=RR_LEGACY_EQUIVALENCE_ATOL,
            ):
                raise RuntimeError(
                    "optimized RR trajectory features differ from legacy "
                    f"implementation: {legacy} vs {trajectory}"
                )

    history_quality_valid = (
        np.isfinite(window_std[trajectory_indices]).all(axis=1)
        & (window_std[trajectory_indices] >= QUALITY_STD_MIN_MV).all(axis=1)
    )
    return {
        "trajectory_starts": trajectory_starts,
        "unique_starts": unique_starts,
        "trajectory_indices": trajectory_indices,
        "rr_last": rr_last,
        "window_std": window_std,
        "history_quality_valid": history_quality_valid,
    }


@torch.inference_mode()
def encode_unique_windows(model, ecg, unique_starts, batch_size):
    cache_dtype = np.float16 if DEVICE.type == "cuda" else np.float32
    embeddings = np.empty((len(unique_starts), 256), dtype=cache_dtype)
    envelopes = np.empty((len(unique_starts), 256), dtype=cache_dtype)
    amp_enabled = DEVICE.type == "cuda"
    for start in range(0, len(unique_starts), batch_size):
        end = min(start + batch_size, len(unique_starts))
        batch = normalized_windows(
            ecg,
            unique_starts,
            np.arange(start, end),
        )
        tensor = torch.from_numpy(batch).to(DEVICE)
        with torch.amp.autocast(DEVICE.type, enabled=amp_enabled):
            encoded = model.window_encoder(tensor)
            envelope = model.env_pool(torch.abs(tensor))
            envelope = model.env_encoder(envelope.view(len(tensor), -1))
        embeddings[start:end] = encoded.cpu().numpy()
        envelopes[start:end] = envelope.cpu().numpy()
    return embeddings, envelopes


def cached_forward(model, z_sequence, env_sequence, rr_last):
    temporal = z_sequence.transpose(1, 2)
    global_skip = 0
    for index, block in enumerate(model.tcn_blocks):
        temporal, skip = block(temporal)
        if index < len(model.skip_adapters):
            global_skip = global_skip + model.skip_adapters[index](skip)
        elif skip.size(1) == model.skip_adapters[-1].out_channels:
            global_skip = global_skip + skip
        else:
            global_skip = global_skip + model.skip_adapters[-1](skip)
    attention = model.temporal_attn(global_skip)
    time_feature = (global_skip * attention).sum(dim=2) / (
        attention.sum(dim=2) + 1e-4
    )
    rr_feature = model.rr_encoder(rr_last)
    env_feature = env_sequence.mean(dim=1)
    fused = model.fusion(
        torch.cat([time_feature, rr_feature, env_feature], dim=-1)
    )
    return model.current_head(fused), model.future_head(fused)


@torch.inference_mode()
def run_cached_model(
    model,
    ecg,
    features,
    window_batch_size,
    trajectory_batch_size,
):
    embeddings, envelopes = encode_unique_windows(
        model,
        ecg,
        features["unique_starts"],
        window_batch_size,
    )
    indices = features["trajectory_indices"]
    rr_last = features["rr_last"]
    logits_cur = np.empty((len(indices), 6), dtype=np.float32)
    logits_fut = np.empty((len(indices), 6), dtype=np.float32)
    amp_enabled = DEVICE.type == "cuda"

    for start in range(0, len(indices), trajectory_batch_size):
        end = min(start + trajectory_batch_size, len(indices))
        selected = indices[start:end]
        z = torch.from_numpy(embeddings[selected]).to(DEVICE)
        env = torch.from_numpy(envelopes[selected]).to(DEVICE)
        rr = torch.from_numpy(rr_last[start:end]).to(DEVICE)
        with torch.amp.autocast(DEVICE.type, enabled=amp_enabled):
            current, future = cached_forward(model, z, env, rr)
        logits_cur[start:end] = current.float().cpu().numpy()
        logits_fut[start:end] = future.float().cpu().numpy()

    check_count = min(2, len(indices))
    check_indices = indices[:check_count]
    direct_windows = np.stack(
        [
            normalized_windows(
                ecg,
                features["unique_starts"],
                row,
            )
            for row in check_indices
        ]
    )
    direct_rr = np.zeros((check_count, N_WINDOWS, 9), dtype=np.float32)
    direct_rr[:, -1, :] = rr_last[:check_count]
    with torch.amp.autocast(DEVICE.type, enabled=amp_enabled):
        direct = model(
            torch.from_numpy(direct_windows).to(DEVICE),
            x_rr=torch.from_numpy(direct_rr).to(DEVICE),
        )
    direct_current_logits = direct["logits_cur"].float().cpu().numpy()
    direct_future_logits = direct["logits_fut"].float().cpu().numpy()
    current_error = float(
        np.max(
            np.abs(
                direct_current_logits - logits_cur[:check_count]
            )
        )
    )
    future_error = float(
        np.max(
            np.abs(
                direct_future_logits - logits_fut[:check_count]
            )
        )
    )
    direct_current_prob = softmax_numpy(direct_current_logits)
    direct_future_prob = softmax_numpy(direct_future_logits)
    cached_current_prob = softmax_numpy(logits_cur[:check_count])
    cached_future_prob = softmax_numpy(logits_fut[:check_count])
    current_probability_error = float(
        np.max(np.abs(direct_current_prob - cached_current_prob))
    )
    future_probability_error = float(
        np.max(np.abs(direct_future_prob - cached_future_prob))
    )
    direct_afib_score = direct_future_prob[:, 2] / np.maximum(
        direct_future_prob[:, 2] + direct_future_prob[:, 0],
        1e-8,
    )
    cached_afib_score = cached_future_prob[:, 2] / np.maximum(
        cached_future_prob[:, 2] + cached_future_prob[:, 0],
        1e-8,
    )
    afib_score_error = float(
        np.max(np.abs(direct_afib_score - cached_afib_score))
    )
    current_argmax_match = bool(
        np.array_equal(
            direct_current_prob.argmax(axis=1),
            cached_current_prob.argmax(axis=1),
        )
    )
    future_argmax_match = bool(
        np.array_equal(
            direct_future_prob.argmax(axis=1),
            cached_future_prob.argmax(axis=1),
        )
    )
    if (
        max(current_error, future_error) > 1e-2
        or max(current_probability_error, future_probability_error) > 1e-3
        or afib_score_error > 1e-3
        or not current_argmax_match
        or not future_argmax_match
    ):
        raise RuntimeError(
            "cached inference differs from direct forward: "
            f"current_logit={current_error}, future_logit={future_error}, "
            f"current_probability={current_probability_error}, "
            f"future_probability={future_probability_error}, "
            f"afib_score={afib_score_error}, "
            f"current_argmax_match={current_argmax_match}, "
            f"future_argmax_match={future_argmax_match}"
        )
    return logits_cur, logits_fut, {
        "current_max_abs_error": current_error,
        "future_max_abs_error": future_error,
        "current_probability_max_abs_error": current_probability_error,
        "future_probability_max_abs_error": future_probability_error,
        "afib_score_max_abs_error": afib_score_error,
        "current_argmax_match": current_argmax_match,
        "future_argmax_match": future_argmax_match,
    }


def softmax_numpy(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def target_table(labels, trajectory_starts):
    history_end = trajectory_starts + HISTORY_SEC * TARGET_FS
    current_start = history_end - 30 * TARGET_FS
    future_end = history_end + PREDICT_SEC * TARGET_FS
    current_counts = np.empty((len(trajectory_starts), 6), dtype=np.int32)
    future_counts = np.empty((len(trajectory_starts), 6), dtype=np.int32)
    for class_index in range(6):
        prefix = np.empty(len(labels) + 1, dtype=np.int32)
        prefix[0] = 0
        np.cumsum(
            labels == class_index,
            dtype=np.int32,
            out=prefix[1:],
        )
        current_counts[:, class_index] = (
            prefix[history_end] - prefix[current_start]
        )
        future_counts[:, class_index] = (
            prefix[future_end] - prefix[history_end]
        )
    priority = (3, 4, 2, 5, 1, 0)
    target_cur = np.zeros(len(trajectory_starts), dtype=np.int8)
    unassigned = np.ones(len(trajectory_starts), dtype=bool)
    for class_index in priority:
        present = current_counts[:, class_index] > 0
        target_cur[unassigned & present] = class_index
        unassigned &= ~present

    future_soft = future_counts.astype(np.float32) / (
        PREDICT_SEC * TARGET_FS
    )
    return target_cur, future_soft.argmax(axis=1), future_soft


def load_model(checkpoint_path):
    model = ArrhythmiaWarningNet().to(DEVICE)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    state = checkpoint.get("ema", checkpoint.get("model", {}))
    if not state:
        raise RuntimeError(f"checkpoint has no model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def infer_record(
    record,
    lead_index,
    records_dir,
    protocol,
    maximum_trajectories,
    window_batch_size,
    trajectory_batch_size,
    rr_workers=1,
    record_quality_pass=True,
    source_integrity_exception=False,
    analysis_role="primary",
):
    record_path = Path(records_dir) / record
    ecg, labels, af_episodes, header = preprocess_record(
        record_path,
        lead_index,
    )
    features = build_record_features(
        ecg,
        maximum_trajectories,
        rr_workers=rr_workers,
    )
    target_cur, target_fut, future_soft = target_table(
        labels,
        features["trajectory_starts"],
    )

    member_current = []
    member_future = []
    equivalence = []
    for spec in protocol["checkpoints"]:
        print(f"[model] record={record} seed={spec['seed']}")
        model = load_model(spec["path"])
        logits_cur, logits_fut, check = run_cached_model(
            model,
            ecg,
            features,
            window_batch_size,
            trajectory_batch_size,
        )
        member_current.append(logits_cur)
        member_future.append(logits_fut)
        equivalence.append({"seed": spec["seed"], **check})
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    member_current = np.stack(member_current, axis=1)
    member_future = np.stack(member_future, axis=1)
    logits_cur = member_current.mean(axis=1)
    logits_fut = member_future.mean(axis=1)
    probs_cur = softmax_numpy(logits_cur)
    probs_fut = softmax_numpy(logits_fut)
    member_probs_fut = np.stack(
        [softmax_numpy(member_future[:, index, :]) for index in range(3)],
        axis=1,
    )

    starts = features["trajectory_starts"]
    frame = pd.DataFrame(
        {
            "patient_id": f"ltafdb:{record}",
            "db": "ltafdb",
            "record_id": record,
            "window_id": np.arange(len(starts), dtype=np.int64),
            "history_start_sec": starts / TARGET_FS,
            "timestamp_sec": starts / TARGET_FS + HISTORY_SEC,
            "future_start_sec": starts / TARGET_FS + HISTORY_SEC,
            "future_end_sec": starts / TARGET_FS + HISTORY_SEC + PREDICT_SEC,
            "source_fs": float(header.fs),
            "selected_lead_index": int(lead_index),
            "record_quality_pass": bool(record_quality_pass),
            "source_integrity_exception": bool(source_integrity_exception),
            "analysis_role": str(analysis_role),
            "history_quality_valid": features["history_quality_valid"],
            "target_cur": target_cur.astype(int),
            "target_fut": target_fut.astype(int),
            "pred_cur": probs_cur.argmax(axis=1),
            "pred_fut": probs_fut.argmax(axis=1),
            "transition_flag": target_cur != target_fut,
        }
    )
    for class_index, key in enumerate(CLASS_KEYS):
        frame[f"target_fut_soft_{key}"] = future_soft[:, class_index]
        frame[f"prob_cur_{key}"] = probs_cur[:, class_index]
        frame[f"prob_fut_{key}"] = probs_fut[:, class_index]
        frame[f"logit_cur_{key}"] = logits_cur[:, class_index]
        frame[f"logit_fut_{key}"] = logits_fut[:, class_index]
    for member_index, spec in enumerate(protocol["checkpoints"]):
        frame[f"prob_fut_afib_seed{spec['seed']}"] = member_probs_fut[
            :,
            member_index,
            2,
        ]
    frame["afib_member_prob_std"] = member_probs_fut[:, :, 2].std(axis=1)
    frame["afib_vs_normal"] = probs_fut[:, 2] / np.maximum(
        probs_fut[:, 2] + probs_fut[:, 0],
        1e-8,
    )
    metadata = {
        "record_id": record,
        "rows": int(len(frame)),
        "source_duration_hours": float(header.sig_len / header.fs / 3600.0),
        "cleaned_af_episode_count": int(len(af_episodes)),
        "quality_valid_rows": int(frame["history_quality_valid"].sum()),
        "record_quality_pass": bool(record_quality_pass),
        "source_integrity_exception": bool(source_integrity_exception),
        "analysis_role": str(analysis_role),
        "cached_direct_equivalence": equivalence,
        "maximum_trajectories": maximum_trajectories,
        "rr_workers": int(rr_workers),
    }
    del ecg, labels, features
    gc.collect()
    return frame, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_pilot_audit\pilot_manifest.csv"
        ),
    )
    parser.add_argument(
        "--selected-leads",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_signal_audit\selected_leads.csv"
        ),
    )
    parser.add_argument(
        "--decision",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\afib_baseline\decision.json"
        ),
    )
    parser.add_argument(
        "--records-dir",
        default=r"C:\HealthMonitor\data\ltafdb_pilot\records",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_external_inference"
        ),
    )
    parser.add_argument("--records", default=None)
    parser.add_argument("--maximum-trajectories", type=int, default=None)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--trajectory-batch-size", type=int, default=16)
    parser.add_argument("--rr-workers", type=int, default=8)
    parser.add_argument(
        "--reuse-inference-dir",
        default=None,
        help=(
            "Optional completed inference directory. Records are reused only "
            "when all numerical protocol fields and per-record invariants match."
        ),
    )
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    protocol_path, protocol_hash = freeze_protocol(
        output_dir,
        args.manifest,
        args.selected_leads,
        args.decision,
        args.window_batch_size,
        args.trajectory_batch_size,
        args.rr_workers,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    print(f"[protocol] sha256={protocol_hash}")
    if args.reuse_inference_dir:
        reuse_protocol_path = (
            Path(args.reuse_inference_dir) / "external_evaluation_protocol.json"
        )
        if not reuse_protocol_path.exists():
            raise FileNotFoundError(reuse_protocol_path)
        reuse_protocol = json.loads(
            reuse_protocol_path.read_text(encoding="utf-8")
        )
        validate_reuse_protocol(protocol, reuse_protocol)
        print(f"[reuse] protocol compatible: {args.reuse_inference_dir}")
    if args.protocol_only:
        return

    manifest = pd.read_csv(args.manifest, dtype={"record_id": str})
    manifest["record_id"] = manifest["record_id"].str.zfill(2)
    leads = pd.read_csv(args.selected_leads, dtype={"record_id": str})
    leads["record_id"] = leads["record_id"].str.zfill(2)
    optional_lead_columns = [
        column
        for column in (
            "evaluation_eligible",
            "source_integrity_exception",
            "analysis_role",
        )
        if column in leads.columns
    ]
    manifest = manifest.merge(
        leads[
            [
                "record_id",
                "selected_lead_index",
                "record_quality_pass",
                *optional_lead_columns,
            ]
        ],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    eligibility = (
        manifest["evaluation_eligible"].astype(bool)
        if "evaluation_eligible" in manifest
        else manifest["record_quality_pass"].astype(bool)
    )
    if not eligibility.all():
        failed = manifest.loc[~eligibility, "record_id"].tolist()
        raise RuntimeError(
            f"frozen manifest contains ineligible signal records: {failed}"
        )
    if "source_integrity_exception" not in manifest:
        manifest["source_integrity_exception"] = False
    if "analysis_role" not in manifest:
        manifest["analysis_role"] = "primary"
    if args.records:
        requested = [
            item.strip().zfill(2)
            for item in args.records.split(",")
            if item.strip()
        ]
        unknown = sorted(set(requested) - set(manifest["record_id"]))
        if unknown:
            raise ValueError(f"records not in frozen manifest: {unknown}")
        manifest = manifest.set_index("record_id").loc[requested].reset_index()

    per_record_dir = output_dir / "per_record"
    per_record_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    record_metadata = []
    run_mode = {
        "protocol_sha256": protocol_hash,
        "maximum_trajectories": args.maximum_trajectories,
    }
    for index, row in enumerate(manifest.itertuples(), start=1):
        record = str(row.record_id).zfill(2)
        csv_path = per_record_dir / f"{record}.csv"
        metadata_path = per_record_dir / f"{record}.json"
        use_cache = False
        if csv_path.exists() and metadata_path.exists() and not args.overwrite:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            use_cache = cached.get("run_mode") == run_mode
        if use_cache:
            frame = pd.read_csv(csv_path, dtype={"record_id": str})
            metadata = cached["record_metadata"]
            print(f"[record] {index}/{len(manifest)} {record} cached")
        else:
            reused = (
                reuse_record_inference(
                    args.reuse_inference_dir,
                    record,
                    int(row.selected_lead_index),
                    bool(row.record_quality_pass),
                    bool(row.source_integrity_exception),
                    str(row.analysis_role),
                )
                if args.reuse_inference_dir
                and args.maximum_trajectories is None
                else None
            )
            if reused is not None:
                frame, metadata = reused
                print(f"[record] {index}/{len(manifest)} {record} reused")
            else:
                frame, metadata = infer_record(
                    record,
                    int(row.selected_lead_index),
                    args.records_dir,
                    protocol,
                    args.maximum_trajectories,
                    args.window_batch_size,
                    args.trajectory_batch_size,
                    args.rr_workers,
                    bool(row.record_quality_pass),
                    bool(row.source_integrity_exception),
                    str(row.analysis_role),
                )
            frame.to_csv(csv_path, index=False)
            metadata_path.write_text(
                canonical_json(
                    {
                        "run_mode": run_mode,
                        "record_metadata": metadata,
                    }
                ),
                encoding="utf-8",
            )
            print(f"[record] {index}/{len(manifest)} {record} complete")
        frames.append(frame)
        record_metadata.append(metadata)

    outputs = pd.concat(frames, ignore_index=True)
    outputs.to_csv(output_dir / "external_inference_outputs.csv", index=False)
    summary = {
        "protocol_sha256": protocol_hash,
        "record_count": int(len(manifest)),
        "primary_record_count": int(manifest["record_quality_pass"].sum()),
        "source_integrity_exception_record_count": int(
            manifest["source_integrity_exception"].sum()
        ),
        "row_count": int(len(outputs)),
        "quality_valid_rows": int(outputs["history_quality_valid"].sum()),
        "model_evaluation_status": "inference_complete_metrics_not_run",
        "external_threshold_tuning_performed": False,
        "records": record_metadata,
    }
    (output_dir / "inference_summary.json").write_text(
        canonical_json(summary),
        encoding="utf-8",
    )
    print(
        f"Saved external inference: records={len(manifest)}, "
        f"rows={len(outputs)}, quality_valid="
        f"{outputs['history_quality_valid'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
