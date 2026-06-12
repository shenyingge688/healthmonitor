"""
Generate hierarchical ensemble explanations down to the ECG sample level.

Level 1 selects an important 30-second history window using class-conditional
Grad-CAM over the 39-window temporal model. Level 2 applies integrated
gradients inside that window. For annotated PVC examples, attribution is
compared with V/E beat neighborhoods.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import wfdb
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.metrics import average_precision_score

import healthmonitor.main as serving
from healthmonitor.constants import (
    API_BUFFER_SIZE,
    CLASS_NAMES,
    HISTORY_PTS,
    N_WINDOWS,
    PTS_PER_WIN,
    STRIDE_PTS,
    TARGET_FS,
)

ENGLISH_CLASS_NAMES = ("Normal", "PVC", "AFib", "VF", "VT", "AT/SVT")


def clean_ecg(sig, fs):
    nyq = 0.5 * float(fs)
    b, a = butter(4, [0.5 / nyq, 45.0 / nyq], btype="band")
    return filtfilt(b, a, sig)


def load_record(base_dir, db, record_id, max_minutes):
    record_path = os.path.join(base_dir, "data", db, record_id)
    header = wfdb.rdheader(record_path)
    source_fs = float(header.fs)
    sample_limit = min(
        int(max_minutes * 60 * source_fs),
        int(header.sig_len),
    )
    record = wfdb.rdrecord(record_path, sampto=sample_limit)
    raw = (
        record.p_signal[:, 0]
        if record.p_signal.ndim > 1
        else record.p_signal
    )
    cleaned = clean_ecg(raw, source_fs)
    if int(source_fs) != TARGET_FS:
        ecg = signal.resample_poly(
            cleaned,
            TARGET_FS,
            int(source_fs),
        ).astype(np.float32)
    else:
        ecg = cleaned.astype(np.float32)
    annotation = wfdb.rdann(record_path, "atr")
    return ecg, annotation, source_fs


def build_model_inputs(ecg, end_point):
    start = max(0, int(end_point) - API_BUFFER_SIZE)
    buffer = ecg[start:int(end_point)]
    if len(buffer) < API_BUFFER_SIZE:
        buffer = np.pad(buffer, (API_BUFFER_SIZE - len(buffer), 0))
    windows, rr = serving.build_window_sequence(buffer)
    bx = torch.from_numpy(windows).unsqueeze(0).to(
        serving.device,
        dtype=torch.float32,
    )
    bx_rr = torch.from_numpy(rr).unsqueeze(0).to(
        serving.device,
        dtype=torch.float32,
    )
    return bx, bx_rr


def ensemble_window_gradcam(models, bx, bx_rr, target_class):
    cams = []
    for model in models:
        cam = model.grad_cam(
            bx,
            bx_rr,
            target_class=target_class,
            head="future",
        )
        cams.append(cam[0].float().cpu().numpy())
    member_cams = np.stack(cams, axis=0)
    ensemble_cam = member_cams.mean(axis=0)
    ensemble_cam /= max(float(ensemble_cam.max()), 1e-8)
    return ensemble_cam, member_cams


def selected_window_integrated_gradients(
    models,
    bx,
    bx_rr,
    target_class,
    window_index,
    steps,
):
    member_attributions = []
    alphas = np.linspace(1.0 / steps, 1.0, steps)
    for model in models:
        original_encoder_flag = model.encoder_grad_enabled
        parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.encoder_grad_enabled = True
        model.eval()

        accumulated_gradient = torch.zeros(
            PTS_PER_WIN,
            device=bx.device,
            dtype=torch.float32,
        )
        try:
            for alpha in alphas:
                interpolated = bx.detach().clone()
                interpolated[:, window_index] *= float(alpha)
                interpolated.requires_grad_(True)
                output = model(interpolated, x_rr=bx_rr)
                score = output["logits_fut"][:, target_class].sum()
                gradient = torch.autograd.grad(
                    score,
                    interpolated,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                accumulated_gradient += gradient[
                    0,
                    window_index,
                    0,
                ].float()

            average_gradient = accumulated_gradient / float(steps)
            attribution = (
                bx[0, window_index, 0].detach().float()
                * average_gradient
            )
            member_attributions.append(attribution.cpu().numpy())
        finally:
            model.encoder_grad_enabled = original_encoder_flag
            for parameter, requires_grad in zip(model.parameters(), parameter_flags):
                parameter.requires_grad_(requires_grad)

    signed = np.mean(np.stack(member_attributions, axis=0), axis=0)
    magnitude = gaussian_filter1d(
        np.abs(signed),
        sigma=max(1.0, TARGET_FS * 0.04),
    )
    magnitude /= max(float(magnitude.max()), 1e-12)
    return signed, magnitude, np.stack(member_attributions, axis=0)


def annotation_points(annotation, source_fs):
    scale = TARGET_FS / float(source_fs)
    points = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        points.append((int(round(float(sample) * scale)), symbol))
    return points


def window_annotation_mask(points, window_start, window_end, symbols):
    selected = [
        (point, symbol)
        for point, symbol in points
        if symbol in symbols and window_start <= point < window_end
    ]
    mask = np.zeros(window_end - window_start, dtype=bool)
    radius = int(0.20 * TARGET_FS)
    for point, _ in selected:
        local = point - window_start
        left = max(0, local - radius)
        right = min(len(mask), local + radius + 1)
        mask[left:right] = True
    return selected, mask


def history_annotation_windows(points, history_start, symbols):
    has_annotation = np.zeros(N_WINDOWS, dtype=bool)
    for window_i in range(N_WINDOWS):
        start = history_start + window_i * STRIDE_PTS
        end = start + PTS_PER_WIN
        has_annotation[window_i] = any(
            symbol in symbols and start <= point < end
            for point, symbol in points
        )
    return has_annotation


def attribution_metrics(attribution, beat_mask):
    if not beat_mask.any() or beat_mask.all():
        return {
            "annotated_neighborhood_points": int(beat_mask.sum()),
            "mean_attribution_annotated": None,
            "mean_attribution_background": None,
            "annotated_to_background_ratio": None,
            "beat_neighborhood_average_precision": None,
        }
    annotated_mean = float(np.mean(attribution[beat_mask]))
    background_mean = float(np.mean(attribution[~beat_mask]))
    return {
        "annotated_neighborhood_points": int(beat_mask.sum()),
        "mean_attribution_annotated": annotated_mean,
        "mean_attribution_background": background_mean,
        "annotated_to_background_ratio": (
            annotated_mean / max(background_mean, 1e-12)
        ),
        "beat_neighborhood_average_precision": float(
            average_precision_score(beat_mask.astype(int), attribution)
        ),
    }


def plot_explanation(
    output_path,
    record_label,
    target_name,
    class_probability,
    risk_std,
    window_cam,
    annotated_windows,
    selected_window,
    waveform,
    attribution,
    selected_beats,
):
    figure = plt.figure(figsize=(14, 8))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 2.4, 0.35), hspace=0.35)

    axis_windows = figure.add_subplot(grid[0])
    colors = [
        "#ef4444" if annotated else "#3b82f6"
        for annotated in annotated_windows
    ]
    axis_windows.bar(np.arange(N_WINDOWS), window_cam, color=colors, alpha=0.85)
    axis_windows.axvline(selected_window, color="#facc15", linewidth=2)
    axis_windows.set_xlim(-0.5, N_WINDOWS - 0.5)
    axis_windows.set_ylim(0.0, 1.05)
    axis_windows.set_ylabel("Window Grad-CAM")
    axis_windows.set_title(
        f"{record_label} | target={target_name} | probability={class_probability:.3f} "
        f"| ensemble risk std={risk_std:.3f}"
    )

    axis_ecg = figure.add_subplot(grid[1])
    seconds = np.arange(len(waveform), dtype=float) / TARGET_FS
    points = np.column_stack([seconds, waveform])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colored_line = LineCollection(
        segments,
        cmap="inferno",
        norm=plt.Normalize(0.0, 1.0),
    )
    colored_line.set_array(attribution[:-1])
    colored_line.set_linewidth(1.2)
    axis_ecg.add_collection(colored_line)
    axis_ecg.set_xlim(seconds[0], seconds[-1])
    margin = max(0.1, float(np.ptp(waveform)) * 0.1)
    axis_ecg.set_ylim(float(waveform.min() - margin), float(waveform.max() + margin))
    for point, symbol in selected_beats:
        beat_second = point / TARGET_FS
        axis_ecg.axvline(beat_second, color="#22c55e", alpha=0.65, linewidth=1)
        axis_ecg.text(
            beat_second,
            axis_ecg.get_ylim()[1],
            symbol,
            color="#15803d",
            fontsize=8,
            ha="center",
            va="bottom",
        )
    axis_ecg.set_xlabel("Seconds within selected 30-second window")
    axis_ecg.set_ylabel("ECG amplitude")
    axis_ecg.set_title(
        f"Selected history window {selected_window}: ECG colored by integrated gradients"
    )

    axis_heat = figure.add_subplot(grid[2], sharex=axis_ecg)
    axis_heat.imshow(
        attribution[np.newaxis, :],
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        extent=(seconds[0], seconds[-1], 0, 1),
    )
    axis_heat.set_yticks([])
    axis_heat.set_xlabel("Attribution intensity")

    figure.colorbar(colored_line, ax=axis_ecg, label="Normalized attribution")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="201")
    ap.add_argument("--db", default="mitdb")
    ap.add_argument("--end-minute", type=float, default=11.0)
    ap.add_argument("--target-class", type=int, default=1)
    ap.add_argument("--annotation-symbols", default="V,E")
    ap.add_argument("--ig-steps", type=int, default=12)
    ap.add_argument(
        "--output-dir",
        default="results_v7_round6/ecg_level_attribution",
    )
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ecg, annotation, source_fs = load_record(
        base_dir,
        args.db,
        args.record,
        max_minutes=max(args.end_minute + 1.0, 20.0),
    )
    end_point = int(round(args.end_minute * 60 * TARGET_FS))
    if end_point < HISTORY_PTS or end_point > len(ecg):
        raise ValueError(
            f"end-minute must provide a full history and fit the record: "
            f"end_point={end_point}, history={HISTORY_PTS}, ecg={len(ecg)}"
        )

    bx, bx_rr = build_model_inputs(ecg, end_point)
    ensemble_output = serving.run_ensemble(bx, bx_rr)
    class_probability = float(
        ensemble_output["probs_fut"][0, args.target_class].cpu()
    )
    risk_std = float(ensemble_output["risk_std"][0].cpu())
    window_cam, member_cams = ensemble_window_gradcam(
        serving.models,
        bx,
        bx_rr,
        args.target_class,
    )
    selected_window = int(np.argmax(window_cam))
    signed_attr, magnitude_attr, member_attr = (
        selected_window_integrated_gradients(
            serving.models,
            bx,
            bx_rr,
            args.target_class,
            selected_window,
            args.ig_steps,
        )
    )

    history_start = end_point - HISTORY_PTS
    window_start = history_start + selected_window * STRIDE_PTS
    window_end = window_start + PTS_PER_WIN
    waveform = ecg[window_start:window_end]
    symbols = {
        symbol.strip()
        for symbol in args.annotation_symbols.split(",")
        if symbol.strip()
    }
    points = annotation_points(annotation, source_fs)
    annotated_windows = history_annotation_windows(
        points,
        history_start,
        symbols,
    )
    selected_beats_global, beat_mask = window_annotation_mask(
        points,
        window_start,
        window_end,
        symbols,
    )
    selected_beats_local = [
        (point - window_start, symbol)
        for point, symbol in selected_beats_global
    ]
    metrics = attribution_metrics(magnitude_attr, beat_mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.db}_{args.record}_class{args.target_class}_m{args.end_minute:g}"
    image_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    plot_explanation(
        image_path,
        record_label=f"{args.db}/{args.record} @ {args.end_minute:g} min",
        target_name=ENGLISH_CLASS_NAMES[args.target_class],
        class_probability=class_probability,
        risk_std=risk_std,
        window_cam=window_cam,
        annotated_windows=annotated_windows,
        selected_window=selected_window,
        waveform=waveform,
        attribution=magnitude_attr,
        selected_beats=selected_beats_local,
    )

    result = {
        "record": args.record,
        "database": args.db,
        "end_minute": args.end_minute,
        "target_class": args.target_class,
        "target_name": CLASS_NAMES[args.target_class],
        "target_probability": class_probability,
        "ensemble_risk_std": risk_std,
        "confidence": ensemble_output["confidence"][0],
        "integrated_gradient_steps": args.ig_steps,
        "selected_window_index": selected_window,
        "selected_window_start_sec": window_start / TARGET_FS,
        "selected_window_end_sec": window_end / TARGET_FS,
        "selected_window_cam": float(window_cam[selected_window]),
        "annotated_history_windows": int(annotated_windows.sum()),
        "selected_window_annotated_beats": len(selected_beats_local),
        "window_cam_annotated_mean": (
            float(window_cam[annotated_windows].mean())
            if annotated_windows.any() else None
        ),
        "window_cam_background_mean": (
            float(window_cam[~annotated_windows].mean())
            if (~annotated_windows).any() else None
        ),
        "member_window_cam_std_mean": float(member_cams.std(axis=0).mean()),
        "member_sample_attribution_std_mean": float(
            member_attr.std(axis=0).mean()
        ),
        "signed_attribution_positive_fraction": float(
            np.mean(signed_attr > 0)
        ),
        **metrics,
        "image_path": str(image_path),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
