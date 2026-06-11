"""
Evaluate causal patient-level risk smoothing for real-time ECG alert policies.

This is a post-training experiment: it consumes saved final_validation_outputs.csv,
selects smoothing/threshold/consecutive-k policies on validation, and can apply
the locked validation-selected policies to test outputs. It never changes model
weights or dataset tensors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alert_policy import (
    HORIZON_SEC,
    contiguous_true_segments,
    event_target_for_task,
    realtime_consecutive_filter,
    score_for_task,
)


TASKS = ("future_arrhythmia", "future_vt_vf")
DEFAULT_SOURCES = ("temp_scaled",)


def infer_horizon_sec(df, override=None):
    if override is not None:
        return float(override)
    required = {"future_start_sec", "future_end_sec"}
    if required.issubset(df.columns):
        span = (
            df["future_end_sec"].to_numpy(dtype=float) -
            df["future_start_sec"].to_numpy(dtype=float)
        )
        span = span[np.isfinite(span) & (span > 0)]
        if len(span):
            return float(np.nanmedian(span))
    return float(HORIZON_SEC)


def causal_smooth_by_patient(df, score, method, window, alpha):
    score = np.asarray(score, dtype=float)
    out = np.full(len(df), np.nan, dtype=float)

    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        idx = patient_df.index.to_numpy()
        s = pd.Series(score[idx])

        if method == "raw":
            smoothed = s
        elif method == "rolling_mean":
            smoothed = s.rolling(window=int(window), min_periods=1).mean()
        elif method == "ewma":
            smoothed = s.ewm(alpha=float(alpha), adjust=False).mean()
        elif method == "rolling_median":
            smoothed = s.rolling(window=int(window), min_periods=1).median()
        else:
            raise ValueError(f"unsupported smoothing method: {method}")

        out[idx] = smoothed.to_numpy(dtype=float)

    return np.clip(out, 0.0, 1.0)


def apply_refractory(alert, refractory_windows):
    alert = np.asarray(alert, dtype=bool)
    refractory_windows = int(refractory_windows)
    if refractory_windows <= 0:
        return alert.copy()

    out = alert.copy()
    cool_until = -1
    in_episode = False
    for i, value in enumerate(alert):
        if i <= cool_until:
            out[i] = False
            continue

        if value and not in_episode:
            in_episode = True
        elif not value and in_episode:
            in_episode = False
            cool_until = i + refractory_windows - 1

    return out


def evaluate_smoothed_policy(
    df,
    task,
    source,
    threshold,
    consecutive_k,
    smoothing_method,
    smoothing_window,
    smoothing_alpha,
    refractory_windows,
    horizon_sec,
    locked=False,
):
    score = score_for_task(df, task, source)
    score = causal_smooth_by_patient(
        df,
        score,
        method=smoothing_method,
        window=smoothing_window,
        alpha=smoothing_alpha,
    )
    event_target = event_target_for_task(df, task)
    raw_alert = score >= float(threshold)

    all_events = 0
    detected_events = 0
    incident_events = 0
    detected_incident_events = 0
    lead_times = []
    incident_lead_times = []
    false_windows = 0
    false_episodes = 0
    patient_hours = 0.0

    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        idx = patient_df.index.to_numpy()
        times = patient_df["timestamp_sec"].to_numpy(dtype=float)
        target = event_target[idx]
        alert = realtime_consecutive_filter(raw_alert[idx], consecutive_k)
        alert = apply_refractory(alert, refractory_windows)

        if len(times) > 1:
            patient_hours += max(float(times[-1] - times[0]), 0.0) / 3600.0

        covered_alert = np.zeros(len(idx), dtype=bool)
        for start_i, end_i in contiguous_true_segments(target):
            all_events += 1
            incident = start_i > 0
            if incident:
                incident_events += 1
            start_t = float(times[start_i])
            end_t = float(times[end_i])
            eligible = (times >= start_t - horizon_sec) & (times <= end_t)
            event_alert_idx = np.where(eligible & alert)[0]
            if len(event_alert_idx):
                detected_events += 1
                first_alert_t = float(times[event_alert_idx[0]])
                lead = start_t - first_alert_t
                lead_times.append(lead)
                if incident:
                    detected_incident_events += 1
                    incident_lead_times.append(lead)
                covered_alert |= eligible & alert

        false_mask = alert & ~covered_alert & ~target
        false_windows += int(false_mask.sum())
        false_episodes += len(contiguous_true_segments(false_mask))

    lead_arr = np.asarray(lead_times, dtype=float)
    inc_lead_arr = np.asarray(incident_lead_times, dtype=float)
    selection = "fixed_smoothed" if locked else "smoothed_val_selected"
    return {
        "task": task,
        "selection": selection,
        "probability_source": source,
        "threshold": float(threshold),
        "consecutive_k": int(consecutive_k),
        "smoothing_method": smoothing_method,
        "smoothing_window": int(smoothing_window),
        "smoothing_alpha": float(smoothing_alpha),
        "refractory_windows": int(refractory_windows),
        "event_horizon_sec": float(horizon_sec),
        "events": int(all_events),
        "event_recall": detected_events / all_events if all_events else np.nan,
        "detected_events": int(detected_events),
        "incident_events": int(incident_events),
        "incident_event_recall": detected_incident_events / incident_events if incident_events else np.nan,
        "detected_incident_events": int(detected_incident_events),
        "median_lead_time_sec": float(np.nanmedian(lead_arr)) if len(lead_arr) else np.nan,
        "incident_median_lead_time_sec": float(np.nanmedian(inc_lead_arr)) if len(inc_lead_arr) else np.nan,
        "false_alert_windows": int(false_windows),
        "false_alert_episodes": int(false_episodes),
        "patient_hours": float(patient_hours),
        "false_alert_windows_per_patient_hour": false_windows / patient_hours if patient_hours > 0 else np.nan,
        "false_alert_episodes_per_patient_hour": false_episodes / patient_hours if patient_hours > 0 else np.nan,
    }


def choose_policy(rows, min_event_recall, max_recall_drop, max_false_window_increase):
    grid = pd.DataFrame(rows)
    selected = []
    for task, group in grid.groupby("task", dropna=False):
        raw = group[
            (group["smoothing_method"] == "raw")
            & (group["smoothing_window"] == 1)
            & (group["refractory_windows"] == 0)
        ]
        raw_feasible = raw[raw["event_recall"] >= float(min_event_recall)]
        if len(raw_feasible):
            raw_anchor = raw_feasible.sort_values(
                ["false_alert_windows_per_patient_hour", "false_alert_episodes_per_patient_hour"],
                ascending=[True, True],
            ).iloc[0]
        elif len(raw):
            raw_anchor = raw.sort_values(
                ["event_recall", "false_alert_windows_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0]
        else:
            raw_anchor = group.sort_values(
                ["event_recall", "false_alert_windows_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0]

        recall_floor = max(
            float(min_event_recall),
            float(raw_anchor["event_recall"]) - float(max_recall_drop),
        )
        false_window_limit = (
            float(raw_anchor["false_alert_windows_per_patient_hour"])
            * (1.0 + float(max_false_window_increase))
        )
        feasible = group[
            (group["event_recall"] >= recall_floor)
            & (group["false_alert_windows_per_patient_hour"] <= false_window_limit)
        ]
        if len(feasible):
            best = feasible.sort_values(
                [
                    "false_alert_episodes_per_patient_hour",
                    "false_alert_windows_per_patient_hour",
                    "consecutive_k",
                    "threshold",
                ],
                ascending=[True, True, True, False],
            ).iloc[0]
        else:
            best = group.sort_values(
                ["event_recall", "false_alert_episodes_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0]
        row = best.to_dict()
        row["validation_recall_floor"] = float(recall_floor)
        row["validation_false_window_limit"] = float(false_window_limit)
        row["raw_anchor_threshold"] = float(raw_anchor["threshold"])
        row["raw_anchor_consecutive_k"] = int(raw_anchor["consecutive_k"])
        row["raw_anchor_event_recall"] = float(raw_anchor["event_recall"])
        row["raw_anchor_false_windows_per_patient_hour"] = float(
            raw_anchor["false_alert_windows_per_patient_hour"]
        )
        row["raw_anchor_false_episodes_per_patient_hour"] = float(
            raw_anchor["false_alert_episodes_per_patient_hour"]
        )
        selected.append(row)
    return selected


def apply_reference(rows, reference):
    grid = pd.DataFrame(rows)
    selected = []
    for _, ref in reference.iterrows():
        task_rows = grid[grid["task"] == ref["task"]]
        matches = task_rows[
            (task_rows["probability_source"] == ref["probability_source"])
            & np.isclose(task_rows["threshold"], float(ref["threshold"]))
            & (task_rows["consecutive_k"] == int(ref["consecutive_k"]))
            & (task_rows["smoothing_method"] == ref["smoothing_method"])
            & (task_rows["smoothing_window"] == int(ref["smoothing_window"]))
            & np.isclose(task_rows["smoothing_alpha"], float(ref["smoothing_alpha"]))
            & (task_rows["refractory_windows"] == int(ref["refractory_windows"]))
        ]
        if len(matches):
            row = matches.iloc[0].to_dict()
            row["selection"] = "fixed_smoothed"
            selected.append(row)
        elif len(task_rows):
            fallback = task_rows.sort_values(
                ["event_recall", "false_alert_episodes_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0].to_dict()
            fallback["selection"] = "fixed_smoothed_fallback"
            fallback["reference_policy_missing"] = True
            selected.append(fallback)
    return selected


def parse_float_list(text):
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def parse_int_list(text):
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--thresholds", default="0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.20,0.24,0.28,0.32,0.36,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--consecutive-k", default="1,2,3,4,5,6,8,10,12")
    ap.add_argument("--smoothing-methods", default="raw,rolling_mean,ewma,rolling_median")
    ap.add_argument("--smoothing-windows", default="1,2,3,4,5,6,8")
    ap.add_argument("--smoothing-alphas", default="0.20,0.35,0.50,0.65,0.80")
    ap.add_argument("--refractory-windows", default="0,2,4,6")
    ap.add_argument("--probability-sources", default=",".join(DEFAULT_SOURCES),
                    help="Comma-separated probability sources: temp_scaled,uncalibrated")
    ap.add_argument("--horizon-sec", type=float, default=None,
                    help="Override event-credit horizon; default infers future_end_sec - future_start_sec.")
    ap.add_argument("--min-event-recall", type=float, default=0.85)
    ap.add_argument("--max-recall-drop", type=float, default=0.03)
    ap.add_argument("--max-false-window-increase", type=float, default=0.05,
                    help="Maximum allowed false-window burden increase over the raw validation anchor.")
    ap.add_argument("--reference-policies", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.outputs_csv)
    horizon_sec = infer_horizon_sec(df, override=args.horizon_sec)
    print(f"[config] event horizon={horizon_sec:.0f}s")
    thresholds = parse_float_list(args.thresholds)
    consecutive_values = parse_int_list(args.consecutive_k)
    methods = [x.strip() for x in str(args.smoothing_methods).split(",") if x.strip()]
    windows = parse_int_list(args.smoothing_windows)
    alphas = parse_float_list(args.smoothing_alphas)
    refractory_values = parse_int_list(args.refractory_windows)
    sources = [x.strip() for x in str(args.probability_sources).split(",") if x.strip()]

    rows = []
    locked = args.reference_policies is not None
    for task in TASKS:
        for source in sources:
            for method in methods:
                method_windows = [1] if method in {"raw", "ewma"} else windows
                method_alphas = [1.0] if method != "ewma" else alphas
                for window in method_windows:
                    for alpha in method_alphas:
                        for threshold in thresholds:
                            for consecutive_k in consecutive_values:
                                for refractory_windows in refractory_values:
                                    rows.append(
                                        evaluate_smoothed_policy(
                                            df,
                                            task=task,
                                            source=source,
                                            threshold=threshold,
                                            consecutive_k=consecutive_k,
                                            smoothing_method=method,
                                            smoothing_window=window,
                                            smoothing_alpha=alpha,
                                            refractory_windows=refractory_windows,
                                            horizon_sec=horizon_sec,
                                            locked=locked,
                                        )
                                    )

    grid = pd.DataFrame(rows)
    if args.reference_policies:
        reference = pd.read_csv(args.reference_policies)
        selected = apply_reference(rows, reference)
    else:
        selected = choose_policy(
            rows,
            min_event_recall=args.min_event_recall,
            max_recall_drop=args.max_recall_drop,
            max_false_window_increase=args.max_false_window_increase,
        )

    selected_df = pd.DataFrame(selected)
    grid.to_csv(out_dir / "smoothed_alert_policy_grid.csv", index=False)
    selected_df.to_csv(out_dir / "smoothed_alert_policy_selected.csv", index=False)
    with (out_dir / "smoothed_alert_policy_selected.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"Saved smoothed alert policy grid: {out_dir / 'smoothed_alert_policy_grid.csv'}")
    for row in selected:
        print(
            f"  {row['task']}: {row['smoothing_method']} "
            f"w={int(row['smoothing_window'])} alpha={row['smoothing_alpha']:.2f} "
            f"thr={row['threshold']:.2f} k={int(row['consecutive_k'])} "
            f"refrac={int(row['refractory_windows'])} "
            f"event_recall={row['event_recall']:.3f} "
            f"false_ep/hr={row['false_alert_episodes_per_patient_hour']:.2f}"
        )


if __name__ == "__main__":
    main()
