"""
Search causal patient-adaptive and hysteresis alert policies.

This is a post-training policy experiment. It consumes saved per-window model
outputs, selects policies on validation, and can apply one locked validation
policy to test. Patient baselines use prior windows only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alert_policy import (
    contiguous_true_segments,
    event_target_for_task,
    score_for_task,
)
from evaluate_smoothed_alert_policy import (
    causal_smooth_by_patient,
    infer_horizon_sec,
)


BASELINE_NONE = "none"
BASELINE_METHODS = (
    BASELINE_NONE,
    "expanding_mean_std",
    "rolling_mean_std",
    "rolling_median_mad",
)
ADAPTIVE_MODES = ("none", "delta", "zscore")


def parse_float_list(text):
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def parse_int_list(text):
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def patient_index_groups(df):
    groups = []
    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        groups.append(patient_df.index.to_numpy())
    return groups


def causal_baseline_by_patient(
    df,
    score,
    method,
    window,
    patient_groups=None,
):
    """Return center, scale, and prior-history count for every window."""
    score = np.asarray(score, dtype=float)
    center = np.full(len(df), np.nan, dtype=float)
    scale = np.full(len(df), np.nan, dtype=float)
    history_count = np.zeros(len(df), dtype=int)

    if method == BASELINE_NONE:
        center.fill(0.0)
        scale.fill(1.0)
        return center, scale, history_count

    if method not in BASELINE_METHODS:
        raise ValueError(f"unsupported baseline method: {method}")

    groups = patient_groups if patient_groups is not None else patient_index_groups(df)
    for idx in groups:
        values = score[idx]

        for local_i, global_i in enumerate(idx):
            start = 0
            if method.startswith("rolling_"):
                start = max(0, local_i - int(window))
            history = values[start:local_i]
            finite = history[np.isfinite(history)]
            history_count[global_i] = len(finite)
            if not len(finite):
                continue

            if method in {"expanding_mean_std", "rolling_mean_std"}:
                center[global_i] = float(np.mean(finite))
                scale[global_i] = float(np.std(finite, ddof=0))
            else:
                median = float(np.median(finite))
                center[global_i] = median
                scale[global_i] = float(
                    1.4826 * np.median(np.abs(finite - median))
                )

    return center, scale, history_count


def adaptive_entry_mask(
    score,
    global_threshold,
    baseline_center,
    baseline_scale,
    history_count,
    min_history,
    adaptive_mode,
    delta,
    z_threshold,
    scale_floor,
    warmup_behavior,
):
    score = np.asarray(score, dtype=float)
    global_pass = score >= float(global_threshold)
    if adaptive_mode == "none":
        return global_pass

    ready = np.asarray(history_count) >= int(min_history)
    if adaptive_mode == "delta":
        adaptive_pass = score >= (
            np.asarray(baseline_center, dtype=float) + float(delta)
        )
    elif adaptive_mode == "zscore":
        denominator = np.maximum(
            np.asarray(baseline_scale, dtype=float),
            float(scale_floor),
        )
        z_score = (
            score - np.asarray(baseline_center, dtype=float)
        ) / denominator
        adaptive_pass = z_score >= float(z_threshold)
    else:
        raise ValueError(f"unsupported adaptive mode: {adaptive_mode}")

    if warmup_behavior == "global":
        adaptive_pass = np.where(ready, adaptive_pass, True)
    elif warmup_behavior == "suppress":
        adaptive_pass = ready & adaptive_pass
    else:
        raise ValueError(f"unsupported warmup behavior: {warmup_behavior}")
    return global_pass & np.asarray(adaptive_pass, dtype=bool)


def hysteresis_alert_by_patient(
    df,
    score,
    entry_mask,
    exit_threshold,
    consecutive_k,
    patient_groups=None,
):
    """Enter after k candidates; once active, remain active above exit threshold."""
    score = np.asarray(score, dtype=float)
    entry_mask = np.asarray(entry_mask, dtype=bool)
    alert = np.zeros(len(df), dtype=bool)

    groups = patient_groups if patient_groups is not None else patient_index_groups(df)
    for idx in groups:
        active = False
        run = 0

        for global_i in idx:
            if active:
                if score[global_i] >= float(exit_threshold):
                    alert[global_i] = True
                    continue
                active = False
                run = 0

            if entry_mask[global_i]:
                run += 1
            else:
                run = 0

            if run >= int(consecutive_k):
                active = True
                alert[global_i] = True

    return alert


def build_event_contexts(df, event_target, horizon_sec, patient_groups=None):
    event_target = np.asarray(event_target, dtype=bool)
    groups = patient_groups if patient_groups is not None else patient_index_groups(df)
    contexts = []
    for idx in groups:
        times = df.loc[idx, "timestamp_sec"].to_numpy(dtype=float)
        target = event_target[idx]
        events = []
        for start_i, end_i in contiguous_true_segments(target):
            start_t = float(times[start_i])
            end_t = float(times[end_i])
            events.append({
                "incident": start_i > 0,
                "start_t": start_t,
                "eligible": (
                    (times >= start_t - float(horizon_sec))
                    & (times <= end_t)
                ),
            })
        contexts.append({
            "idx": idx,
            "times": times,
            "target": target,
            "events": events,
            "patient_hours": (
                max(float(times[-1] - times[0]), 0.0) / 3600.0
                if len(times) > 1 else 0.0
            ),
        })
    return contexts


def evaluate_alert_array(
    df,
    event_target,
    alert,
    horizon_sec,
    event_contexts=None,
):
    alert = np.asarray(alert, dtype=bool)
    all_events = 0
    detected_events = 0
    incident_events = 0
    detected_incident_events = 0
    lead_times = []
    incident_lead_times = []
    false_windows = 0
    false_episodes = 0
    patient_hours = 0.0

    contexts = (
        event_contexts
        if event_contexts is not None
        else build_event_contexts(df, event_target, horizon_sec)
    )
    for context in contexts:
        idx = context["idx"]
        times = context["times"]
        target = context["target"]
        patient_alert = alert[idx]
        patient_hours += context["patient_hours"]

        covered_alert = np.zeros(len(idx), dtype=bool)
        for event in context["events"]:
            all_events += 1
            incident = event["incident"]
            if incident:
                incident_events += 1
            start_t = event["start_t"]
            eligible = event["eligible"]
            event_alert_idx = np.where(eligible & patient_alert)[0]
            if not len(event_alert_idx):
                continue

            detected_events += 1
            lead = start_t - float(times[event_alert_idx[0]])
            lead_times.append(lead)
            if incident:
                detected_incident_events += 1
                incident_lead_times.append(lead)
            covered_alert |= eligible & patient_alert

        false_mask = patient_alert & ~covered_alert & ~target
        false_windows += int(false_mask.sum())
        false_episodes += len(contiguous_true_segments(false_mask))

    lead_arr = np.asarray(lead_times, dtype=float)
    incident_lead_arr = np.asarray(incident_lead_times, dtype=float)
    return {
        "events": int(all_events),
        "event_recall": detected_events / all_events if all_events else np.nan,
        "detected_events": int(detected_events),
        "incident_events": int(incident_events),
        "incident_event_recall": (
            detected_incident_events / incident_events
            if incident_events else np.nan
        ),
        "detected_incident_events": int(detected_incident_events),
        "median_lead_time_sec": (
            float(np.nanmedian(lead_arr)) if len(lead_arr) else np.nan
        ),
        "incident_median_lead_time_sec": (
            float(np.nanmedian(incident_lead_arr))
            if len(incident_lead_arr) else np.nan
        ),
        "false_alert_windows": int(false_windows),
        "false_alert_episodes": int(false_episodes),
        "patient_hours": float(patient_hours),
        "false_alert_windows_per_patient_hour": (
            false_windows / patient_hours if patient_hours > 0 else np.nan
        ),
        "false_alert_episodes_per_patient_hour": (
            false_episodes / patient_hours if patient_hours > 0 else np.nan
        ),
    }


def evaluate_policy(
    df,
    score,
    event_target,
    baseline_values,
    task,
    source,
    smoothing_method,
    smoothing_window,
    smoothing_alpha,
    enter_threshold,
    exit_threshold,
    consecutive_k,
    baseline_method,
    baseline_window,
    min_history,
    adaptive_mode,
    delta,
    z_threshold,
    scale_floor,
    warmup_behavior,
    horizon_sec,
    selection,
    patient_groups=None,
    event_contexts=None,
):
    center, scale, history_count = baseline_values
    entry_mask = adaptive_entry_mask(
        score=score,
        global_threshold=enter_threshold,
        baseline_center=center,
        baseline_scale=scale,
        history_count=history_count,
        min_history=min_history,
        adaptive_mode=adaptive_mode,
        delta=delta,
        z_threshold=z_threshold,
        scale_floor=scale_floor,
        warmup_behavior=warmup_behavior,
    )
    alert = hysteresis_alert_by_patient(
        df=df,
        score=score,
        entry_mask=entry_mask,
        exit_threshold=exit_threshold,
        consecutive_k=consecutive_k,
        patient_groups=patient_groups,
    )
    metrics = evaluate_alert_array(
        df=df,
        event_target=event_target,
        alert=alert,
        horizon_sec=horizon_sec,
        event_contexts=event_contexts,
    )
    return {
        "task": task,
        "selection": selection,
        "probability_source": source,
        "smoothing_method": smoothing_method,
        "smoothing_window": int(smoothing_window),
        "smoothing_alpha": float(smoothing_alpha),
        "enter_threshold": float(enter_threshold),
        "exit_threshold": float(exit_threshold),
        "consecutive_k": int(consecutive_k),
        "baseline_method": baseline_method,
        "baseline_window": int(baseline_window),
        "min_history": int(min_history),
        "adaptive_mode": adaptive_mode,
        "delta": float(delta),
        "z_threshold": float(z_threshold),
        "scale_floor": float(scale_floor),
        "warmup_behavior": warmup_behavior,
        "event_horizon_sec": float(horizon_sec),
        **metrics,
    }


def policy_configs(args):
    enter_thresholds = parse_float_list(args.enter_thresholds)
    exit_thresholds = parse_float_list(args.exit_thresholds)
    consecutive_values = parse_int_list(args.consecutive_k)
    baseline_methods = parse_str_list(args.baseline_methods)
    baseline_windows = parse_int_list(args.baseline_windows)
    min_history_values = parse_int_list(args.min_history)
    adaptive_modes = parse_str_list(args.adaptive_modes)
    deltas = parse_float_list(args.deltas)
    z_thresholds = parse_float_list(args.z_thresholds)

    for method in baseline_methods:
        if method not in BASELINE_METHODS:
            raise ValueError(f"unsupported baseline method: {method}")
        method_windows = [0] if method in {BASELINE_NONE, "expanding_mean_std"} else baseline_windows
        method_histories = [0] if method == BASELINE_NONE else min_history_values

        for window in method_windows:
            for min_history in method_histories:
                if method == BASELINE_NONE:
                    mode_values = ["none"]
                else:
                    mode_values = [x for x in adaptive_modes if x != "none"]

                for mode in mode_values:
                    delta_values = deltas if mode == "delta" else [0.0]
                    z_values = z_thresholds if mode == "zscore" else [0.0]
                    for delta in delta_values:
                        for z_threshold in z_values:
                            for enter in enter_thresholds:
                                for exit_threshold in exit_thresholds:
                                    if exit_threshold > enter:
                                        continue
                                    for consecutive_k in consecutive_values:
                                        yield {
                                            "enter_threshold": enter,
                                            "exit_threshold": exit_threshold,
                                            "consecutive_k": consecutive_k,
                                            "baseline_method": method,
                                            "baseline_window": window,
                                            "min_history": min_history,
                                            "adaptive_mode": mode,
                                            "delta": delta,
                                            "z_threshold": z_threshold,
                                        }


def is_round5_anchor(row, args):
    return (
        row["adaptive_mode"] == "none"
        and row["baseline_method"] == BASELINE_NONE
        and np.isclose(row["enter_threshold"], args.anchor_threshold)
        and np.isclose(row["exit_threshold"], args.anchor_threshold)
        and int(row["consecutive_k"]) == int(args.anchor_consecutive_k)
    )


def choose_policy(grid, args):
    anchor_rows = grid[grid.apply(lambda row: is_round5_anchor(row, args), axis=1)]
    if len(anchor_rows) != 1:
        raise RuntimeError(
            "Round5 anchor is missing or duplicated. Ensure the search includes "
            "enter=exit=anchor-threshold, baseline=none, adaptive=none, and anchor k."
        )
    anchor = anchor_rows.iloc[0]
    event_floor = max(
        float(args.min_event_recall),
        float(anchor["event_recall"]) - float(args.max_event_recall_drop),
    )
    incident_floor = max(
        float(args.min_incident_recall),
        float(anchor["incident_event_recall"]) - float(args.max_incident_recall_drop),
    )
    false_window_limit = (
        float(anchor["false_alert_windows_per_patient_hour"])
        * (1.0 + float(args.max_false_window_increase))
    )

    feasible = grid[
        (grid["event_recall"] >= event_floor)
        & (grid["incident_event_recall"] >= incident_floor)
        & (grid["median_lead_time_sec"] >= float(args.min_median_lead_sec))
        & (
            grid["false_alert_windows_per_patient_hour"]
            <= false_window_limit
        )
    ].copy()

    if len(feasible):
        selected = feasible.sort_values(
            [
                "false_alert_episodes_per_patient_hour",
                "false_alert_windows_per_patient_hour",
                "event_recall",
                "incident_event_recall",
                "median_lead_time_sec",
            ],
            ascending=[True, True, False, False, False],
        ).iloc[0].copy()
        selection = "adaptive_val_selected"
    else:
        selected = grid.sort_values(
            [
                "event_recall",
                "incident_event_recall",
                "false_alert_episodes_per_patient_hour",
            ],
            ascending=[False, False, True],
        ).iloc[0].copy()
        selection = "adaptive_val_fallback"

    selected["selection"] = selection
    selected["validation_event_recall_floor"] = event_floor
    selected["validation_incident_recall_floor"] = incident_floor
    selected["validation_min_median_lead_sec"] = float(args.min_median_lead_sec)
    selected["validation_false_window_limit"] = false_window_limit
    selected["anchor_event_recall"] = float(anchor["event_recall"])
    selected["anchor_incident_event_recall"] = float(
        anchor["incident_event_recall"]
    )
    selected["anchor_median_lead_time_sec"] = float(
        anchor["median_lead_time_sec"]
    )
    selected["anchor_false_windows_per_patient_hour"] = float(
        anchor["false_alert_windows_per_patient_hour"]
    )
    selected["anchor_false_episodes_per_patient_hour"] = float(
        anchor["false_alert_episodes_per_patient_hour"]
    )
    return anchor.to_dict(), selected.to_dict(), feasible


def load_reference_policy(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = [rows]
    if len(rows) != 1:
        raise ValueError("reference policy must contain exactly one policy row")
    return rows[0]


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_summary(path, anchor, selected, feasible_count, grid_count, locked):
    lines = [
        "# Round6 Adaptive Alert Policy",
        "",
        f"- mode: {'locked test application' if locked else 'validation search'}",
        f"- evaluated policies: {grid_count}",
        f"- feasible validation policies: {feasible_count}",
        "",
    ]
    if anchor is not None:
        lines.extend([
            "## Round5 Anchor",
            "",
            f"- event recall: {anchor['event_recall']:.3f}",
            f"- incident recall: {anchor['incident_event_recall']:.3f}",
            f"- median lead: {anchor['median_lead_time_sec']:.0f} s",
            f"- false windows/hr: {anchor['false_alert_windows_per_patient_hour']:.2f}",
            f"- false episodes/hr: {anchor['false_alert_episodes_per_patient_hour']:.2f}",
            "",
        ])
    lines.extend([
        "## Selected Policy",
        "",
        (
            f"- entry/exit: {selected['enter_threshold']:.3f} / "
            f"{selected['exit_threshold']:.3f}"
        ),
        f"- consecutive k: {int(selected['consecutive_k'])}",
        (
            f"- adaptive baseline: {selected['baseline_method']} "
            f"(window={int(selected['baseline_window'])}, "
            f"min_history={int(selected['min_history'])})"
        ),
        (
            f"- adaptive condition: {selected['adaptive_mode']} "
            f"(delta={selected['delta']:.3f}, z={selected['z_threshold']:.3f})"
        ),
        f"- event recall: {selected['event_recall']:.3f}",
        f"- incident recall: {selected['incident_event_recall']:.3f}",
        f"- median lead: {selected['median_lead_time_sec']:.0f} s",
        f"- false windows/hr: {selected['false_alert_windows_per_patient_hour']:.2f}",
        f"- false episodes/hr: {selected['false_alert_episodes_per_patient_hour']:.2f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task", default="future_arrhythmia")
    ap.add_argument("--probability-source", default="uncalibrated")
    ap.add_argument("--smoothing-method", default="ewma")
    ap.add_argument("--smoothing-window", type=int, default=1)
    ap.add_argument("--smoothing-alpha", type=float, default=0.65)
    ap.add_argument("--enter-thresholds", default="0.08,0.10,0.12,0.14,0.16,0.18")
    ap.add_argument("--exit-thresholds", default="0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18")
    ap.add_argument("--consecutive-k", default="1,2,3,4")
    ap.add_argument(
        "--baseline-methods",
        default="none,expanding_mean_std,rolling_mean_std,rolling_median_mad",
    )
    ap.add_argument("--baseline-windows", default="6,12,30")
    ap.add_argument("--min-history", default="3,5,10")
    ap.add_argument("--adaptive-modes", default="delta,zscore")
    ap.add_argument("--deltas", default="0.02,0.05,0.10")
    ap.add_argument("--z-thresholds", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--scale-floor", type=float, default=0.02)
    ap.add_argument(
        "--warmup-behavior",
        choices=("global", "suppress"),
        default="global",
    )
    ap.add_argument("--anchor-threshold", type=float, default=0.10)
    ap.add_argument("--anchor-consecutive-k", type=int, default=2)
    ap.add_argument("--min-event-recall", type=float, default=0.90)
    ap.add_argument("--min-incident-recall", type=float, default=0.88)
    ap.add_argument("--min-median-lead-sec", type=float, default=260.0)
    ap.add_argument("--max-event-recall-drop", type=float, default=0.02)
    ap.add_argument("--max-incident-recall-drop", type=float, default=0.03)
    ap.add_argument("--max-false-window-increase", type=float, default=0.05)
    ap.add_argument("--horizon-sec", type=float, default=None)
    ap.add_argument("--reference-policy", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.outputs_csv).reset_index(drop=True)
    horizon_sec = infer_horizon_sec(df, override=args.horizon_sec)
    raw_score = score_for_task(df, args.task, args.probability_source)
    score = causal_smooth_by_patient(
        df,
        raw_score,
        method=args.smoothing_method,
        window=args.smoothing_window,
        alpha=args.smoothing_alpha,
    )
    event_target = event_target_for_task(df, args.task)
    patient_groups = patient_index_groups(df)
    event_contexts = build_event_contexts(
        df=df,
        event_target=event_target,
        horizon_sec=horizon_sec,
        patient_groups=patient_groups,
    )
    baseline_cache = {}

    def baseline_values(method, window):
        key = (method, int(window))
        if key not in baseline_cache:
            baseline_cache[key] = causal_baseline_by_patient(
                df=df,
                score=score,
                method=method,
                window=window,
                patient_groups=patient_groups,
            )
        return baseline_cache[key]

    if args.reference_policy:
        ref = load_reference_policy(args.reference_policy)
        selected = evaluate_policy(
            df=df,
            score=score,
            event_target=event_target,
            baseline_values=baseline_values(
                ref["baseline_method"],
                int(ref["baseline_window"]),
            ),
            task=args.task,
            source=args.probability_source,
            smoothing_method=args.smoothing_method,
            smoothing_window=args.smoothing_window,
            smoothing_alpha=args.smoothing_alpha,
            enter_threshold=float(ref["enter_threshold"]),
            exit_threshold=float(ref["exit_threshold"]),
            consecutive_k=int(ref["consecutive_k"]),
            baseline_method=ref["baseline_method"],
            baseline_window=int(ref["baseline_window"]),
            min_history=int(ref["min_history"]),
            adaptive_mode=ref["adaptive_mode"],
            delta=float(ref["delta"]),
            z_threshold=float(ref["z_threshold"]),
            scale_floor=float(ref["scale_floor"]),
            warmup_behavior=ref["warmup_behavior"],
            horizon_sec=horizon_sec,
            selection="fixed_adaptive_val_policy",
            patient_groups=patient_groups,
            event_contexts=event_contexts,
        )
        grid = pd.DataFrame([selected])
        anchor = None
        feasible_count = 0
    else:
        rows = []
        configs = list(policy_configs(args))
        print(f"[search] evaluating {len(configs)} validation policies")
        for i, config in enumerate(configs, start=1):
            rows.append(
                evaluate_policy(
                    df=df,
                    score=score,
                    event_target=event_target,
                    baseline_values=baseline_values(
                        config["baseline_method"],
                        config["baseline_window"],
                    ),
                    task=args.task,
                    source=args.probability_source,
                    smoothing_method=args.smoothing_method,
                    smoothing_window=args.smoothing_window,
                    smoothing_alpha=args.smoothing_alpha,
                    scale_floor=args.scale_floor,
                    warmup_behavior=args.warmup_behavior,
                    horizon_sec=horizon_sec,
                    selection="adaptive_grid",
                    patient_groups=patient_groups,
                    event_contexts=event_contexts,
                    **config,
                )
            )
            if i % 2000 == 0:
                print(f"[search] completed {i}/{len(configs)}")
        grid = pd.DataFrame(rows)
        anchor, selected, feasible = choose_policy(grid, args)
        feasible_count = len(feasible)

    selected_df = pd.DataFrame([selected])
    grid.to_csv(out_dir / "adaptive_alert_policy_grid.csv", index=False)
    selected_df.to_csv(out_dir / "adaptive_alert_policy_selected.csv", index=False)
    with (out_dir / "adaptive_alert_policy_selected.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump([json_value(selected)], f, indent=2, ensure_ascii=False)
    write_summary(
        out_dir / "adaptive_alert_policy_summary.md",
        anchor=anchor,
        selected=selected,
        feasible_count=feasible_count,
        grid_count=len(grid),
        locked=args.reference_policy is not None,
    )

    print(f"Saved grid: {out_dir / 'adaptive_alert_policy_grid.csv'}")
    print(
        f"Selected: enter={selected['enter_threshold']:.2f} "
        f"exit={selected['exit_threshold']:.2f} "
        f"k={int(selected['consecutive_k'])} "
        f"baseline={selected['baseline_method']} "
        f"mode={selected['adaptive_mode']} "
        f"event_recall={selected['event_recall']:.3f} "
        f"incident_recall={selected['incident_event_recall']:.3f} "
        f"lead={selected['median_lead_time_sec']:.0f}s "
        f"false_ep/hr={selected['false_alert_episodes_per_patient_hour']:.2f} "
        f"false_win/hr={selected['false_alert_windows_per_patient_hour']:.2f}"
    )


if __name__ == "__main__":
    main()
