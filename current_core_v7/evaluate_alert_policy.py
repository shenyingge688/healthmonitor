"""
Evaluate real-time alert policies on saved final-validation outputs.

The model emits per-window risks. This script evaluates deployment-style alarm
filters such as "fire only after K consecutive positive windows" so early-warning
recall and false alert episodes can be compared without retraining the model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_SEC = 300.0


def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segments = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def realtime_consecutive_filter(raw_alert, k):
    """Fire at the kth consecutive alert and keep firing while the run continues."""
    raw_alert = np.asarray(raw_alert, dtype=bool)
    if k <= 1:
        return raw_alert.copy()
    out = np.zeros_like(raw_alert, dtype=bool)
    run = 0
    for i, value in enumerate(raw_alert):
        run = run + 1 if value else 0
        if run >= k:
            out[i] = True
    return out


def score_for_task(df, task, source):
    prefix = "prob_fut_temp_" if source == "temp_scaled" else "prob_fut_"
    if task == "future_arrhythmia":
        return 1.0 - df[f"{prefix}normal"].to_numpy(dtype=float)
    if task == "future_vt_vf":
        return (
            df[f"{prefix}vf"].to_numpy(dtype=float) +
            df[f"{prefix}vt"].to_numpy(dtype=float)
        )
    raise ValueError(f"unsupported task: {task}")


def event_target_for_task(df, task):
    cur = df["target_cur"].to_numpy(dtype=int)
    if task == "future_arrhythmia":
        return cur != 0
    if task == "future_vt_vf":
        return np.isin(cur, [3, 4])
    raise ValueError(f"unsupported task: {task}")


def evaluate_policy(df, task, source, selection, threshold, consecutive_k):
    score = score_for_task(df, task, source)
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
            eligible = (times >= start_t - HORIZON_SEC) & (times <= end_t)
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
    return {
        "task": task,
        "selection": selection,
        "probability_source": source,
        "threshold": float(threshold),
        "consecutive_k": int(consecutive_k),
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


def normalize_selection_for_split(selection, locked=False):
    selection = str(selection)
    if locked and not selection.startswith("fixed_"):
        return f"fixed_{selection}"
    if not locked and selection.startswith("fixed_"):
        return selection.replace("fixed_", "", 1)
    return selection


def choose_policy(rows, min_event_recall=0.85):
    df = pd.DataFrame(rows)
    selected = []
    for task, group in df.groupby("task", dropna=False):
        feasible = group[group["event_recall"] >= min_event_recall]
        if len(feasible):
            best = feasible.sort_values(
                ["false_alert_episodes_per_patient_hour", "consecutive_k", "threshold"],
                ascending=[True, True, False],
            ).iloc[0]
        else:
            # Fallback: maximize recall, then minimize false alert burden.
            best = group.sort_values(
                ["event_recall", "false_alert_episodes_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0]
        selected.append(best.to_dict())
    return selected


def apply_reference_policies(rows, reference_policies):
    df = pd.DataFrame(rows)
    selected = []
    locked = df["selection"].astype(str).str.startswith("fixed_").any() if len(df) else False
    for _, ref in reference_policies.iterrows():
        task = ref["task"]
        selection = normalize_selection_for_split(ref["selection"], locked=locked)
        k = int(ref["consecutive_k"])
        matches = df[
            (df["task"] == task) &
            (df["selection"] == selection) &
            (df["consecutive_k"] == k)
        ]
        if len(matches):
            selected.append(matches.iloc[0].to_dict())
            continue
        task_rows = df[df["task"] == task]
        if len(task_rows):
            fallback = task_rows.sort_values(
                ["event_recall", "false_alert_episodes_per_patient_hour"],
                ascending=[False, True],
            ).iloc[0].to_dict()
            fallback["reference_policy_missing"] = True
            fallback["requested_selection"] = selection
            fallback["requested_consecutive_k"] = k
            selected.append(fallback)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--operating-points", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--consecutive-k", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--min-event-recall", type=float, default=0.85)
    ap.add_argument("--reference-policies", default=None,
                    help="Optional alert_policy_selected.csv from validation; applies those policies without reselection.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.outputs_csv)
    ops = pd.read_csv(args.operating_points)
    keep_selections = {
        "best_f1", "best_youden", "sens_ge_0.90",
        "fixed_best_f1", "fixed_best_youden", "fixed_sens_ge_0.90",
    }
    rows = []
    for _, op in ops.iterrows():
        if op["task"] not in {"future_arrhythmia", "future_vt_vf"}:
            continue
        if op["selection"] not in keep_selections:
            continue
        if not np.isfinite(float(op["threshold"])):
            continue
        for k in args.consecutive_k:
            rows.append(
                evaluate_policy(
                    df,
                    task=op["task"],
                    source=op["probability_source"],
                    selection=op["selection"],
                    threshold=float(op["threshold"]),
                    consecutive_k=int(k),
                )
            )

    policy_df = pd.DataFrame(rows)
    if args.reference_policies:
        reference_policies = pd.read_csv(args.reference_policies)
        selected = apply_reference_policies(rows, reference_policies)
    else:
        selected = choose_policy(rows, min_event_recall=args.min_event_recall)
    policy_df.to_csv(out_dir / "alert_policy_grid.csv", index=False)
    pd.DataFrame(selected).to_csv(out_dir / "alert_policy_selected.csv", index=False)
    with (out_dir / "alert_policy_selected.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"Saved alert policy grid: {out_dir / 'alert_policy_grid.csv'}")
    for row in selected:
        print(
            f"  {row['task']}: {row['selection']} k={int(row['consecutive_k'])} "
            f"event_recall={row['event_recall']:.3f} "
            f"false_ep/hr={row['false_alert_episodes_per_patient_hour']:.2f}"
        )


if __name__ == "__main__":
    main()
