"""
Evaluate event detection sensitivity at multiple allowed lead-time horizons.

The current V7 model is trained with a 5-minute future target. This script does
not rebuild labels or retrain; it asks how the locked alert policy performs when
an alert is only credited if it occurs within 1/3/5/... minutes before event
onset. It supports both raw consecutive policies and smoothed policies.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alert_policy import (
    contiguous_true_segments,
    event_target_for_task,
    realtime_consecutive_filter,
    score_for_task,
)
from evaluate_smoothed_alert_policy import (
    apply_refractory,
    causal_smooth_by_patient,
)


def policy_score(df, policy):
    source = policy.get("probability_source", "temp_scaled")
    task = policy["task"]
    score = score_for_task(df, task, source)
    method = policy.get("smoothing_method", "raw")
    window = int(policy.get("smoothing_window", 1))
    alpha = float(policy.get("smoothing_alpha", 1.0))
    return causal_smooth_by_patient(df, score, method=method, window=window, alpha=alpha)


def evaluate_policy_at_horizon(df, policy, horizon_sec):
    task = policy["task"]
    score = policy_score(df, policy)
    target_all = event_target_for_task(df, task)
    raw_alert = score >= float(policy["threshold"])
    consecutive_k = int(policy["consecutive_k"])
    refractory_windows = int(policy.get("refractory_windows", 0))

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
        target = target_all[idx]
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
            eligible = (times >= start_t - float(horizon_sec)) & (times <= end_t)
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
        "policy_label": policy.get("policy_label", "policy"),
        "horizon_sec": float(horizon_sec),
        "horizon_min": float(horizon_sec) / 60.0,
        "probability_source": policy.get("probability_source", "temp_scaled"),
        "threshold": float(policy["threshold"]),
        "consecutive_k": consecutive_k,
        "smoothing_method": policy.get("smoothing_method", "raw"),
        "smoothing_window": int(policy.get("smoothing_window", 1)),
        "smoothing_alpha": float(policy.get("smoothing_alpha", 1.0)),
        "refractory_windows": refractory_windows,
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


def load_policy_csv(path, label):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        d["policy_label"] = label
        if "smoothing_method" not in d or pd.isna(d.get("smoothing_method")):
            d["smoothing_method"] = "raw"
            d["smoothing_window"] = 1
            d["smoothing_alpha"] = 1.0
            d["refractory_windows"] = 0
        rows.append(d)
    return rows


def write_markdown(summary_df, out_dir):
    lines = [
        "# Lead-Horizon Sensitivity",
        "",
        "This is a post-hoc alert-policy analysis on saved V7 outputs. The model label horizon remains 5 minutes; rows below vary the allowed event-credit horizon.",
        "",
    ]
    for split in sorted(summary_df["split"].unique()):
        lines.append(f"## {split}")
        for task in sorted(summary_df["task"].unique()):
            sub = summary_df[(summary_df["split"] == split) & (summary_df["task"] == task)]
            if sub.empty:
                continue
            lines.append("")
            lines.append(f"### {task}")
            lines.append("")
            lines.append("| policy | horizon_min | event_recall | incident_recall | median_lead_sec | false_ep/hr | false_windows/hr |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, row in sub.sort_values(["policy_label", "horizon_min"]).iterrows():
                lines.append(
                    f"| {row['policy_label']} | {row['horizon_min']:.1f} | "
                    f"{row['event_recall']:.3f} | {row['incident_event_recall']:.3f} | "
                    f"{row['median_lead_time_sec']:.0f} | "
                    f"{row['false_alert_episodes_per_patient_hour']:.2f} | "
                    f"{row['false_alert_windows_per_patient_hour']:.2f} |"
                )
            lines.append("")
    path = out_dir / "lead_horizon_sensitivity.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-outputs-csv", required=True)
    ap.add_argument("--test-outputs-csv", required=True)
    ap.add_argument("--baseline-policy-csv", required=True)
    ap.add_argument("--smoothed-policy-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--horizons-sec", default="60,180,300,600")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [float(x) for x in args.horizons_sec.split(",") if x.strip()]
    policies = []
    policies.extend(load_policy_csv(args.baseline_policy_csv, "baseline_consecutive"))
    policies.extend(load_policy_csv(args.smoothed_policy_csv, "smoothed_candidate"))

    rows = []
    for split, outputs_csv in [
        ("validation", args.val_outputs_csv),
        ("locked_test", args.test_outputs_csv),
    ]:
        df = pd.read_csv(outputs_csv)
        for policy in policies:
            for horizon in horizons:
                row = evaluate_policy_at_horizon(df, policy, horizon)
                row["split"] = split
                rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / "lead_horizon_sensitivity.csv"
    summary_df.to_csv(csv_path, index=False)
    md_path = write_markdown(summary_df, out_dir)
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
