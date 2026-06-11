"""
Patient-level bootstrap confidence intervals for locked alert policies.

This script evaluates final deployment policies on saved validation/test outputs
and estimates uncertainty by resampling patient/record units with replacement.
It is post-training only: no model, dataset, threshold, or policy parameter is
changed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alert_policy import (
    contiguous_true_segments,
    event_target_for_task,
    realtime_consecutive_filter,
    score_for_task,
)
from evaluate_smoothed_alert_policy import apply_refractory, causal_smooth_by_patient


def _policy_value(policy, key, default):
    value = policy.get(key, default)
    if pd.isna(value):
        return default
    return value


def score_for_policy(df, policy):
    score = score_for_task(
        df,
        task=policy["task"],
        source=_policy_value(policy, "probability_source", "temp_scaled"),
    )
    method = str(_policy_value(policy, "smoothing_method", "raw"))
    window = int(_policy_value(policy, "smoothing_window", 1))
    alpha = float(_policy_value(policy, "smoothing_alpha", 1.0))
    return causal_smooth_by_patient(df, score, method=method, window=window, alpha=alpha)


def patient_contributions(df, policy, horizon_sec):
    task = policy["task"]
    threshold = float(policy["threshold"])
    consecutive_k = int(policy["consecutive_k"])
    refractory_windows = int(_policy_value(policy, "refractory_windows", 0))
    score = score_for_policy(df, policy)
    raw_alert = score >= threshold
    event_target = event_target_for_task(df, task)

    rows = []
    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        idx = patient_df.index.to_numpy()
        times = patient_df["timestamp_sec"].to_numpy(dtype=float)
        target = event_target[idx]
        alert = realtime_consecutive_filter(raw_alert[idx], consecutive_k)
        alert = apply_refractory(alert, refractory_windows)

        patient_hours = 0.0
        if len(times) > 1:
            patient_hours = max(float(times[-1] - times[0]), 0.0) / 3600.0

        all_events = 0
        detected_events = 0
        incident_events = 0
        detected_incident_events = 0
        lead_times = []
        incident_lead_times = []
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
                lead_times.append(float(lead))
                if incident:
                    detected_incident_events += 1
                    incident_lead_times.append(float(lead))
                covered_alert |= eligible & alert

        false_mask = alert & ~covered_alert & ~target
        false_windows = int(false_mask.sum())
        false_episodes = int(len(contiguous_true_segments(false_mask)))
        rows.append(
            {
                "patient_id": patient,
                "events": int(all_events),
                "detected_events": int(detected_events),
                "incident_events": int(incident_events),
                "detected_incident_events": int(detected_incident_events),
                "false_alert_windows": false_windows,
                "false_alert_episodes": false_episodes,
                "patient_hours": float(patient_hours),
                "lead_times": lead_times,
                "incident_lead_times": incident_lead_times,
            }
        )
    return rows


def aggregate_contributions(contribs, picked_indices=None):
    if picked_indices is None:
        picked = contribs
    else:
        picked = [contribs[int(i)] for i in picked_indices]

    events = sum(row["events"] for row in picked)
    detected_events = sum(row["detected_events"] for row in picked)
    incident_events = sum(row["incident_events"] for row in picked)
    detected_incident_events = sum(row["detected_incident_events"] for row in picked)
    false_windows = sum(row["false_alert_windows"] for row in picked)
    false_episodes = sum(row["false_alert_episodes"] for row in picked)
    patient_hours = sum(row["patient_hours"] for row in picked)
    lead_times = [x for row in picked for x in row["lead_times"]]
    incident_lead_times = [x for row in picked for x in row["incident_lead_times"]]

    return {
        "events": int(events),
        "detected_events": int(detected_events),
        "event_recall": detected_events / events if events else np.nan,
        "incident_events": int(incident_events),
        "detected_incident_events": int(detected_incident_events),
        "incident_event_recall": (
            detected_incident_events / incident_events if incident_events else np.nan
        ),
        "median_lead_time_sec": float(np.nanmedian(lead_times)) if lead_times else np.nan,
        "incident_median_lead_time_sec": (
            float(np.nanmedian(incident_lead_times)) if incident_lead_times else np.nan
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


def bootstrap_contributions(contribs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(contribs)
    samples = []
    for b in range(int(n_boot)):
        picked = rng.integers(0, n, size=n)
        row = aggregate_contributions(contribs, picked_indices=picked)
        row["bootstrap_id"] = int(b)
        samples.append(row)
    return pd.DataFrame(samples)


def ci_from_samples(samples, observed, metrics, alpha):
    rows = []
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    for metric in metrics:
        vals = samples[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "metric": metric,
                "observed": float(observed.get(metric, np.nan)),
                "ci_low": float(np.percentile(vals, lo_q)) if len(vals) else np.nan,
                "ci_high": float(np.percentile(vals, hi_q)) if len(vals) else np.nan,
                "n_finite_bootstrap": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def write_summary(out_dir, observed_df, ci_df, n_boot, horizon_sec):
    lines = [
        "# Alert Policy Bootstrap CI",
        "",
        f"- bootstrap resamples: {int(n_boot)}",
        f"- event-credit horizon: {float(horizon_sec):.0f}s",
        "- resampling unit: patient/record",
        "- policies: loaded from final validation-selected policy file",
        "",
        "## Observed Locked Metrics",
        "",
        "| split | task | event recall | incident recall | median lead | false episodes/hr | false windows/hr |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in observed_df.iterrows():
        lines.append(
            f"| {row['split']} | {row['task']} | "
            f"{row['event_recall']:.3f} | {row['incident_event_recall']:.3f} | "
            f"{row['median_lead_time_sec']:.0f}s | "
            f"{row['false_alert_episodes_per_patient_hour']:.2f} | "
            f"{row['false_alert_windows_per_patient_hour']:.2f} |"
        )
    lines.extend(["", "## 95% Patient Bootstrap CI", ""])
    for (split, task), group in ci_df.groupby(["split", "task"]):
        lines.append(f"### {split} / {task}")
        lines.append("")
        lines.append("| metric | observed | 95% CI |")
        lines.append("|---|---:|---:|")
        for _, row in group.iterrows():
            lines.append(
                f"| {row['metric']} | {row['observed']:.3f} | "
                f"{row['ci_low']:.3f}-{row['ci_high']:.3f} |"
            )
        lines.append("")
    path = out_dir / "alert_policy_bootstrap_ci.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-csv", required=True)
    ap.add_argument("--val-outputs-csv", required=True)
    ap.add_argument("--test-outputs-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon-sec", type=float, default=300.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = pd.read_csv(args.policy_csv).to_dict(orient="records")
    metrics = [
        "event_recall",
        "incident_event_recall",
        "median_lead_time_sec",
        "incident_median_lead_time_sec",
        "false_alert_windows_per_patient_hour",
        "false_alert_episodes_per_patient_hour",
    ]

    observed_rows = []
    ci_rows = []
    sample_frames = []
    patient_frames = []

    for split, outputs_csv in [
        ("validation", args.val_outputs_csv),
        ("locked_test", args.test_outputs_csv),
    ]:
        df = pd.read_csv(outputs_csv)
        for policy in policies:
            task = policy["task"]
            contribs = patient_contributions(df, policy, horizon_sec=args.horizon_sec)
            observed = aggregate_contributions(contribs)
            observed.update(
                {
                    "split": split,
                    "task": task,
                    "policy_role": policy.get("policy_role", ""),
                    "threshold": float(policy["threshold"]),
                    "consecutive_k": int(policy["consecutive_k"]),
                    "smoothing_method": _policy_value(policy, "smoothing_method", "raw"),
                    "smoothing_window": int(_policy_value(policy, "smoothing_window", 1)),
                    "smoothing_alpha": float(_policy_value(policy, "smoothing_alpha", 1.0)),
                    "refractory_windows": int(_policy_value(policy, "refractory_windows", 0)),
                    "n_patients": int(len(contribs)),
                }
            )
            observed_rows.append(observed)

            samples = bootstrap_contributions(contribs, n_boot=args.bootstrap, seed=args.seed)
            samples.insert(0, "task", task)
            samples.insert(0, "split", split)
            sample_frames.append(samples)

            ci = ci_from_samples(samples, observed, metrics=metrics, alpha=args.alpha)
            ci.insert(0, "task", task)
            ci.insert(0, "split", split)
            ci_rows.append(ci)

            patient_df = pd.DataFrame(
                [{k: v for k, v in row.items() if k not in {"lead_times", "incident_lead_times"}}
                 for row in contribs]
            )
            patient_df.insert(0, "task", task)
            patient_df.insert(0, "split", split)
            patient_frames.append(patient_df)

    observed_df = pd.DataFrame(observed_rows)
    ci_df = pd.concat(ci_rows, ignore_index=True)
    samples_df = pd.concat(sample_frames, ignore_index=True)
    patient_df = pd.concat(patient_frames, ignore_index=True)

    observed_df.to_csv(out_dir / "alert_policy_observed_metrics.csv", index=False)
    ci_df.to_csv(out_dir / "alert_policy_patient_bootstrap_ci.csv", index=False)
    samples_df.to_csv(out_dir / "alert_policy_patient_bootstrap_samples.csv", index=False)
    patient_df.to_csv(out_dir / "alert_policy_patient_contributions.csv", index=False)

    with (out_dir / "alert_policy_patient_bootstrap_ci.json").open("w", encoding="utf-8") as f:
        json.dump(ci_df.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    summary_path = write_summary(
        out_dir,
        observed_df=observed_df,
        ci_df=ci_df,
        n_boot=args.bootstrap,
        horizon_sec=args.horizon_sec,
    )
    print(f"Saved observed metrics: {out_dir / 'alert_policy_observed_metrics.csv'}")
    print(f"Saved bootstrap CI: {out_dir / 'alert_policy_patient_bootstrap_ci.csv'}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
