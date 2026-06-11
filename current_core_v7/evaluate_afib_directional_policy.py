"""
Evaluate a validation-only AFib directional warning policy.

The accepted overall arrhythmia alarm is not changed. This script only studies
whether an auxiliary AFib direction tag can improve on future-class argmax
without an unacceptable increase in AFib-specific false alert episodes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from evaluate_alert_policy import contiguous_true_segments, realtime_consecutive_filter
from evaluate_smoothed_alert_policy import causal_smooth_by_patient, infer_horizon_sec


AFIB_CLASS = 2
DEFAULT_THRESHOLDS = (
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.050,
    0.060,
    0.080,
    0.100,
    0.120,
    0.150,
    0.200,
    0.300,
)
DEFAULT_PAIRWISE_THRESHOLDS = (
    0.050,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
    0.400,
    0.500,
)
KEY_COLUMNS = (
    "patient_id",
    "window_id",
    "target_cur",
    "target_fut",
    "target_fut_soft_afib",
)
REQUIRED_COLUMNS = (
    *KEY_COLUMNS,
    "timestamp_sec",
    "future_start_sec",
    "future_end_sec",
    "pred_fut",
    "prob_fut_afib",
    "transition_flag",
    "rr_rmssd_norm",
    "rr_cv",
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def validate_inputs(outputs_csv, member_csvs):
    df = pd.read_csv(outputs_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"outputs CSV is missing required columns: {missing}")
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError("outputs CSV must load with a contiguous RangeIndex")
    if df["patient_id"].isna().any() or df["window_id"].isna().any():
        raise ValueError("patient_id and window_id must not contain missing values")
    if df.duplicated(["patient_id", "window_id"]).any():
        raise ValueError("patient_id/window_id pairs must be unique")

    members = []
    for member_csv in member_csvs:
        member = pd.read_csv(member_csv)
        if len(member) != len(df):
            raise ValueError(
                f"member row count mismatch: {member_csv} has {len(member)}, "
                f"expected {len(df)}"
            )
        member_missing = [
            column
            for column in (*KEY_COLUMNS, "pred_fut", "prob_fut_afib")
            if column not in member.columns
        ]
        if member_missing:
            raise ValueError(f"{member_csv} is missing columns: {member_missing}")
        for column in KEY_COLUMNS:
            left = df[column].to_numpy()
            right = member[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                aligned = np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True)
            else:
                aligned = np.array_equal(left, right)
            if not aligned:
                raise ValueError(f"member alignment failed for {member_csv}: {column}")
        members.append(member)

    member_probs = np.stack(
        [member["prob_fut_afib"].to_numpy(dtype=float) for member in members],
        axis=1,
    )
    member_votes = np.stack(
        [member["pred_fut"].to_numpy(dtype=int) == AFIB_CLASS for member in members],
        axis=1,
    )
    df = df.copy()
    df["afib_member_prob_std"] = np.std(member_probs, axis=1)
    df["afib_member_vote_count"] = np.sum(member_votes, axis=1)
    return df


def classification_metrics(y_true, score, alert):
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        alert,
        average="binary",
        zero_division=0,
    )
    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, score)
        ap = average_precision_score(y_true, score)
    else:
        auroc = np.nan
        ap = np.nan
    return {
        "support": int(y_true.sum()),
        "predicted_positive": int(alert.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "average_precision": float(ap),
    }


def build_alert(df, score, threshold=None, smoothing_method="raw", consecutive_k=1):
    score = np.asarray(score, dtype=float)
    if threshold is None:
        raw_alert = df["pred_fut"].to_numpy(dtype=int) == AFIB_CLASS
    else:
        smoothed = causal_smooth_by_patient(
            df,
            score,
            method=smoothing_method,
            window=1,
            alpha=0.65,
        )
        raw_alert = smoothed >= float(threshold)

    alert = np.zeros(len(df), dtype=bool)
    for patient in sorted(df["patient_id"].unique()):
        idx = (
            df[df["patient_id"] == patient]
            .sort_values("timestamp_sec")
            .index.to_numpy()
        )
        alert[idx] = realtime_consecutive_filter(raw_alert[idx], int(consecutive_k))
    return alert


def patient_event_contributions(df, alert, horizon_sec):
    rows = []
    current_afib = df["target_cur"].to_numpy(dtype=int) == AFIB_CLASS
    alert = np.asarray(alert, dtype=bool)

    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        idx = patient_df.index.to_numpy()
        times = patient_df["timestamp_sec"].to_numpy(dtype=float)
        target = current_afib[idx]
        patient_alert = alert[idx]

        patient_hours = 0.0
        if len(times) > 1:
            patient_hours = max(float(times[-1] - times[0]), 0.0) / 3600.0

        events = 0
        detected_events = 0
        incident_events = 0
        detected_incident_events = 0
        lead_times = []
        incident_lead_times = []
        covered_alert = np.zeros(len(idx), dtype=bool)

        for start_i, end_i in contiguous_true_segments(target):
            events += 1
            incident = start_i > 0
            incident_events += int(incident)
            start_t = float(times[start_i])
            end_t = float(times[end_i])
            eligible = (times >= start_t - float(horizon_sec)) & (times <= end_t)
            event_alert_idx = np.where(eligible & patient_alert)[0]
            if len(event_alert_idx):
                detected_events += 1
                detected_incident_events += int(incident)
                lead = start_t - float(times[event_alert_idx[0]])
                lead_times.append(float(lead))
                if incident:
                    incident_lead_times.append(float(lead))
                covered_alert |= eligible & patient_alert

        false_mask = patient_alert & ~covered_alert & ~target
        rows.append(
            {
                "patient_id": patient,
                "events": int(events),
                "detected_events": int(detected_events),
                "incident_events": int(incident_events),
                "detected_incident_events": int(detected_incident_events),
                "false_alert_windows": int(false_mask.sum()),
                "false_alert_episodes": int(len(contiguous_true_segments(false_mask))),
                "patient_hours": float(patient_hours),
                "lead_times": lead_times,
                "incident_lead_times": incident_lead_times,
            }
        )
    return rows


def aggregate_contributions(contributions, picked_indices=None):
    if picked_indices is None:
        picked = contributions
    else:
        picked = [contributions[int(index)] for index in picked_indices]

    events = sum(row["events"] for row in picked)
    detected_events = sum(row["detected_events"] for row in picked)
    incident_events = sum(row["incident_events"] for row in picked)
    detected_incident_events = sum(row["detected_incident_events"] for row in picked)
    false_windows = sum(row["false_alert_windows"] for row in picked)
    false_episodes = sum(row["false_alert_episodes"] for row in picked)
    patient_hours = sum(row["patient_hours"] for row in picked)
    lead_times = [value for row in picked for value in row["lead_times"]]
    incident_lead_times = [
        value for row in picked for value in row["incident_lead_times"]
    ]
    return {
        "events": int(events),
        "detected_events": int(detected_events),
        "event_recall": safe_div(detected_events, events),
        "incident_events": int(incident_events),
        "detected_incident_events": int(detected_incident_events),
        "incident_event_recall": safe_div(
            detected_incident_events,
            incident_events,
        ),
        "median_lead_time_sec": (
            float(np.nanmedian(lead_times)) if lead_times else np.nan
        ),
        "incident_median_lead_time_sec": (
            float(np.nanmedian(incident_lead_times))
            if incident_lead_times
            else np.nan
        ),
        "false_alert_windows": int(false_windows),
        "false_alert_episodes": int(false_episodes),
        "patient_hours": float(patient_hours),
        "false_alert_windows_per_patient_hour": safe_div(
            false_windows,
            patient_hours,
        ),
        "false_alert_episodes_per_patient_hour": safe_div(
            false_episodes,
            patient_hours,
        ),
    }


def evaluate_policy(df, score, alert, horizon_sec):
    dominant = df["target_fut"].to_numpy(dtype=int) == AFIB_CLASS
    any_mass = df["target_fut_soft_afib"].to_numpy(dtype=float) > 0.0
    row = {}
    for prefix, label in (("dominant", dominant), ("any_mass", any_mass)):
        metrics = classification_metrics(label, score, alert)
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    contributions = patient_event_contributions(df, alert, horizon_sec=horizon_sec)
    row.update(aggregate_contributions(contributions))
    return row, contributions


def make_threshold_sweep(df, scores, threshold_sets, horizon_sec):
    rows = []
    baseline_score = scores["afib_probability"]
    baseline_alert = build_alert(df, baseline_score, threshold=None)
    baseline_metrics, _ = evaluate_policy(
        df,
        baseline_score,
        baseline_alert,
        horizon_sec=horizon_sec,
    )
    rows.append(
        {
            "policy_name": "future_class_argmax",
            "policy_kind": "argmax",
            "score_name": "afib_probability",
            "threshold": np.nan,
            "smoothing_method": "raw",
            "smoothing_alpha": 1.0,
            "consecutive_k": 1,
            **baseline_metrics,
        }
    )

    for score_name, score in scores.items():
        for smoothing_method in ("raw", "ewma"):
            for consecutive_k in (1, 2):
                for threshold in threshold_sets[score_name]:
                    alert = build_alert(
                        df,
                        score,
                        threshold=threshold,
                        smoothing_method=smoothing_method,
                        consecutive_k=consecutive_k,
                    )
                    metrics, _ = evaluate_policy(
                        df,
                        score,
                        alert,
                        horizon_sec=horizon_sec,
                    )
                    rows.append(
                        {
                            "policy_name": (
                                f"{score_name}_{smoothing_method}_k{consecutive_k}_"
                                f"t{threshold:.3f}"
                            ),
                            "policy_kind": "probability_threshold",
                            "score_name": score_name,
                            "threshold": float(threshold),
                            "smoothing_method": smoothing_method,
                            "smoothing_alpha": (
                                0.65 if smoothing_method == "ewma" else 1.0
                            ),
                            "consecutive_k": int(consecutive_k),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def candidate_gate(row, baseline):
    recall_gain = float(row["dominant_recall"] - baseline["dominant_recall"])
    event_gain = float(row["event_recall"] - baseline["event_recall"])
    baseline_false = float(baseline["false_alert_episodes_per_patient_hour"])
    false_limit = baseline_false * 1.20
    checks = {
        "recall_or_event_gain": bool(recall_gain >= 0.08 or event_gain >= 0.05),
        "false_episode_limit": bool(
            row["false_alert_episodes_per_patient_hour"] <= false_limit + 1e-12
        ),
        "median_lead_time": bool(row["median_lead_time_sec"] >= 240.0),
        "overall_alarm_unchanged": True,
    }
    return checks, recall_gain, event_gain, false_limit


def select_candidate(sweep):
    baseline = sweep[sweep["policy_kind"] == "argmax"].iloc[0]
    candidates = sweep[sweep["policy_kind"] == "probability_threshold"].copy()
    gate_columns = {
        "gate_recall_or_event_gain": [],
        "gate_false_episode_limit": [],
        "gate_median_lead_time": [],
        "dominant_recall_gain": [],
        "event_recall_gain": [],
        "false_episode_limit": [],
    }
    for _, row in candidates.iterrows():
        checks, recall_gain, event_gain, false_limit = candidate_gate(row, baseline)
        gate_columns["gate_recall_or_event_gain"].append(
            checks["recall_or_event_gain"]
        )
        gate_columns["gate_false_episode_limit"].append(
            checks["false_episode_limit"]
        )
        gate_columns["gate_median_lead_time"].append(checks["median_lead_time"])
        gate_columns["dominant_recall_gain"].append(recall_gain)
        gate_columns["event_recall_gain"].append(event_gain)
        gate_columns["false_episode_limit"].append(false_limit)
    for column, values in gate_columns.items():
        candidates[column] = values
    candidates["passes_full_gate"] = (
        candidates["gate_recall_or_event_gain"]
        & candidates["gate_false_episode_limit"]
        & candidates["gate_median_lead_time"]
    )
    sweep = sweep.merge(
        candidates[
            [
                "policy_name",
                *gate_columns.keys(),
                "passes_full_gate",
            ]
        ],
        on="policy_name",
        how="left",
    )

    feasible = candidates[candidates["passes_full_gate"]]
    if not len(feasible):
        return sweep, None
    selected = feasible.sort_values(
        [
            "dominant_f1",
            "event_recall",
            "false_alert_episodes_per_patient_hour",
            "threshold",
        ],
        ascending=[False, False, True, False],
    ).iloc[0]
    return sweep, selected


def bootstrap_policy(df, score, alert, contributions, n_boot, seed, policy_name):
    rng = np.random.default_rng(seed)
    patients = [row["patient_id"] for row in contributions]
    patient_indices = {
        patient: df.index[df["patient_id"] == patient].to_numpy()
        for patient in patients
    }
    dominant = df["target_fut"].to_numpy(dtype=int) == AFIB_CLASS
    any_mass = df["target_fut_soft_afib"].to_numpy(dtype=float) > 0.0
    rows = []
    for bootstrap_id in range(int(n_boot)):
        picked = rng.integers(0, len(patients), size=len(patients))
        event_metrics = aggregate_contributions(contributions, picked_indices=picked)
        row_indices = np.concatenate(
            [patient_indices[patients[int(index)]] for index in picked]
        )
        row = {
            "policy_name": policy_name,
            "bootstrap_id": int(bootstrap_id),
            **event_metrics,
        }
        for prefix, label in (("dominant", dominant), ("any_mass", any_mass)):
            metrics = classification_metrics(
                label[row_indices],
                score[row_indices],
                alert[row_indices],
            )
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(samples, observed_rows):
    metrics = (
        "dominant_auroc",
        "dominant_average_precision",
        "dominant_precision",
        "dominant_recall",
        "dominant_f1",
        "any_mass_auroc",
        "any_mass_average_precision",
        "any_mass_precision",
        "any_mass_recall",
        "any_mass_f1",
        "event_recall",
        "incident_event_recall",
        "median_lead_time_sec",
        "incident_median_lead_time_sec",
        "false_alert_windows_per_patient_hour",
        "false_alert_episodes_per_patient_hour",
    )
    rows = []
    observed = observed_rows.set_index("policy_name")
    for policy_name, group in samples.groupby("policy_name", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "policy_name": policy_name,
                    "metric": metric,
                    "observed": float(observed.loc[policy_name, metric]),
                    "ci_low": (
                        float(np.percentile(values, 2.5)) if len(values) else np.nan
                    ),
                    "ci_high": (
                        float(np.percentile(values, 97.5)) if len(values) else np.nan
                    ),
                    "n_finite_bootstrap": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def leave_one_positive_record_out(df, score, baseline_alert, selected_alert):
    positive_patients = sorted(
        df.loc[df["target_fut"] == AFIB_CLASS, "patient_id"].unique()
    )
    rows = []
    for held_out in positive_patients:
        keep = df["patient_id"] != held_out
        subset = df.loc[keep].copy().reset_index(drop=True)
        subset_score = score[keep.to_numpy()]
        subset_baseline_alert = baseline_alert[keep.to_numpy()]
        subset_selected_alert = selected_alert[keep.to_numpy()]
        baseline_metrics, _ = evaluate_policy(
            subset,
            subset_score,
            subset_baseline_alert,
            horizon_sec=infer_horizon_sec(subset),
        )
        selected_metrics, _ = evaluate_policy(
            subset,
            subset_score,
            subset_selected_alert,
            horizon_sec=infer_horizon_sec(subset),
        )
        checks, recall_gain, event_gain, false_limit = candidate_gate(
            pd.Series(selected_metrics),
            pd.Series(baseline_metrics),
        )
        rows.append(
            {
                "held_out_patient_id": held_out,
                "remaining_positive_records": int(
                    subset.loc[
                        subset["target_fut"] == AFIB_CLASS,
                        "patient_id",
                    ].nunique()
                ),
                "baseline_dominant_recall": baseline_metrics["dominant_recall"],
                "selected_dominant_recall": selected_metrics["dominant_recall"],
                "dominant_recall_gain": recall_gain,
                "baseline_event_recall": baseline_metrics["event_recall"],
                "selected_event_recall": selected_metrics["event_recall"],
                "event_recall_gain": event_gain,
                "baseline_false_episodes_per_hour": baseline_metrics[
                    "false_alert_episodes_per_patient_hour"
                ],
                "selected_false_episodes_per_hour": selected_metrics[
                    "false_alert_episodes_per_patient_hour"
                ],
                "false_episode_limit": false_limit,
                "selected_median_lead_time_sec": selected_metrics[
                    "median_lead_time_sec"
                ],
                **{f"gate_{key}": value for key, value in checks.items()},
                "passes_full_gate": bool(all(checks.values())),
            }
        )
    return pd.DataFrame(rows)


def audit_candidate_stability(df, scores, baseline_alert, sweep):
    feasible = sweep[sweep["passes_full_gate"] == True].copy()  # noqa: E712
    stability_rows = []
    loo_by_policy = {}
    for _, candidate in feasible.iterrows():
        score = scores[str(candidate["score_name"])]
        alert = build_alert(
            df,
            score,
            threshold=float(candidate["threshold"]),
            smoothing_method=str(candidate["smoothing_method"]),
            consecutive_k=int(candidate["consecutive_k"]),
        )
        loo = leave_one_positive_record_out(df, score, baseline_alert, alert)
        policy_name = str(candidate["policy_name"])
        loo_by_policy[policy_name] = loo
        stability_rows.append(
            {
                "policy_name": policy_name,
                "score_name": str(candidate["score_name"]),
                "dominant_f1": float(candidate["dominant_f1"]),
                "dominant_recall": float(candidate["dominant_recall"]),
                "event_recall": float(candidate["event_recall"]),
                "median_lead_time_sec": float(candidate["median_lead_time_sec"]),
                "false_alert_episodes_per_patient_hour": float(
                    candidate["false_alert_episodes_per_patient_hour"]
                ),
                "positive_record_loo_passes": int(loo["passes_full_gate"].sum()),
                "positive_record_loo_total": int(len(loo)),
                "positive_record_loo_pass_rate": safe_div(
                    int(loo["passes_full_gate"].sum()),
                    len(loo),
                ),
                "minimum_dominant_recall_gain": float(
                    loo["dominant_recall_gain"].min()
                ),
                "minimum_event_recall_gain": float(loo["event_recall_gain"].min()),
                "minimum_median_lead_time_sec": float(
                    loo["selected_median_lead_time_sec"].min()
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)
    if len(stability):
        stability = stability.sort_values(
            [
                "positive_record_loo_pass_rate",
                "dominant_f1",
                "false_alert_episodes_per_patient_hour",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return stability, loo_by_policy


def quantile_bucket(series, labels):
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.nunique() < 2:
        return pd.Series(["single_bin"] * len(series), index=series.index)
    ranks = numeric.rank(method="average", pct=True)
    edges = np.linspace(0.0, 1.0, len(labels) + 1)
    return pd.cut(
        ranks,
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)


def future_run_duration(df):
    durations = np.zeros(len(df), dtype=float)
    for patient in sorted(df["patient_id"].unique()):
        patient_df = df[df["patient_id"] == patient].sort_values("timestamp_sec")
        idx = patient_df.index.to_numpy()
        times = patient_df["timestamp_sec"].to_numpy(dtype=float)
        target = patient_df["target_fut"].to_numpy(dtype=int) == AFIB_CLASS
        positive_steps = np.diff(times)
        positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
        step = float(np.median(positive_steps)) if len(positive_steps) else 10.0
        for start_i, end_i in contiguous_true_segments(target):
            durations[idx[start_i : end_i + 1]] = (
                float(times[end_i] - times[start_i]) + step
            )
    return durations


def error_buckets(df, baseline_alert, selected_alert):
    dominant = df["target_fut"].to_numpy(dtype=int) == AFIB_CLASS
    positive = df.loc[dominant].copy()
    positive["baseline_detected"] = baseline_alert[dominant]
    positive["selected_detected"] = selected_alert[dominant]
    positive["recovered_by_selected"] = (
        ~positive["baseline_detected"] & positive["selected_detected"]
    )
    positive["future_mass_bin"] = pd.cut(
        positive["target_fut_soft_afib"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.75", ">0.75"],
    ).astype(str)
    positive["future_run_duration_sec"] = future_run_duration(df)[dominant]
    positive["future_run_duration_bin"] = pd.cut(
        positive["future_run_duration_sec"],
        bins=[-np.inf, 60.0, 180.0, 300.0, np.inf],
        labels=["<=60s", "60-180s", "180-300s", ">300s"],
    ).astype(str)
    positive["rr_cv_quartile"] = quantile_bucket(
        positive["rr_cv"],
        ["Q1", "Q2", "Q3", "Q4"],
    )
    positive["rr_rmssd_quartile"] = quantile_bucket(
        positive["rr_rmssd_norm"],
        ["Q1", "Q2", "Q3", "Q4"],
    )
    positive["disagreement_quartile"] = quantile_bucket(
        positive["afib_member_prob_std"],
        ["Q1", "Q2", "Q3", "Q4"],
    )
    positive["transition_state"] = np.where(
        positive["transition_flag"].astype(bool),
        "transition",
        "stable_label",
    )
    positive["current_state"] = np.where(
        positive["target_cur"] == AFIB_CLASS,
        "current_afib",
        "not_current_afib",
    )

    rows = []
    dimensions = (
        "patient_id",
        "future_mass_bin",
        "future_run_duration_bin",
        "rr_cv_quartile",
        "rr_rmssd_quartile",
        "disagreement_quartile",
        "transition_state",
        "current_state",
    )
    for dimension in dimensions:
        for category, group in positive.groupby(dimension, dropna=False, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "category": str(category),
                    "positive_windows": int(len(group)),
                    "baseline_detected": int(group["baseline_detected"].sum()),
                    "baseline_missed": int((~group["baseline_detected"]).sum()),
                    "baseline_recall": float(group["baseline_detected"].mean()),
                    "selected_detected": int(group["selected_detected"].sum()),
                    "selected_missed": int((~group["selected_detected"]).sum()),
                    "selected_recall": float(group["selected_detected"].mean()),
                    "recovered_by_selected": int(group["recovered_by_selected"].sum()),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    output_dir,
    df,
    baseline,
    selected,
    ci,
    loo,
    decision,
    horizon_sec,
):
    def ci_text(policy_name, metric):
        row = ci[
            (ci["policy_name"] == policy_name)
            & (ci["metric"] == metric)
        ].iloc[0]
        return f"{row['ci_low']:.3f}-{row['ci_high']:.3f}"

    selected_name = selected["policy_name"]
    lines = [
        "# AFib Validation Baseline and Directional Policy",
        "",
        "This analysis uses validation data only. The accepted overall arrhythmia "
        "alarm is unchanged.",
        "",
        "## Support and Audit",
        "",
        f"- rows: {len(df)}",
        f"- patient/record units: {df['patient_id'].nunique()}",
        f"- current AFib windows: {(df['target_cur'] == AFIB_CLASS).sum()} "
        f"from {df.loc[df['target_cur'] == AFIB_CLASS, 'patient_id'].nunique()} records",
        f"- dominant future AFib windows: {(df['target_fut'] == AFIB_CLASS).sum()} "
        f"from {df.loc[df['target_fut'] == AFIB_CLASS, 'patient_id'].nunique()} records",
        f"- any future AFib mass windows: {(df['target_fut_soft_afib'] > 0).sum()} "
        f"from {df.loc[df['target_fut_soft_afib'] > 0, 'patient_id'].nunique()} records",
        f"- event-credit horizon: {horizon_sec:.0f}s",
        "- ensemble member row and label alignment: exact",
        "",
        "## Observed Results",
        "",
        "| policy | dominant precision | dominant recall | dominant F1 | "
        "event recall | median lead | false episodes/hr |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| future-class argmax | {baseline['dominant_precision']:.3f} | "
        f"{baseline['dominant_recall']:.3f} | {baseline['dominant_f1']:.3f} | "
        f"{baseline['event_recall']:.3f} | {baseline['median_lead_time_sec']:.0f}s | "
        f"{baseline['false_alert_episodes_per_patient_hour']:.2f} |",
        f"| {selected_name} | {selected['dominant_precision']:.3f} | "
        f"{selected['dominant_recall']:.3f} | {selected['dominant_f1']:.3f} | "
        f"{selected['event_recall']:.3f} | {selected['median_lead_time_sec']:.0f}s | "
        f"{selected['false_alert_episodes_per_patient_hour']:.2f} |",
        "",
        "## Patient Bootstrap",
        "",
        f"- argmax dominant recall 95% CI: "
        f"{ci_text('future_class_argmax', 'dominant_recall')}",
        f"- selected dominant recall 95% CI: "
        f"{ci_text(selected_name, 'dominant_recall')}",
        f"- argmax event recall 95% CI: "
        f"{ci_text('future_class_argmax', 'event_recall')}",
        f"- selected event recall 95% CI: "
        f"{ci_text(selected_name, 'event_recall')}",
        "",
        "## Stability and Decision",
        "",
        f"- positive-record leave-one-out gate pass: "
        f"{int(loo['passes_full_gate'].sum())}/{len(loo)}",
        f"- decision: `{decision['status']}`",
        f"- permitted scope: `{decision['scope']}`",
        f"- reason: {decision['reason']}",
        "",
        "The candidate is not a replacement for the Round5 overall arrhythmia "
        "alarm. External AFib validation remains required before any deployment "
        "claim.",
    ]
    (output_dir / "AFIB_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-csv", required=True)
    parser.add_argument("--member-csvs", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-sec", type=float, default=None)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--pairwise-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_PAIRWISE_THRESHOLDS),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = validate_inputs(args.outputs_csv, args.member_csvs)
    horizon_sec = infer_horizon_sec(df, override=args.horizon_sec)
    afib_probability = df["prob_fut_afib"].to_numpy(dtype=float)
    normal_probability = df["prob_fut_normal"].to_numpy(dtype=float)
    scores = {
        "afib_probability": afib_probability,
        "afib_vs_normal": afib_probability
        / np.maximum(afib_probability + normal_probability, 1e-8),
    }
    threshold_sets = {
        "afib_probability": sorted(set(args.thresholds)),
        "afib_vs_normal": sorted(set(args.pairwise_thresholds)),
    }

    sweep = make_threshold_sweep(
        df,
        scores,
        threshold_sets=threshold_sets,
        horizon_sec=horizon_sec,
    )
    sweep, selected = select_candidate(sweep)
    if selected is None:
        sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
        decision = {
            "status": "not_promoted",
            "scope": "none",
            "reason": "No simple AFib probability policy passed all validation gates.",
        }
        (output_dir / "decision.json").write_text(
            json.dumps(decision, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(decision["reason"])

    baseline = sweep[sweep["policy_kind"] == "argmax"].iloc[0]
    baseline_score = scores["afib_probability"]
    baseline_alert = build_alert(df, baseline_score, threshold=None)
    stability, loo_by_policy = audit_candidate_stability(
        df,
        scores,
        baseline_alert,
        sweep,
    )
    stable_policy_names = set(
        stability.loc[
            stability["positive_record_loo_pass_rate"] >= 0.80,
            "policy_name",
        ]
    )
    if stable_policy_names:
        stable_candidates = sweep[sweep["policy_name"].isin(stable_policy_names)]
        selected = stable_candidates.sort_values(
            [
                "dominant_f1",
                "event_recall",
                "false_alert_episodes_per_patient_hour",
                "threshold",
            ],
            ascending=[False, False, True, False],
        ).iloc[0]

    selected_score = scores[str(selected["score_name"])]
    selected_alert = build_alert(
        df,
        selected_score,
        threshold=float(selected["threshold"]),
        smoothing_method=str(selected["smoothing_method"]),
        consecutive_k=int(selected["consecutive_k"]),
    )
    baseline_metrics, baseline_contributions = evaluate_policy(
        df,
        baseline_score,
        baseline_alert,
        horizon_sec=horizon_sec,
    )
    selected_metrics, selected_contributions = evaluate_policy(
        df,
        selected_score,
        selected_alert,
        horizon_sec=horizon_sec,
    )
    baseline_observed = {
        "policy_name": "future_class_argmax",
        "policy_kind": "argmax",
        "score_name": "afib_probability",
        "threshold": np.nan,
        "smoothing_method": "raw",
        "smoothing_alpha": 1.0,
        "consecutive_k": 1,
        **baseline_metrics,
    }
    selected_observed = {
        "policy_name": str(selected["policy_name"]),
        "policy_kind": "probability_threshold",
        "score_name": str(selected["score_name"]),
        "threshold": float(selected["threshold"]),
        "smoothing_method": str(selected["smoothing_method"]),
        "smoothing_alpha": float(selected["smoothing_alpha"]),
        "consecutive_k": int(selected["consecutive_k"]),
        **selected_metrics,
    }
    observed = pd.DataFrame([baseline_observed, selected_observed])

    bootstrap_frames = [
        bootstrap_policy(
            df,
            baseline_score,
            baseline_alert,
            baseline_contributions,
            n_boot=args.bootstrap,
            seed=args.seed,
            policy_name="future_class_argmax",
        ),
        bootstrap_policy(
            df,
            selected_score,
            selected_alert,
            selected_contributions,
            n_boot=args.bootstrap,
            seed=args.seed + 1,
            policy_name=selected_observed["policy_name"],
        ),
    ]
    bootstrap_samples = pd.concat(bootstrap_frames, ignore_index=True)
    ci = bootstrap_ci(bootstrap_samples, observed)
    loo = loo_by_policy[str(selected["policy_name"])]
    loo_pass_rate = safe_div(int(loo["passes_full_gate"].sum()), len(loo))

    positive_records = int(
        df.loc[df["target_fut"] == AFIB_CLASS, "patient_id"].nunique()
    )
    if positive_records < 5:
        status = "not_promoted"
        scope = "none"
        reason = (
            "Fewer than five AFib-positive validation records are available; "
            "the candidate is too weakly supported."
        )
    elif loo_pass_rate < 0.80:
        status = "not_promoted"
        scope = "none"
        reason = (
            "No full-gate candidate remained stable in at least 80% of positive-"
            "record leave-one-out influence checks."
        )
    else:
        status = "validation_candidate"
        scope = "external_evaluation_only"
        reason = (
            "The simple policy passed the full validation gate and at least 80% "
            "of positive-record leave-one-out influence checks, but only five "
            "positive records are available."
        )
    decision = {
        "status": status,
        "scope": scope,
        "reason": reason,
        "official_round5_alarm_changed": False,
        "positive_validation_records": positive_records,
        "leave_one_positive_record_out_pass_rate": loo_pass_rate,
        "selected_policy": {
            "score": selected_observed["score_name"],
            "score_formula": (
                "prob_fut_afib / max(prob_fut_afib + prob_fut_normal, 1e-8)"
                if selected_observed["score_name"] == "afib_vs_normal"
                else "prob_fut_afib"
            ),
            "threshold": selected_observed["threshold"],
            "smoothing_method": selected_observed["smoothing_method"],
            "smoothing_alpha": selected_observed["smoothing_alpha"],
            "consecutive_k": selected_observed["consecutive_k"],
            "label": "AFib risk direction",
        },
    }

    contributions = []
    for policy_name, policy_contributions in (
        ("future_class_argmax", baseline_contributions),
        (selected_observed["policy_name"], selected_contributions),
    ):
        for row in policy_contributions:
            contributions.append(
                {
                    "policy_name": policy_name,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"lead_times", "incident_lead_times"}
                    },
                }
            )
    contribution_df = pd.DataFrame(contributions)
    buckets = error_buckets(df, baseline_alert, selected_alert)

    support = {
        "rows": int(len(df)),
        "patient_records": int(df["patient_id"].nunique()),
        "current_afib_windows": int((df["target_cur"] == AFIB_CLASS).sum()),
        "current_afib_records": int(
            df.loc[df["target_cur"] == AFIB_CLASS, "patient_id"].nunique()
        ),
        "dominant_future_afib_windows": int(
            (df["target_fut"] == AFIB_CLASS).sum()
        ),
        "dominant_future_afib_records": positive_records,
        "any_future_afib_mass_windows": int(
            (df["target_fut_soft_afib"] > 0).sum()
        ),
        "any_future_afib_mass_records": int(
            df.loc[df["target_fut_soft_afib"] > 0, "patient_id"].nunique()
        ),
        "current_afib_events": int(baseline_observed["events"]),
        "incident_current_afib_events": int(baseline_observed["incident_events"]),
    }
    observed_json = {
        "analysis_split": "validation_only",
        "event_horizon_sec": float(horizon_sec),
        "alignment_checks": {
            "member_count": len(args.member_csvs),
            "row_and_label_alignment": "exact",
        },
        "support": support,
        "policies": json_ready(observed.to_dict(orient="records")),
        "decision": json_ready(decision),
    }

    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    observed.to_csv(output_dir / "observed_metrics.csv", index=False)
    bootstrap_samples.to_csv(output_dir / "patient_bootstrap_samples.csv", index=False)
    ci.to_csv(output_dir / "patient_bootstrap_ci.csv", index=False)
    contribution_df.to_csv(output_dir / "patient_contributions.csv", index=False)
    loo.to_csv(output_dir / "leave_one_positive_record_out.csv", index=False)
    stability.to_csv(output_dir / "candidate_stability.csv", index=False)
    buckets.to_csv(output_dir / "error_buckets.csv", index=False)
    (output_dir / "observed_metrics.json").write_text(
        json.dumps(json_ready(observed_json), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "decision.json").write_text(
        json.dumps(json_ready(decision), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(
        output_dir,
        df,
        baseline_observed,
        selected_observed,
        ci,
        loo,
        decision,
        horizon_sec,
    )

    print(f"Saved AFib validation analysis: {output_dir}")
    print(
        f"Selected {selected_observed['policy_name']}: "
        f"recall={selected_observed['dominant_recall']:.3f}, "
        f"event_recall={selected_observed['event_recall']:.3f}, "
        f"lead={selected_observed['median_lead_time_sec']:.0f}s, "
        f"false_ep/hr={selected_observed['false_alert_episodes_per_patient_hour']:.2f}"
    )
    print(
        f"Decision: {decision['status']} ({decision['scope']}), "
        f"LOO pass={loo_pass_rate:.1%}"
    )


if __name__ == "__main__":
    main()
