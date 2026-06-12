"""Evaluate frozen AFib policies on completed LTAFDB external inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_afib_directional_policy import (
    bootstrap_ci,
    bootstrap_policy,
    build_alert,
    evaluate_policy,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def per_record_metrics(df, baseline_alert, candidate_alert):
    rows = []
    for patient_id, group in df.groupby("patient_id", sort=True):
        index = group.index.to_numpy()
        target = group["target_fut"].to_numpy(dtype=int) == 2
        score = group["afib_vs_normal"].to_numpy(dtype=float)
        row = {
            "patient_id": patient_id,
            "record_id": str(group["record_id"].iloc[0]).zfill(2),
            "rows": int(len(group)),
            "dominant_afib_rows": int(target.sum()),
            "quality_valid_fraction": float(
                group["history_quality_valid"].mean()
            ),
            "baseline_alert_fraction": float(baseline_alert[index].mean()),
            "candidate_alert_fraction": float(candidate_alert[index].mean()),
        }
        if len(np.unique(target)) == 2:
            row["candidate_auroc"] = float(roc_auc_score(target, score))
            row["candidate_average_precision"] = float(
                average_precision_score(target, score)
            )
        else:
            row["candidate_auroc"] = np.nan
            row["candidate_average_precision"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def quality_sensitivity(df, baseline_alert, candidate_alert):
    valid = df["history_quality_valid"].to_numpy(dtype=bool)
    target = df["target_fut"].to_numpy(dtype=int) == 2
    score = df["afib_vs_normal"].to_numpy(dtype=float)
    rows = []
    for name, alert in (
        ("future_class_argmax", baseline_alert),
        ("afib_vs_normal_ewma_k1_t0.100", candidate_alert),
    ):
        y = target[valid]
        prediction = alert[valid]
        true_positive = int((y & prediction).sum())
        false_positive = int((~y & prediction).sum())
        rows.append(
            {
                "policy_name": name,
                "rows": int(valid.sum()),
                "support": int(y.sum()),
                "predicted_positive": int(prediction.sum()),
                "precision": (
                    true_positive / int(prediction.sum())
                    if int(prediction.sum())
                    else 0.0
                ),
                "recall": true_positive / int(y.sum()) if int(y.sum()) else 0.0,
                "false_positive_rows": false_positive,
                "score_auroc": (
                    float(roc_auc_score(y, score[valid]))
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
                "score_average_precision": (
                    float(average_precision_score(y, score[valid]))
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_cohort(df, bootstrap, seed):
    df = df.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("external evaluation cohort is empty")
    score = df["afib_vs_normal"].to_numpy(dtype=float)
    baseline_alert = df["pred_fut"].to_numpy(dtype=int) == 2
    candidate_alert = build_alert(
        df,
        score,
        threshold=0.10,
        smoothing_method="ewma",
        consecutive_k=1,
    )
    baseline, baseline_contributions = evaluate_policy(
        df,
        df["prob_fut_afib"].to_numpy(dtype=float),
        baseline_alert,
        horizon_sec=300.0,
    )
    candidate, candidate_contributions = evaluate_policy(
        df,
        score,
        candidate_alert,
        horizon_sec=300.0,
    )
    observed = pd.DataFrame(
        [
            {"policy_name": "future_class_argmax", **baseline},
            {
                "policy_name": "afib_vs_normal_ewma_k1_t0.100",
                **candidate,
            },
        ]
    )
    if int(bootstrap) > 0:
        samples = pd.concat(
            [
                bootstrap_policy(
                    df,
                    df["prob_fut_afib"].to_numpy(dtype=float),
                    baseline_alert,
                    baseline_contributions,
                    bootstrap,
                    seed,
                    "future_class_argmax",
                ),
                bootstrap_policy(
                    df,
                    score,
                    candidate_alert,
                    candidate_contributions,
                    bootstrap,
                    seed,
                    "afib_vs_normal_ewma_k1_t0.100",
                ),
            ],
            ignore_index=True,
        )
        intervals = bootstrap_ci(samples, observed)
    else:
        samples = pd.DataFrame()
        intervals = pd.DataFrame()
    return {
        "df": df,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_alert": baseline_alert,
        "candidate_alert": candidate_alert,
        "observed": observed,
        "samples": samples,
        "intervals": intervals,
        "per_record": per_record_metrics(
            df,
            baseline_alert,
            candidate_alert,
        ),
        "quality": quality_sensitivity(
            df,
            baseline_alert,
            candidate_alert,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_external_inference"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_external_evaluation"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = inference_dir / "external_inference_outputs.csv"
    protocol_path = inference_dir / "external_evaluation_protocol.json"
    summary = json.loads(
        (inference_dir / "inference_summary.json").read_text(encoding="utf-8")
    )
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != summary["protocol_sha256"]:
        raise RuntimeError("inference summary/protocol hash mismatch")

    df = pd.read_csv(outputs_path, dtype={"record_id": str})
    if df.duplicated(["patient_id", "window_id"]).any():
        raise ValueError("duplicate patient/window rows")
    if not np.allclose(
        df.groupby("patient_id")["timestamp_sec"].diff().dropna(),
        10.0,
    ):
        raise ValueError("external inference time axis is not uniformly 10 s")

    if "record_quality_pass" in df:
        quality_flag = df["record_quality_pass"]
        if quality_flag.dtype == object:
            quality_flag = quality_flag.astype(str).str.lower().map(
                {"true": True, "false": False}
            )
        if quality_flag.isna().any():
            raise ValueError("record_quality_pass contains invalid values")
        primary_df = df[quality_flag.astype(bool)]
    else:
        primary_df = df
    if primary_df["patient_id"].nunique() != int(
        summary.get("primary_record_count", primary_df["patient_id"].nunique())
    ):
        raise ValueError("primary record count does not match inference summary")
    if df["patient_id"].nunique() != int(summary["record_count"]):
        raise ValueError("all-record count does not match inference summary")
    primary = evaluate_cohort(primary_df, args.bootstrap, args.seed)
    sensitivity = (
        evaluate_cohort(df, 0, args.seed + 1)
        if len(primary_df) != len(df)
        else None
    )
    baseline = primary["baseline"]
    candidate = primary["candidate"]
    observed = primary["observed"]
    intervals = primary["intervals"]
    per_record = primary["per_record"]
    quality = primary["quality"]

    evaluable = per_record["candidate_auroc"].dropna()
    support_checks = {
        "candidate_dominant_auroc_at_least_0_65": bool(
            candidate["dominant_auroc"] >= 0.65
        ),
        "candidate_event_recall_not_below_baseline": bool(
            candidate["event_recall"] >= baseline["event_recall"]
        ),
        "candidate_false_episode_burden_within_20_percent": bool(
            candidate["false_alert_episodes_per_patient_hour"]
            <= baseline["false_alert_episodes_per_patient_hour"] * 1.20
        ),
        "per_record_median_auroc_at_least_0_60": bool(
            len(evaluable) and evaluable.median() >= 0.60
        ),
    }
    passed = sum(support_checks.values())
    if passed == len(support_checks):
        status = "full_external_support"
    elif support_checks["candidate_dominant_auroc_at_least_0_65"]:
        status = "partial_external_support"
    else:
        status = "external_support_not_established"
    decision = {
        "status": status,
        "scope": (
            f"full_ltafdb_{primary_df['patient_id'].nunique()}_record_"
            "fixed_quality_primary_analysis"
        ),
        "official_round5_alarm_changed": False,
        "external_threshold_tuning_performed": False,
        "patient_bootstrap_replicates": int(args.bootstrap),
        "checks": support_checks,
        "passed_checks": passed,
        "total_checks": len(support_checks),
    }

    observed.to_csv(output_dir / "observed_metrics.csv", index=False)
    intervals.to_csv(output_dir / "patient_bootstrap_ci.csv", index=False)
    per_record.to_csv(output_dir / "per_record_metrics.csv", index=False)
    quality.to_csv(output_dir / "quality_sensitivity.csv", index=False)
    if sensitivity is not None:
        sensitivity["observed"].to_csv(
            output_dir / "all_84_records_sensitivity_metrics.csv",
            index=False,
        )
    (output_dir / "decision.json").write_text(
        json.dumps(json_ready(decision), indent=2),
        encoding="utf-8",
    )
    payload = {
        "protocol_sha256": protocol_hash,
        "inference_outputs_sha256": sha256_file(outputs_path),
        "support": {
            "rows": int(len(primary["df"])),
            "records": int(primary["df"]["patient_id"].nunique()),
            "all_inferred_records": int(df["patient_id"].nunique()),
            "dominant_future_afib_rows": int(
                (primary["df"]["target_fut"] == 2).sum()
            ),
            "current_afib_rows": int(
                (primary["df"]["target_cur"] == 2).sum()
            ),
            "quality_valid_rows": int(
                primary["df"]["history_quality_valid"].sum()
            ),
        },
        "policies": json_ready(observed.to_dict(orient="records")),
        "all_record_sensitivity_policies": (
            json_ready(sensitivity["observed"].to_dict(orient="records"))
            if sensitivity is not None
            else None
        ),
        "decision": decision,
        "patient_bootstrap_replicates": int(args.bootstrap),
    }
    (output_dir / "observed_metrics.json").write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# LTAFDB Frozen Full External Evaluation",
        "",
        f"- primary fixed-quality records: {primary_df['patient_id'].nunique()}",
        f"- all inferred records: {df['patient_id'].nunique()}",
        f"- primary 10-second evaluation rows: {len(primary_df)}",
        f"- protocol SHA-256: `{protocol_hash}`",
        "- external threshold tuning: False",
        "",
        "## Baseline",
        "",
        f"- dominant AFib AUROC: {baseline['dominant_auroc']:.3f}",
        f"- dominant AFib recall: {baseline['dominant_recall']:.3f}",
        f"- AFib event recall: {baseline['event_recall']:.3f}",
        f"- false episodes/hr: "
        f"{baseline['false_alert_episodes_per_patient_hour']:.3f}",
        "",
        "## Frozen AFib Directional Candidate",
        "",
        f"- dominant AFib AUROC: {candidate['dominant_auroc']:.3f}",
        f"- dominant AFib AP: {candidate['dominant_average_precision']:.3f}",
        f"- dominant AFib precision: {candidate['dominant_precision']:.3f}",
        f"- dominant AFib recall: {candidate['dominant_recall']:.3f}",
        f"- AFib event recall: {candidate['event_recall']:.3f}",
        f"- incident event recall: {candidate['incident_event_recall']:.3f}",
        f"- median lead time: {candidate['median_lead_time_sec']:.1f} s",
        f"- false episodes/hr: "
        f"{candidate['false_alert_episodes_per_patient_hour']:.3f}",
        "",
        f"Decision: `{status}`.",
        "",
        "The primary analysis excludes source-header integrity exceptions. "
        "All-record results are reported only as a sensitivity analysis.",
        "",
        "This remains research evidence, not a clinical performance claim.",
    ]
    (output_dir / "LTAFDB_EXTERNAL_EVALUATION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
