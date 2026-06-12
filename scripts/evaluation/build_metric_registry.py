"""Build the single metric registry used by docs and the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from healthmonitor.monitoring_policy import POLICY_CONFIG_VERSION
from healthmonitor.paths import ARTIFACTS_DIR, DEMO_DIR, EVIDENCE_DIR, POLICY_DIR


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _demo_case(case_id: str, index: int):
    replay = _load_json(DEMO_DIR / f"mitdb_{case_id}.json")
    row = replay["rows"][index]
    return {
        "case_id": case_id,
        "short_title": replay["short_title"],
        "display_index": index + 1,
        "row_index": index,
        "timestamp_min": row["timestamp_min"],
        "current_name": row["current_name"],
        "future_name": row["future_name"],
        "probabilities_fut": row["probabilities_fut"],
        "probability_normal": row["probabilities_fut"][0],
        "probability_pvc": row["probabilities_fut"][1],
        "probability_afib": row["probabilities_fut"][2],
        "overall_risk_raw": row["overall_risk_raw"],
        "signal_quality_score": row["signal_quality"]["score"],
        "signal_quality_level": row["signal_quality"]["level"],
        "future_vote_count": row["future_vote_count"],
        "risk_std": row["risk_std"],
        "agreement_level": row["agreement_level"],
        "afib_direction_score": row["afib_direction_score"],
        "official_alert_active": bool(row["policy"]["alert_active"]),
        "policy_status": row["policy"]["status"],
        "summary": replay["summary"],
    }


def build_registry():
    formal_policy = _load_json(POLICY_DIR / "final_alert_policy_selected.json")[0]
    validation = _load_json(EVIDENCE_DIR / "locked_validation_observed_metrics.json")
    ltafdb = _load_json(EVIDENCE_DIR / "ltafdb_full_observed_metrics.json")
    ltafdb_policy = next(
        item
        for item in ltafdb["policies"]
        if item["policy_name"].startswith("afib_vs_normal")
    )
    long_tail = _load_json(EVIDENCE_DIR / "long_tail_directional_decision.json")
    uncertainty = _load_json(POLICY_DIR / "ensemble_uncertainty_config.json")

    boundary_summary = {
        name: {
            "status": decision.get("status"),
            "positive_records": decision.get(
                "positive_records",
                decision.get("selected_policy", {}).get(
                    "positive_records",
                    decision.get("best_event_recall_candidate", {}).get("positive_records"),
                ),
            ),
            "event_recall": decision.get(
                "selected_policy",
                decision.get("best_event_recall_candidate", {}),
            ).get("event_recall"),
            "median_lead_time_sec": decision.get(
                "selected_policy",
                decision.get("best_event_recall_candidate", {}),
            ).get("median_lead_time_sec"),
            "false_alert_episodes_per_patient_hour": decision.get(
                "selected_policy",
                decision.get("best_event_recall_candidate", {}),
            ).get("false_alert_episodes_per_patient_hour"),
            "official_alarm_changed": False,
        }
        for name, decision in long_tail["decisions"].items()
    }

    registry = {
        "registry_version": "pre-submission-20260612",
        "policy_config_version": POLICY_CONFIG_VERSION,
        "demo_manifest_version": "competition-demo-v2",
        "formal_capability": {
            "name": "future_5min_overall_arrhythmia_warning",
            "score": "1 - P(future Normal)",
            "threshold": formal_policy["threshold"],
            "consecutive_k": formal_policy["consecutive_k"],
            "ewma_alpha": formal_policy["smoothing_alpha"],
            "locked_test_event_recall": formal_policy["locked_test_event_recall"],
            "locked_test_incident_event_recall": formal_policy[
                "locked_test_incident_event_recall"
            ],
            "locked_test_median_lead_time_sec": formal_policy[
                "locked_test_median_lead_time_sec"
            ],
            "locked_test_false_alert_episodes_per_patient_hour": formal_policy[
                "locked_test_false_alert_episodes_per_patient_hour"
            ],
            "validation_future_arrhythmia_auroc": validation[
                "future_arrhythmia_auroc"
            ],
            "validation_future_arrhythmia_ap": validation["future_arrhythmia_ap"],
            "validation_future_pvc_recall": validation["future_pvc_recall"],
            "validation_future_pvc_auroc": validation["future_pvc_auroc"],
            "validation_future_pvc_ap": validation["future_pvc_ap"],
        },
        "afib_auxiliary_research": {
            "scope": "LTAFDB 83-record primary analysis; 84-record sensitivity analysis",
            "records_primary": ltafdb["support"]["records"],
            "records_sensitivity": ltafdb["support"]["all_inferred_records"],
            "dominant_auroc": ltafdb_policy["dominant_auroc"],
            "dominant_average_precision": ltafdb_policy[
                "dominant_average_precision"
            ],
            "event_recall": ltafdb_policy["event_recall"],
            "incident_event_recall": ltafdb_policy["incident_event_recall"],
            "median_lead_time_sec": ltafdb_policy["median_lead_time_sec"],
            "false_alert_episodes_per_patient_hour": ltafdb_policy[
                "false_alert_episodes_per_patient_hour"
            ],
        },
        "agreement": {
            "high_max_risk_std": uncertainty["high"]["max_risk_std"],
            "high_required_vote_count": uncertainty["high"][
                "required_vote_count"
            ],
            "medium_max_risk_std": uncertainty["medium"]["max_risk_std"],
            "medium_required_vote_count": uncertainty["medium"][
                "required_vote_count"
            ],
        },
        "demo_cases": {
            "pvc_alarm": _demo_case("119", 4),
            "normal_no_alarm": _demo_case("100", 17),
            "afib_auxiliary": _demo_case("201", 10),
        },
        "boundary_cases": boundary_summary,
        "sources": {
            "formal_policy": str(POLICY_DIR / "final_alert_policy_selected.json"),
            "locked_validation": str(
                EVIDENCE_DIR / "locked_validation_observed_metrics.json"
            ),
            "ltafdb_external": str(EVIDENCE_DIR / "ltafdb_full_observed_metrics.json"),
            "demo_replays": str(DEMO_DIR),
        },
    }
    return registry


def main():
    out = ARTIFACTS_DIR / "metric_registry.json"
    registry = build_registry()
    out.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
