"""Freeze the full LTAFDB evaluation cohort from signal-audit evidence.

The fixed quality gate remains authoritative. Record 20 is retained only as
an explicitly marked source-header-anomaly sensitivity record after its local
signal file was verified byte-for-byte against the PhysioNet source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


RECORD20_SOURCE_SHA256 = (
    "1cd790a57f70c7ffbef0f020d04910ff381d22656984aeb601037f71433c2eb3"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round8"
            r"\ltafdb_full_signal_audit"
        ),
    )
    parser.add_argument(
        "--records-dir",
        default=r"C:\HealthMonitor\data\ltafdb_pilot\records",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round8"
            r"\ltafdb_full_evaluation_inputs"
        ),
    )
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(
        (audit_dir / "signal_audit_summary.json").read_text(encoding="utf-8")
    )
    if int(summary["record_count"]) != 84:
        raise RuntimeError("full evaluation requires the frozen 84-record audit")
    if not bool(summary["all_file_sizes_match"]):
        raise RuntimeError("signal audit contains a file-size mismatch")

    manifest = pd.read_csv(
        audit_dir / "pilot_signal_manifest.csv",
        dtype={"record_id": str},
    )
    leads = pd.read_csv(
        audit_dir / "selected_leads.csv",
        dtype={"record_id": str},
    )
    manifest["record_id"] = manifest["record_id"].str.zfill(2)
    leads["record_id"] = leads["record_id"].str.zfill(2)
    if len(manifest) != 84 or len(leads) != 84:
        raise RuntimeError("audit outputs do not contain exactly 84 records")

    failed = leads.loc[~leads["record_quality_pass"].astype(bool), "record_id"]
    if failed.tolist() != ["20"]:
        raise RuntimeError(f"unexpected fixed-quality failures: {failed.tolist()}")

    record20_path = records_dir / "20.dat"
    local_sha256 = sha256_file(record20_path)
    if local_sha256 != RECORD20_SOURCE_SHA256:
        raise RuntimeError(
            "record 20 no longer matches the independently verified "
            "PhysioNet source hash"
        )

    leads["evaluation_eligible"] = leads["record_quality_pass"].astype(bool)
    leads["source_integrity_exception"] = False
    record20 = leads["record_id"] == "20"
    leads.loc[record20, "selected_lead_index"] = 0
    leads.loc[record20, "selection_reason"] = (
        "physionet_source_header_initial_and_checksum_anomaly_sensitivity_only"
    )
    leads.loc[record20, "evaluation_eligible"] = True
    leads.loc[record20, "source_integrity_exception"] = True
    leads["analysis_role"] = "primary"
    leads.loc[record20, "analysis_role"] = "sensitivity_only"

    manifest = manifest.drop(
        columns=[
            column
            for column in (
                "selected_lead_index",
                "selection_reason",
                "record_quality_pass",
                "selection_uses_model_performance",
                "signal_download_status",
                "model_evaluation_status",
            )
            if column in manifest.columns
        ]
    )
    manifest["evaluation_scope"] = "full_ltafdb_84_frozen"

    manifest_path = output_dir / "full_manifest.csv"
    leads_path = output_dir / "full_selected_leads.csv"
    manifest.to_csv(manifest_path, index=False)
    leads.to_csv(leads_path, index=False)

    evidence = {
        "status": "frozen",
        "record_count": 84,
        "primary_analysis_records": 83,
        "sensitivity_only_records": ["20"],
        "fixed_signal_audit_pass": bool(summary["audit_pass"]),
        "fixed_quality_gate_changed": False,
        "external_threshold_tuning_performed": False,
        "record_20": {
            "record_quality_pass": False,
            "preprocessing_probes_pass": True,
            "physionet_source_full_sha256": RECORD20_SOURCE_SHA256,
            "local_full_sha256": local_sha256,
            "source_and_local_exact_match": True,
            "reason": (
                "official header initial values and checksums disagree with "
                "the official signal bytes; record excluded from primary "
                "analysis and retained only for sensitivity analysis"
            ),
        },
        "source_audit_summary_sha256": sha256_file(
            audit_dir / "signal_audit_summary.json"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_leads_sha256": sha256_file(leads_path),
    }
    (output_dir / "freeze_evidence.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
