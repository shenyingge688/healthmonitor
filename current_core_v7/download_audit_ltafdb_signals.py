"""
Download and audit the preselected LTAFDB external-pilot signal files.

Pilot selection is read from the annotation-only manifest. This stage does not
run a model or change any threshold. Lead 0 remains the primary input whenever
it passes fixed signal-quality gates; lead 1 is only a data-quality fallback.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import wfdb
from requests.adapters import HTTPAdapter
from scipy import signal
from urllib3.util.retry import Retry

from build_dataset_factory import (
    TARGET_FS,
    WINDOW_SEC,
    clean_ecg_signal,
)


DATABASE = "ltafdb"
VERSION = "1.0.0"
BASE_URL = f"https://physionet.org/files/{DATABASE}/{VERSION}"
PRESERVE_FREE_BYTES = 80 * 1024**3
READ_CHUNK_SAMPLES = 1_000_000
PROBE_COUNT = 24
EXTREME_AMPLITUDE_MV = 10.0


def build_session():
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "HealthMonitor-LTAFDB-signal-audit/1.0"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_16(value):
    value = int(value) % 65536
    return value - 65536 if value >= 32768 else value


def expected_dat_bytes(header):
    formats = {str(item) for item in header.fmt}
    files = {str(item) for item in header.file_name}
    if formats != {"16"} or len(files) != 1:
        raise ValueError(
            f"unsupported WFDB layout: formats={sorted(formats)}, "
            f"files={sorted(files)}"
        )
    return int(header.sig_len) * int(header.n_sig) * 2


def download_dat(
    session,
    record,
    destination,
    expected_bytes,
    overwrite=False,
    attempts=5,
):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        if destination.stat().st_size == expected_bytes:
            return "cached"
        print(
            f"[replace] {destination.name}: cached size "
            f"{destination.stat().st_size} != {expected_bytes}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(1, int(attempts) + 1):
        temporary.unlink(missing_ok=True)
        try:
            with session.get(
                f"{BASE_URL}/{record}.dat",
                stream=True,
                timeout=(20, 180),
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_bytes = temporary.stat().st_size
            if actual_bytes != expected_bytes:
                raise IOError(
                    f"incomplete {record}.dat: "
                    f"{actual_bytes}/{expected_bytes} bytes"
                )
            temporary.replace(destination)
            return "downloaded"
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt >= int(attempts):
                break
            wait_sec = min(2 ** (attempt - 1), 8)
            print(
                f"[retry] {record}.dat attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            time.sleep(wait_sec)
    raise RuntimeError(f"failed to download {record}.dat") from last_error


def copy_metadata(record, metadata_dir, records_dir):
    for extension in ("hea", "atr"):
        source = Path(metadata_dir) / f"{record}.{extension}"
        destination = Path(records_dir) / source.name
        if not source.exists():
            raise FileNotFoundError(source)
        if (
            not destination.exists()
            or source.stat().st_size != destination.stat().st_size
            or sha256_file(source) != sha256_file(destination)
        ):
            shutil.copy2(source, destination)


def empty_accumulator(n_sig):
    return {
        "count": np.zeros(n_sig, dtype=np.int64),
        "sum": np.zeros(n_sig, dtype=np.float64),
        "sumsq": np.zeros(n_sig, dtype=np.float64),
        "minimum": np.full(n_sig, np.inf, dtype=np.float64),
        "maximum": np.full(n_sig, -np.inf, dtype=np.float64),
        "extreme": np.zeros(n_sig, dtype=np.int64),
        "clipped": np.zeros(n_sig, dtype=np.int64),
        "flat_pairs": np.zeros(n_sig, dtype=np.int64),
        "pair_count": np.zeros(n_sig, dtype=np.int64),
        "flat_blocks": np.zeros(n_sig, dtype=np.int64),
        "block_count": np.zeros(n_sig, dtype=np.int64),
        "checksum_sum": np.zeros(n_sig, dtype=np.int64),
    }


def audit_full_signal(dat_path, header):
    n_sig = int(header.n_sig)
    sig_len = int(header.sig_len)
    raw = np.memmap(dat_path, dtype="<i2", mode="r")
    expected_values = sig_len * n_sig
    if raw.size != expected_values:
        raise ValueError(
            f"{Path(dat_path).name}: {raw.size} values, expected "
            f"{expected_values}"
        )
    matrix = raw.reshape(sig_len, n_sig)
    accumulator = empty_accumulator(n_sig)
    previous = None

    gains = np.asarray(header.adc_gain, dtype=np.float64)
    baselines = np.asarray(header.baseline, dtype=np.float64)
    if np.any(~np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError(f"{Path(dat_path).name}: invalid ADC gains {gains}")

    for start in range(0, sig_len, READ_CHUNK_SAMPLES):
        end = min(start + READ_CHUNK_SAMPLES, sig_len)
        digital = np.asarray(matrix[start:end], dtype=np.int32)
        physical = (digital.astype(np.float64) - baselines) / gains
        accumulator["count"] += physical.shape[0]
        accumulator["sum"] += physical.sum(axis=0)
        accumulator["sumsq"] += np.square(physical).sum(axis=0)
        accumulator["minimum"] = np.minimum(
            accumulator["minimum"], physical.min(axis=0)
        )
        accumulator["maximum"] = np.maximum(
            accumulator["maximum"], physical.max(axis=0)
        )
        accumulator["extreme"] += (
            np.abs(physical) > EXTREME_AMPLITUDE_MV
        ).sum(axis=0)
        accumulator["clipped"] += (
            (digital == np.iinfo(np.int16).min)
            | (digital == np.iinfo(np.int16).max)
        ).sum(axis=0)
        accumulator["checksum_sum"] += digital.sum(axis=0, dtype=np.int64)

        if previous is not None:
            first_diff = np.abs(digital[0] - previous)
            accumulator["flat_pairs"] += first_diff <= 1
            accumulator["pair_count"] += 1
        if len(digital) > 1:
            diffs = np.abs(np.diff(digital, axis=0))
            accumulator["flat_pairs"] += (diffs <= 1).sum(axis=0)
            accumulator["pair_count"] += len(digital) - 1
        previous = digital[-1]

    flat_block_samples = max(int(round(2.0 * float(header.fs))), 1)
    blocks_per_chunk = 4096
    block_chunk_samples = flat_block_samples * blocks_per_chunk
    usable_samples = sig_len - (sig_len % flat_block_samples)
    for start in range(0, usable_samples, block_chunk_samples):
        end = min(start + block_chunk_samples, usable_samples)
        digital = np.asarray(matrix[start:end], dtype=np.int32)
        blocks = digital.reshape(-1, flat_block_samples, n_sig)
        peak_to_peak = np.ptp(blocks, axis=1)
        accumulator["flat_blocks"] += (peak_to_peak <= 2).sum(axis=0)
        accumulator["block_count"] += len(blocks)

    rows = []
    initial = np.asarray(matrix[0], dtype=np.int64)
    for lead in range(n_sig):
        count = int(accumulator["count"][lead])
        mean = accumulator["sum"][lead] / count
        variance = max(accumulator["sumsq"][lead] / count - mean**2, 0.0)
        calculated_checksum = signed_16(accumulator["checksum_sum"][lead])
        expected_checksum = int(header.checksum[lead])
        initial_value = int(initial[lead])
        expected_initial = int(header.init_value[lead])
        rows.append(
            {
                "lead_index": lead,
                "signal_name": str(header.sig_name[lead]),
                "units": str(header.units[lead]),
                "adc_gain": float(gains[lead]),
                "baseline": float(baselines[lead]),
                "sample_count": count,
                "physical_mean_mv": float(mean),
                "physical_std_mv": float(np.sqrt(variance)),
                "physical_min_mv": float(accumulator["minimum"][lead]),
                "physical_max_mv": float(accumulator["maximum"][lead]),
                "extreme_fraction": float(
                    accumulator["extreme"][lead] / count
                ),
                "int16_clip_fraction": float(
                    accumulator["clipped"][lead] / count
                ),
                "flat_pair_fraction": float(
                    accumulator["flat_pairs"][lead]
                    / max(accumulator["pair_count"][lead], 1)
                ),
                "flat_2s_block_fraction": float(
                    accumulator["flat_blocks"][lead]
                    / max(accumulator["block_count"][lead], 1)
                ),
                "initial_value": initial_value,
                "expected_initial_value": expected_initial,
                "initial_value_match": initial_value == expected_initial,
                "calculated_checksum": calculated_checksum,
                "expected_checksum": expected_checksum,
                "checksum_match": calculated_checksum == expected_checksum,
            }
        )
    del matrix
    del raw
    return rows


def probe_starts(sig_len, fs):
    probe_samples = int(round(WINDOW_SEC * fs))
    maximum_start = sig_len - probe_samples
    if maximum_start < 0:
        raise ValueError("record is shorter than one preprocessing window")
    if maximum_start == 0:
        return np.array([0], dtype=np.int64)
    return np.unique(
        np.linspace(0, maximum_start, PROBE_COUNT, dtype=np.int64)
    )


def audit_preprocessing_probes(dat_path, header):
    n_sig = int(header.n_sig)
    sig_len = int(header.sig_len)
    fs = float(header.fs)
    gains = np.asarray(header.adc_gain, dtype=np.float64)
    baselines = np.asarray(header.baseline, dtype=np.float64)
    raw = np.memmap(dat_path, dtype="<i2", mode="r").reshape(sig_len, n_sig)
    expected_output = int(round(WINDOW_SEC * TARGET_FS))
    rows = []

    for probe_index, start in enumerate(probe_starts(sig_len, fs)):
        end = start + int(round(WINDOW_SEC * fs))
        digital = np.asarray(raw[start:end], dtype=np.float64)
        physical = (digital - baselines) / gains
        for lead in range(n_sig):
            source = physical[:, lead]
            try:
                filtered = clean_ecg_signal(source, fs=fs)
                resampled = signal.resample_poly(
                    filtered,
                    TARGET_FS,
                    int(round(fs)),
                ).astype(np.float32)
                normalized = (
                    resampled - np.mean(resampled)
                ) / (np.std(resampled) + 1e-8)
                finite = bool(
                    np.isfinite(filtered).all()
                    and np.isfinite(resampled).all()
                    and np.isfinite(normalized).all()
                )
                output_length = int(len(resampled))
                normalized_mean = float(np.mean(normalized))
                normalized_std = float(np.std(normalized))
                pass_probe = bool(
                    finite
                    and output_length == expected_output
                    and abs(normalized_mean) <= 1e-4
                    and 0.999 <= normalized_std <= 1.001
                )
                error = ""
            except Exception as exc:
                output_length = 0
                finite = False
                normalized_mean = np.nan
                normalized_std = np.nan
                pass_probe = False
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "probe_index": int(probe_index),
                    "start_sample": int(start),
                    "start_sec": float(start / fs),
                    "lead_index": int(lead),
                    "source_length": int(len(source)),
                    "output_length": output_length,
                    "expected_output_length": expected_output,
                    "finite": finite,
                    "normalized_mean": normalized_mean,
                    "normalized_std": normalized_std,
                    "pass_probe": pass_probe,
                    "error": error,
                }
            )
    del raw
    return rows


def apply_quality_gates(lead_quality, probe_table):
    probe_summary = (
        probe_table.groupby(["record_id", "lead_index"], as_index=False)
        .agg(
            probe_count=("pass_probe", "size"),
            probe_pass_count=("pass_probe", "sum"),
        )
    )
    probe_summary["probe_pass_fraction"] = (
        probe_summary["probe_pass_count"] / probe_summary["probe_count"]
    )
    quality = lead_quality.merge(
        probe_summary,
        on=["record_id", "lead_index"],
        how="left",
        validate="one_to_one",
    )
    quality["quality_pass"] = (
        quality["initial_value_match"]
        & quality["checksum_match"]
        & quality["physical_std_mv"].between(0.01, 10.0, inclusive="both")
        & (quality["extreme_fraction"] <= 0.01)
        & (quality["int16_clip_fraction"] <= 0.0001)
        & (quality["flat_2s_block_fraction"] <= 0.05)
        & (quality["probe_pass_fraction"] >= 0.95)
    )
    return quality


def choose_leads(quality):
    rows = []
    for record, group in quality.groupby("record_id", sort=False):
        group = group.sort_values("lead_index")
        primary = group[group["lead_index"] == 0]
        fallback = group[group["lead_index"] == 1]
        if len(primary) != 1 or len(fallback) != 1:
            raise ValueError(f"{record}: expected exactly two leads")
        primary_pass = bool(primary.iloc[0]["quality_pass"])
        fallback_pass = bool(fallback.iloc[0]["quality_pass"])
        if primary_pass:
            selected = 0
            reason = "primary_lead_0_passed_fixed_quality_gates"
        elif fallback_pass:
            selected = 1
            reason = "lead_0_failed_quality_gates_lead_1_fallback"
        else:
            selected = -1
            reason = "both_leads_failed_quality_gates"
        rows.append(
            {
                "record_id": str(record),
                "selected_lead_index": selected,
                "selection_reason": reason,
                "record_quality_pass": selected >= 0,
                "selection_uses_model_performance": False,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir,
    manifest,
    inventory,
    quality,
    probes,
    selected_leads,
    expected_full_count,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "signal_file_inventory.csv", index=False)
    quality.to_csv(output_dir / "lead_quality.csv", index=False)
    probes.to_csv(output_dir / "preprocessing_probes.csv", index=False)
    selected_leads.to_csv(output_dir / "selected_leads.csv", index=False)

    signal_manifest = manifest.merge(
        selected_leads,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    signal_manifest["signal_download_status"] = "present_and_verified"
    signal_manifest["model_evaluation_status"] = "not_run"
    signal_manifest.to_csv(
        output_dir / "pilot_signal_manifest.csv",
        index=False,
    )

    full_pilot = len(manifest) == int(expected_full_count)
    audit_pass = bool(
        full_pilot
        and len(inventory) == expected_full_count
        and inventory["byte_count_match"].all()
        and inventory["sha256"].str.len().eq(64).all()
        and quality["checksum_match"].all()
        and quality["initial_value_match"].all()
        and selected_leads["record_quality_pass"].all()
        and not probes["error"].fillna("").astype(str).str.len().gt(0).any()
    )
    summary = {
        "database": DATABASE,
        "version": VERSION,
        "record_count": int(len(manifest)),
        "expected_full_pilot_record_count": int(expected_full_count),
        "full_pilot": full_pilot,
        "download_bytes": int(inventory["actual_bytes"].sum()),
        "all_file_sizes_match": bool(inventory["byte_count_match"].all()),
        "all_header_checksums_match": bool(quality["checksum_match"].all()),
        "all_initial_values_match": bool(quality["initial_value_match"].all()),
        "all_records_pass_quality": bool(
            selected_leads["record_quality_pass"].all()
        ),
        "selected_lead_0_count": int(
            (selected_leads["selected_lead_index"] == 0).sum()
        ),
        "selected_lead_1_count": int(
            (selected_leads["selected_lead_index"] == 1).sum()
        ),
        "failed_record_count": int(
            (~selected_leads["record_quality_pass"]).sum()
        ),
        "preprocessing_probe_count": int(len(probes)),
        "preprocessing_probe_pass_count": int(probes["pass_probe"].sum()),
        "preprocessing_probe_failure_count": int((~probes["pass_probe"]).sum()),
        "preprocessing_probe_pass_fraction": float(
            probes["pass_probe"].mean()
        ),
        "records_with_probe_failures": sorted(
            probes.loc[~probes["pass_probe"], "record_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "target_fs_hz": int(TARGET_FS),
        "window_sec": int(WINDOW_SEC),
        "expected_window_samples": int(TARGET_FS * WINDOW_SEC),
        "lead_selection_rule": (
            "use lead 0 when it passes fixed integrity/quality gates; "
            "otherwise use lead 1 only if it passes the same gates"
        ),
        "selection_uses_model_performance": False,
        "model_evaluation_status": "not_run",
        "minimum_free_disk_gate_gb": int(PRESERVE_FREE_BYTES / 1024**3),
        "audit_pass": audit_pass,
    }
    (output_dir / "signal_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# LTAFDB Pilot Signal Integrity and Preprocessing Audit",
        "",
        f"- records audited: {summary['record_count']}",
        f"- downloaded signal size: {summary['download_bytes'] / 1024**2:.1f} MB",
        f"- all file sizes match headers: {summary['all_file_sizes_match']}",
        f"- all WFDB checksums match: {summary['all_header_checksums_match']}",
        f"- all initial samples match headers: {summary['all_initial_values_match']}",
        f"- preprocessing probes passed: "
        f"{summary['preprocessing_probe_pass_count']}/"
        f"{summary['preprocessing_probe_count']} "
        f"({summary['preprocessing_probe_pass_fraction']:.3f})",
        f"- records with isolated failed probes: "
        f"{summary['records_with_probe_failures']}",
        f"- lead 0 selected: {summary['selected_lead_0_count']}",
        f"- lead 1 quality fallback selected: {summary['selected_lead_1_count']}",
        f"- failed records: {summary['failed_record_count']}",
        f"- audit pass: {summary['audit_pass']}",
        "",
        "## Frozen Data-Quality Rule",
        "",
        "Lead 0 is used whenever it passes fixed integrity, amplitude, clipping, "
        "flatline, and preprocessing gates. Lead 1 is used only as a quality "
        "fallback. Model outputs are not consulted.",
        "",
        "Each probe follows the existing project preprocessing path: 0.5-45 Hz "
        "Butterworth filtering, 128-to-250 Hz polyphase resampling, and "
        "per-window z-normalization. Every 30-second probe must contain 7,500 "
        "finite samples after resampling.",
        "",
        "Record `113` contains an isolated interval where both leads are nearly "
        "constant. It passes the pre-specified record-level 95% probe gate, but "
        "external evaluation must retain the full-time primary analysis and "
        "separately report a signal-quality-valid sensitivity analysis.",
        "",
        "No model inference or external threshold selection was performed.",
    ]
    (output_dir / "LTAFDB_SIGNAL_AUDIT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_pilot_audit\pilot_manifest.csv"
        ),
    )
    parser.add_argument(
        "--metadata-dir",
        default=r"C:\HealthMonitor\data\ltafdb_pilot\metadata",
    )
    parser.add_argument(
        "--records-dir",
        default=r"C:\HealthMonitor\data\ltafdb_pilot\records",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            r"C:\HealthMonitor\current_core_v7\results_v7_round7"
            r"\ltafdb_signal_audit"
        ),
    )
    parser.add_argument(
        "--records",
        default=None,
        help="Optional comma-separated manifest subset for a smoke audit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    metadata_dir = Path(args.metadata_dir)
    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir)
    manifest = pd.read_csv(manifest_path, dtype={"record_id": str})
    manifest["record_id"] = manifest["record_id"].astype(str).str.zfill(2)
    expected_full_count = len(manifest)

    if args.records:
        requested = [
            item.strip().zfill(2)
            for item in args.records.split(",")
            if item.strip()
        ]
        unknown = sorted(set(requested) - set(manifest["record_id"]))
        if unknown:
            raise ValueError(f"records not in frozen manifest: {unknown}")
        manifest = manifest.set_index("record_id").loc[requested].reset_index()

    records_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {}
    expected_total = 0
    for record in manifest["record_id"]:
        header = wfdb.rdheader(str((metadata_dir / record).resolve()))
        headers[record] = header
        expected_total += expected_dat_bytes(header)

    disk_free_before = shutil.disk_usage(records_dir).free
    additional_required = sum(
        max(
            expected_dat_bytes(headers[record])
            - (
                records_dir / f"{record}.dat"
            ).stat().st_size
            if (records_dir / f"{record}.dat").exists()
            else expected_dat_bytes(headers[record]),
            0,
        )
        for record in manifest["record_id"]
    )
    if disk_free_before - additional_required < PRESERVE_FREE_BYTES:
        raise RuntimeError(
            "download would violate the 80 GB free-disk preservation gate: "
            f"free={disk_free_before / 1024**3:.2f} GB, "
            f"additional={additional_required / 1024**3:.2f} GB"
        )
    print(
        f"[space] free={disk_free_before / 1024**3:.2f} GB, "
        f"pilot={expected_total / 1024**2:.1f} MB"
    )

    session = build_session()
    inventory_rows = []
    quality_rows = []
    probe_rows = []
    for index, record in enumerate(manifest["record_id"], start=1):
        header = headers[record]
        copy_metadata(record, metadata_dir, records_dir)
        dat_path = records_dir / f"{record}.dat"
        expected_bytes = expected_dat_bytes(header)
        status = download_dat(
            session,
            record,
            dat_path,
            expected_bytes,
            overwrite=args.overwrite,
        )
        actual_bytes = dat_path.stat().st_size
        inventory_rows.append(
            {
                "record_id": record,
                "status": "present_and_verified",
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "byte_count_match": actual_bytes == expected_bytes,
                "sha256": sha256_file(dat_path),
                "local_path": str(dat_path.resolve()),
            }
        )

        for row in audit_full_signal(dat_path, header):
            row["record_id"] = record
            quality_rows.append(row)
        for row in audit_preprocessing_probes(dat_path, header):
            row["record_id"] = record
            probe_rows.append(row)
        print(f"[signal] {index}/{len(manifest)} {record} {status}")

    inventory = pd.DataFrame(inventory_rows)
    lead_quality = pd.DataFrame(quality_rows)
    probes = pd.DataFrame(probe_rows)
    quality = apply_quality_gates(lead_quality, probes)
    selected_leads = choose_leads(quality)
    summary = write_outputs(
        output_dir,
        manifest,
        inventory,
        quality,
        probes,
        selected_leads,
        expected_full_count,
    )
    print(f"Saved signal audit: {output_dir}")
    print(
        f"records={len(manifest)}, bytes={inventory['actual_bytes'].sum()}, "
        f"lead0={summary['selected_lead_0_count']}, "
        f"lead1={summary['selected_lead_1_count']}, "
        f"audit_pass={summary['audit_pass']}"
    )


if __name__ == "__main__":
    main()
