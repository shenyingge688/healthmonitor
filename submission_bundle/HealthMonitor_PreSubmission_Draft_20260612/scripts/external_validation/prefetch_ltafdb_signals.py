"""Concurrent LTAFDB signal downloader for the Round 8 external audit.

This script only downloads and verifies raw signal files. It does not run model
inference, tune thresholds, select leads, or write evaluation metrics.
"""
import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import wfdb

from download_audit_ltafdb_signals import (
    PRESERVE_FREE_BYTES,
    build_session,
    copy_metadata,
    download_dat,
    expected_dat_bytes,
    sha256_file,
)


def load_manifest(manifest_path):
    manifest = pd.read_csv(manifest_path, dtype={"record_id": str})
    manifest["record_id"] = manifest["record_id"].astype(str).str.zfill(2)
    return manifest


def load_headers(records, metadata_dir):
    return {
        record: wfdb.rdheader(str((Path(metadata_dir) / record).resolve()))
        for record in records
    }


def summarize_cache(records, headers, records_dir):
    complete = 0
    complete_bytes = 0
    in_order = 0
    first_incomplete = None
    partials = []
    missing = []

    for record in records:
        expected = expected_dat_bytes(headers[record])
        dat_path = Path(records_dir) / f"{record}.dat"
        part_path = Path(records_dir) / f"{record}.dat.part"
        actual = dat_path.stat().st_size if dat_path.exists() else 0
        ok = dat_path.exists() and actual == expected
        if ok:
            complete += 1
            complete_bytes += actual
        else:
            missing.append(record)
        if first_incomplete is None:
            if ok:
                in_order += 1
            else:
                first_incomplete = record
        if part_path.exists():
            partials.append(
                {
                    "record_id": record,
                    "partial_bytes": int(part_path.stat().st_size),
                    "expected_bytes": int(expected),
                }
            )

    return {
        "complete_signal_files_in_shared_cache": int(complete),
        "complete_signal_bytes": int(complete_bytes),
        "manifest_records_fully_downloaded_in_order": int(in_order),
        "first_incomplete_record": first_incomplete,
        "missing_records": missing,
        "partial_records": partials,
    }


def missing_download_bytes(records, headers, records_dir):
    total = 0
    for record in records:
        expected = expected_dat_bytes(headers[record])
        dat_path = Path(records_dir) / f"{record}.dat"
        part_path = Path(records_dir) / f"{record}.dat.part"
        if dat_path.exists() and dat_path.stat().st_size == expected:
            continue
        retained = part_path.stat().st_size if part_path.exists() else 0
        total += max(expected - retained, 0)
    return int(total)


class ProgressWriter:
    def __init__(self, path, records, headers, records_dir):
        self.path = Path(path)
        self.records = list(records)
        self.headers = headers
        self.records_dir = Path(records_dir)
        self.lock = threading.Lock()
        self.started = set()
        self.finished = []
        self.failed = {}

    def write(self, status, current_step=None):
        with self.lock:
            cache = summarize_cache(
                self.records, self.headers, self.records_dir
            )
            finished_records = {
                row["record_id"] for row in self.finished
            }
            active = sorted(
                self.started - finished_records - set(self.failed)
            )
            payload = {
                "status": status,
                "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "manifest_records": int(len(self.records)),
                "prefetch_started_count": int(len(self.started)),
                "prefetch_finished_count": int(len(self.finished)),
                "prefetch_failed_count": int(len(self.failed)),
                "active_records": active,
                "last_finished_record": (
                    self.finished[-1]["record_id"] if self.finished else None
                ),
                "current_step": current_step,
                "external_threshold_tuning_performed": False,
                "full_database_model_inference_started": False,
                "official_round5_alarm_changed": False,
                **cache,
            }
            if self.failed:
                payload["failed_records"] = self.failed
            self.path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

    def mark_started(self, record):
        with self.lock:
            self.started.add(record)
        self.write("running", current_step="download")

    def mark_finished(self, row):
        with self.lock:
            self.finished.append(row)
        self.write("running", current_step="download")

    def mark_failed(self, record, exc):
        with self.lock:
            self.failed[record] = str(exc)
        self.write("failed", current_step="download_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="results_v7_round7/ltafdb_pilot_audit/record_metadata.csv",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/ltafdb_pilot/metadata",
    )
    parser.add_argument(
        "--records-dir",
        default="data/ltafdb_pilot/records",
    )
    parser.add_argument(
        "--output-dir",
        default="results_v7_round8/ltafdb_full_signal_prefetch",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--records", default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    records = manifest["record_id"].tolist()
    if args.records:
        requested = [
            item.strip().zfill(2)
            for item in args.records.split(",")
            if item.strip()
        ]
        unknown = sorted(set(requested) - set(records))
        if unknown:
            raise ValueError(f"records not in frozen manifest: {unknown}")
        records = requested

    metadata_dir = Path(args.metadata_dir)
    records_dir = Path(args.records_dir)
    output_dir = Path(args.output_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = load_headers(records, metadata_dir)

    additional_required = missing_download_bytes(records, headers, records_dir)
    free_bytes = shutil.disk_usage(records_dir).free
    if free_bytes - additional_required < PRESERVE_FREE_BYTES:
        raise RuntimeError(
            "download would violate the 80 GB free-disk preservation gate: "
            f"free={free_bytes / 1024**3:.2f} GB, "
            f"additional={additional_required / 1024**3:.2f} GB"
        )

    progress = ProgressWriter(
        output_dir / "ltafdb_full_signal_prefetch_progress.json",
        records,
        headers,
        records_dir,
    )
    progress.write("running", current_step="started")
    cache = summarize_cache(records, headers, records_dir)
    to_download = cache["missing_records"]
    print(
        f"[prefetch] records={len(records)} missing={len(to_download)} "
        f"workers={args.workers} additional={additional_required / 1024**2:.1f} MB"
    )

    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = build_session()
        return thread_local.session

    def fetch(record):
        progress.mark_started(record)
        header = headers[record]
        copy_metadata(record, metadata_dir, records_dir)
        dat_path = records_dir / f"{record}.dat"
        expected = expected_dat_bytes(header)
        status = download_dat(
            get_session(),
            record,
            dat_path,
            expected,
            attempts=args.attempts,
        )
        actual = dat_path.stat().st_size
        if actual != expected:
            raise IOError(f"{record}.dat: {actual}/{expected} bytes")
        row = {
            "record_id": record,
            "status": status,
            "expected_bytes": int(expected),
            "actual_bytes": int(actual),
            "sha256": sha256_file(dat_path),
            "local_path": str(dat_path.resolve()),
        }
        progress.mark_finished(row)
        print(f"[prefetch] {record} {status}")
        return row

    rows = []
    if to_download:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(fetch, record): record for record in to_download}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    progress.mark_failed(record, exc)
                    raise

    inventory = pd.DataFrame(rows)
    if not inventory.empty:
        inventory.sort_values("record_id").to_csv(
            output_dir / "prefetch_download_inventory.csv", index=False
        )
    progress.write("complete", current_step="downloads_complete")
    final_cache = summarize_cache(records, headers, records_dir)
    print(
        f"[prefetch] complete={final_cache['complete_signal_files_in_shared_cache']}/"
        f"{len(records)} in_order={final_cache['manifest_records_fully_downloaded_in_order']}/"
        f"{len(records)}"
    )


if __name__ == "__main__":
    main()
