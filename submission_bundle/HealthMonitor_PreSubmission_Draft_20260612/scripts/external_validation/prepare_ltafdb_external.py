"""
Prepare an annotation-selected Long-Term AF Database external pilot.

This stage downloads only RECORDS, .hea, and .atr files. Signal .dat files are
intentionally excluded until annotation mapping and pilot selection pass audit.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import wfdb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATABASE = "ltafdb"
VERSION = "1.0.0"
BASE_URL = f"https://physionet.org/files/{DATABASE}/{VERSION}"
SOURCE_URL = f"https://physionet.org/content/{DATABASE}/{VERSION}/"
DOI = "10.13026/C2QG6Q"
DEFAULT_PILOT_SIZE = 12
AF_MERGE_GAP_SEC = 60.0
AF_MIN_EPISODE_SEC = 30.0
AF_VALID_ONSET_MIN_EPISODE_SEC = 60.0
AF_FREE_HISTORY_SEC = 600.0
PREDICTION_HORIZON_SEC = 300.0
BURDEN_TARGETS = {
    "low": (0.025, 0.075),
    "medium": (0.20, 0.40),
    "high": (0.65, 0.90),
}


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
        {"User-Agent": "HealthMonitor-LTAFDB-audit/1.0 (metadata-only)"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(session, url, destination, overwrite=False, attempts=5):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(1, int(attempts) + 1):
        temporary.unlink(missing_ok=True)
        try:
            expected_bytes = None
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                if response.headers.get("Content-Length"):
                    expected_bytes = int(response.headers["Content-Length"])
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_bytes = temporary.stat().st_size
            if actual_bytes <= 0:
                raise ValueError(f"downloaded empty file: {url}")
            if expected_bytes is not None and actual_bytes != expected_bytes:
                raise IOError(
                    f"incomplete download for {url}: "
                    f"{actual_bytes}/{expected_bytes} bytes"
                )
            temporary.replace(destination)
            return "downloaded"
        except (OSError, ValueError, requests.RequestException) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt >= int(attempts):
                break
            wait_sec = min(2 ** (attempt - 1), 8)
            print(
                f"[retry] {destination.name} attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            time.sleep(wait_sec)
    raise RuntimeError(f"failed to download {url}") from last_error


def load_record_list(session, metadata_dir, overwrite=False):
    records_file = Path(metadata_dir) / "RECORDS"
    download_file(
        session,
        f"{BASE_URL}/RECORDS",
        records_file,
        overwrite=overwrite,
    )
    records = [
        line.strip()
        for line in records_file.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(records) != len(set(records)):
        raise ValueError("RECORDS contains duplicate record names")
    if not records:
        raise ValueError("RECORDS is empty")
    return records


def download_metadata(session, records, metadata_dir, overwrite=False):
    metadata_dir = Path(metadata_dir)
    rows = []
    for index, record in enumerate(records, start=1):
        for extension in ("hea", "atr"):
            destination = metadata_dir / f"{record}.{extension}"
            status = download_file(
                session,
                f"{BASE_URL}/{record}.{extension}",
                destination,
                overwrite=overwrite,
            )
            rows.append(
                {
                    "record_id": record,
                    "extension": extension,
                    "status": status,
                    "bytes": int(destination.stat().st_size),
                    "local_path": str(destination.resolve()),
                }
            )
        print(f"[metadata] {index}/{len(records)} {record}")
    return pd.DataFrame(rows)


def normalize_rhythm_note(note):
    if not isinstance(note, str):
        return None
    cleaned = note.replace("\x00", "").strip()
    if not cleaned.startswith("("):
        return None
    rhythm = cleaned[1:].strip().upper()
    return rhythm or None


def burden_stratum(af_burden):
    if af_burden < 0.10:
        return "low"
    if af_burden < 0.50:
        return "medium"
    return "high"


def analyze_record(metadata_dir, record):
    record_path = str((Path(metadata_dir) / record).resolve())
    header = wfdb.rdheader(record_path)
    annotation = wfdb.rdann(record_path, "atr", pn_dir=None)
    fs = float(header.fs)
    sig_len = int(header.sig_len)
    if fs <= 0 or sig_len <= 0:
        raise ValueError(f"{record}: invalid fs or signal length")
    if len(annotation.sample) != len(annotation.aux_note):
        raise ValueError(f"{record}: annotation sample/aux_note length mismatch")

    raw_counts = Counter()
    markers = []
    for sample, aux_note in zip(annotation.sample, annotation.aux_note):
        rhythm = normalize_rhythm_note(aux_note)
        if rhythm is None:
            continue
        sample = min(max(int(sample), 0), sig_len)
        raw_counts[rhythm] += 1
        markers.append((sample, rhythm))
    markers.sort(key=lambda item: item[0])

    deduplicated = []
    for sample, rhythm in markers:
        if deduplicated and deduplicated[-1][0] == sample:
            deduplicated[-1] = (sample, rhythm)
        else:
            deduplicated.append((sample, rhythm))
    markers = deduplicated
    if not markers:
        raise ValueError(f"{record}: no rhythm markers found in atr annotations")

    duration_sec = sig_len / fs
    unknown_initial_sec = markers[0][0] / fs
    rhythm_durations = Counter()
    rhythm_segments = []

    for index, (start_sample, rhythm) in enumerate(markers):
        end_sample = markers[index + 1][0] if index + 1 < len(markers) else sig_len
        end_sample = max(end_sample, start_sample)
        segment_sec = (end_sample - start_sample) / fs
        rhythm_durations[rhythm] += segment_sec
        rhythm_segments.append((start_sample, end_sample, rhythm))

    raw_af_segments = [
        [start_sample, end_sample]
        for start_sample, end_sample, rhythm in rhythm_segments
        if rhythm == "AFIB" and end_sample > start_sample
    ]
    merge_gap_samples = int(round(AF_MERGE_GAP_SEC * fs))
    merged_af_segments = []
    for start_sample, end_sample in raw_af_segments:
        if (
            merged_af_segments
            and start_sample - merged_af_segments[-1][1] <= merge_gap_samples
        ):
            merged_af_segments[-1][1] = max(
                merged_af_segments[-1][1],
                end_sample,
            )
        else:
            merged_af_segments.append([start_sample, end_sample])
    minimum_episode_samples = int(round(AF_MIN_EPISODE_SEC * fs))
    af_episodes = [
        (start_sample, end_sample)
        for start_sample, end_sample in merged_af_segments
        if end_sample - start_sample >= minimum_episode_samples
    ]

    rhythm_before_sample = {}
    rhythm_after_sample = {}
    for index, (start_sample, end_sample, rhythm) in enumerate(rhythm_segments):
        if index > 0:
            rhythm_before_sample[start_sample] = rhythm_segments[index - 1][2]
        if index + 1 < len(rhythm_segments):
            rhythm_after_sample[end_sample] = rhythm_segments[index + 1][2]

    incident_af_onsets = 0
    normal_to_af_onsets = 0
    valid_prediction_onsets = 0
    af_terminations = 0
    previous_episode_end = None
    for start_sample, end_sample in af_episodes:
        previous_rhythm = rhythm_before_sample.get(start_sample)
        if previous_rhythm is not None:
            incident_af_onsets += 1
        is_normal_onset = previous_rhythm == "N"
        normal_to_af_onsets += int(is_normal_onset)

        episode_sec = (end_sample - start_sample) / fs
        af_free_history_sec = (
            np.inf
            if previous_episode_end is None
            else (start_sample - previous_episode_end) / fs
        )
        has_history = start_sample >= int(round(AF_FREE_HISTORY_SEC * fs))
        has_future = (
            start_sample + int(round(PREDICTION_HORIZON_SEC * fs)) <= sig_len
        )
        valid_prediction_onsets += int(
            is_normal_onset
            and episode_sec >= AF_VALID_ONSET_MIN_EPISODE_SEC
            and af_free_history_sec >= AF_FREE_HISTORY_SEC
            and has_history
            and has_future
        )
        af_terminations += int(rhythm_after_sample.get(end_sample) == "N")
        previous_episode_end = end_sample

    af_duration_sec = float(rhythm_durations.get("AFIB", 0.0))
    labeled_duration_sec = float(sum(rhythm_durations.values()))
    af_burden_total = af_duration_sec / duration_sec
    af_burden_labeled = (
        af_duration_sec / labeled_duration_sec if labeled_duration_sec > 0 else np.nan
    )
    longest_af_sec = max(
        ((end - start) / fs for start, end in af_episodes),
        default=0.0,
    )
    return {
        "record_id": record,
        "fs_hz": fs,
        "signal_count": int(header.n_sig),
        "signal_names": "|".join(header.sig_name),
        "signal_length_samples": sig_len,
        "duration_hours": duration_sec / 3600.0,
        "annotation_count": int(len(annotation.sample)),
        "rhythm_marker_count": int(len(markers)),
        "unknown_initial_sec": float(unknown_initial_sec),
        "labeled_coverage": labeled_duration_sec / duration_sec,
        "af_duration_hours": af_duration_sec / 3600.0,
        "af_burden_total": af_burden_total,
        "af_burden_labeled": af_burden_labeled,
        "raw_af_segment_count": int(len(raw_af_segments)),
        "af_episode_count": int(len(af_episodes)),
        "incident_af_onsets": int(incident_af_onsets),
        "normal_to_af_onsets": int(normal_to_af_onsets),
        "valid_prediction_onsets": int(valid_prediction_onsets),
        "af_terminations": int(af_terminations),
        "longest_af_episode_hours": longest_af_sec / 3600.0,
        "burden_stratum": burden_stratum(af_burden_total),
        "rhythm_notes_json": json.dumps(dict(sorted(raw_counts.items()))),
    }, raw_counts


def select_pilot(metadata, pilot_size):
    if pilot_size < 3:
        raise ValueError("pilot_size must be at least 3")
    eligible = metadata[
        (metadata["duration_hours"] >= 20.0)
        & (metadata["labeled_coverage"] >= 0.80)
        & (metadata["af_episode_count"] >= 1)
    ].copy()
    if len(eligible) < pilot_size:
        raise ValueError(
            f"only {len(eligible)} records pass audit, fewer than pilot_size={pilot_size}"
        )

    base_quota = pilot_size // 3
    remainder = pilot_size % 3
    quotas = {
        stratum: base_quota + int(index < remainder)
        for index, stratum in enumerate(("low", "medium", "high"))
    }
    selected_rows = []
    selected_ids = set()

    for stratum in ("low", "medium", "high"):
        group = eligible[eligible["burden_stratum"] == stratum].copy()
        quota = quotas[stratum]
        if len(group) < quota:
            raise ValueError(
                f"stratum {stratum} has {len(group)} eligible records, needs {quota}"
            )

        transition_slots = quota // 2
        transition_ranked = group.sort_values(
            [
                "valid_prediction_onsets",
                "normal_to_af_onsets",
                "incident_af_onsets",
                "record_id",
            ],
            ascending=[False, False, False, True],
        )
        for _, row in transition_ranked.head(transition_slots).iterrows():
            record_id = str(row["record_id"])
            selected_ids.add(record_id)
            item = row.to_dict()
            item["selection_reason"] = f"{stratum}_burden_transition_rich"
            selected_rows.append(item)

        remaining_targets = BURDEN_TARGETS[stratum]
        for target in remaining_targets:
            if sum(
                row["burden_stratum"] == stratum for row in selected_rows
            ) >= quota:
                break
            candidates = group[~group["record_id"].astype(str).isin(selected_ids)].copy()
            candidates["target_distance"] = (
                candidates["af_burden_total"] - float(target)
            ).abs()
            best = candidates.sort_values(
                ["target_distance", "valid_prediction_onsets", "record_id"],
                ascending=[True, False, True],
            ).iloc[0]
            record_id = str(best["record_id"])
            selected_ids.add(record_id)
            item = best.drop(labels=["target_distance"]).to_dict()
            item["selection_reason"] = (
                f"{stratum}_burden_near_{float(target):.3f}"
            )
            selected_rows.append(item)

        while sum(row["burden_stratum"] == stratum for row in selected_rows) < quota:
            candidates = group[~group["record_id"].astype(str).isin(selected_ids)]
            best = candidates.sort_values(
                ["valid_prediction_onsets", "record_id"],
                ascending=[False, True],
            ).iloc[0]
            record_id = str(best["record_id"])
            selected_ids.add(record_id)
            item = best.to_dict()
            item["selection_reason"] = f"{stratum}_burden_fill"
            selected_rows.append(item)

    manifest = pd.DataFrame(selected_rows)
    manifest["signal_download_status"] = "not_downloaded"
    manifest["model_evaluation_status"] = "not_run"
    manifest["selection_uses_model_performance"] = False
    return manifest.sort_values(
        ["burden_stratum", "af_burden_total", "record_id"]
    ).reset_index(drop=True)


def write_summary(
    output_dir,
    metadata_dir,
    metadata,
    manifest,
    file_inventory,
    note_counts,
):
    output_dir = Path(output_dir)
    selected_counts = manifest["burden_stratum"].value_counts().to_dict()
    eligible = metadata[
        (metadata["duration_hours"] >= 20.0)
        & (metadata["labeled_coverage"] >= 0.80)
        & (metadata["af_episode_count"] >= 1)
    ]
    summary = {
        "database": DATABASE,
        "version": VERSION,
        "doi": DOI,
        "source_url": SOURCE_URL,
        "record_count": int(len(metadata)),
        "pilot_record_count": int(len(manifest)),
        "metadata_download_bytes": int(file_inventory["bytes"].sum()),
        "signal_files_downloaded": 0,
        "all_records_fs_128_hz": bool(np.allclose(metadata["fs_hz"], 128.0)),
        "all_records_two_signal": bool((metadata["signal_count"] == 2).all()),
        "minimum_duration_hours": float(metadata["duration_hours"].min()),
        "median_duration_hours": float(metadata["duration_hours"].median()),
        "minimum_labeled_coverage": float(metadata["labeled_coverage"].min()),
        "eligible_record_count": int(len(eligible)),
        "pilot_minimum_duration_hours": float(manifest["duration_hours"].min()),
        "pilot_minimum_labeled_coverage": float(
            manifest["labeled_coverage"].min()
        ),
        "af_burden_range": [
            float(metadata["af_burden_total"].min()),
            float(metadata["af_burden_total"].max()),
        ],
        "total_valid_prediction_onsets": int(
            metadata["valid_prediction_onsets"].sum()
        ),
        "pilot_stratum_counts": {
            key: int(selected_counts.get(key, 0))
            for key in ("low", "medium", "high")
        },
        "selection_uses_model_performance": False,
        "event_definition": {
            "merge_af_gaps_sec": AF_MERGE_GAP_SEC,
            "minimum_af_episode_sec": AF_MIN_EPISODE_SEC,
            "valid_onset_minimum_af_sec": AF_VALID_ONSET_MIN_EPISODE_SEC,
            "valid_onset_af_free_history_sec": AF_FREE_HISTORY_SEC,
            "prediction_horizon_sec": PREDICTION_HORIZON_SEC,
            "valid_onset_requires_previous_rhythm": "N",
        },
        "audit_pass": bool(
            len(metadata) == 84
            and len(manifest) > 0
            and np.allclose(metadata["fs_hz"], 128.0)
            and (metadata["signal_count"] == 2).all()
            and len(eligible) >= len(manifest)
            and (manifest["duration_hours"] >= 20.0).all()
            and (manifest["labeled_coverage"] >= 0.80).all()
            and (manifest["af_episode_count"] >= 1).all()
            and not any(Path(metadata_dir).rglob("*.dat"))
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# LTAFDB External Pilot Metadata Audit",
        "",
        f"- source: {SOURCE_URL}",
        f"- database version: {VERSION}",
        f"- records audited: {len(metadata)}",
        f"- pilot records selected: {len(manifest)}",
        f"- downloaded metadata size: {summary['metadata_download_bytes'] / 1024**2:.1f} MB",
        "- signal `.dat` files downloaded: 0",
        f"- all records at 128 Hz: {summary['all_records_fs_128_hz']}",
        f"- all records have two signals: {summary['all_records_two_signal']}",
        f"- duration range starts at: {summary['minimum_duration_hours']:.2f} h",
        f"- minimum labeled coverage: {summary['minimum_labeled_coverage']:.3f}",
        f"- records eligible for pilot selection: {summary['eligible_record_count']}",
        f"- pilot minimum duration: {summary['pilot_minimum_duration_hours']:.2f} h",
        f"- pilot minimum labeled coverage: "
        f"{summary['pilot_minimum_labeled_coverage']:.3f}",
        f"- total valid 10-min-history/5-min-future AF onsets: "
        f"{summary['total_valid_prediction_onsets']}",
        f"- pilot burden strata: {summary['pilot_stratum_counts']}",
        f"- audit pass: {summary['audit_pass']}",
        "",
        "## Selection Rule",
        "",
        "Records were selected before model inference using only annotation-derived "
        "AF burden, valid AF onset count, duration, and annotation coverage. Each "
        "record remains an independent patient/record unit. Thresholds must not be "
        "reselected on this external pilot.",
        "",
        "AF segments separated by at most 60 seconds are merged, episodes shorter "
        "than 30 seconds are excluded, and a valid prediction onset requires a "
        "previous `N` rhythm, at least 60 seconds of AF, and 10 AF-free minutes of "
        "history.",
        "",
        "## Selected Records",
        "",
        "| record | stratum | AF burden | AF episodes | valid onsets | reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in manifest.iterrows():
        lines.append(
            f"| {row['record_id']} | {row['burden_stratum']} | "
            f"{row['af_burden_total']:.3f} | {int(row['af_episode_count'])} | "
            f"{int(row['valid_prediction_onsets'])} | {row['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Rhythm Notes",
            "",
            ", ".join(
                f"`{row.rhythm_note}`={int(row.count)}"
                for row in note_counts.itertuples()
            ),
            "",
            "No external performance is reported at this stage.",
        ]
    )
    (output_dir / "LTAFDB_PILOT_AUDIT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        default=r"C:\HealthMonitor\data\ltafdb_pilot\metadata",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\HealthMonitor\current_core_v7\results_v7_round7\ltafdb_pilot_audit",
    )
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Smoke-test limit. Full audit must leave this unset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    records = load_record_list(session, metadata_dir, overwrite=args.overwrite)
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    start = time.time()
    file_inventory = download_metadata(
        session,
        records,
        metadata_dir,
        overwrite=args.overwrite,
    )

    metadata_rows = []
    note_counts = Counter()
    for index, record in enumerate(records, start=1):
        row, record_notes = analyze_record(metadata_dir, record)
        metadata_rows.append(row)
        note_counts.update(record_notes)
        print(f"[audit] {index}/{len(records)} {record}")
    metadata = pd.DataFrame(metadata_rows).sort_values("record_id").reset_index(
        drop=True
    )
    pilot_size = min(int(args.pilot_size), len(metadata))
    manifest = select_pilot(metadata, pilot_size=pilot_size)
    notes = pd.DataFrame(
        [
            {"rhythm_note": rhythm, "count": int(count)}
            for rhythm, count in sorted(note_counts.items())
        ]
    )

    metadata.to_csv(output_dir / "record_metadata.csv", index=False)
    file_inventory.to_csv(output_dir / "metadata_file_inventory.csv", index=False)
    manifest.to_csv(output_dir / "pilot_manifest.csv", index=False)
    notes.to_csv(output_dir / "rhythm_note_counts.csv", index=False)
    summary = write_summary(
        output_dir,
        metadata_dir,
        metadata,
        manifest,
        file_inventory,
        notes,
    )
    print(f"Saved LTAFDB audit: {output_dir}")
    print(
        f"records={len(metadata)}, pilot={len(manifest)}, "
        f"signals_downloaded=0, audit_pass={summary['audit_pass']}, "
        f"elapsed_sec={time.time() - start:.1f}"
    )


if __name__ == "__main__":
    main()
