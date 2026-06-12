"""Generate deterministic 10-second frozen replay outputs for the dashboard."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import healthmonitor.main as main
from healthmonitor.constants import API_BUFFER_SIZE, CLASS_NAMES, HISTORY_PTS, TARGET_FS
from healthmonitor.demo_replay import (
    DEMO_CASES,
    count_ventricular_annotations,
    load_case_signal,
    rhythm_annotations,
)
from healthmonitor.monitoring_policy import (
    AlarmPolicyState,
    DEFAULT_POLICY,
    policy_config_dict,
    update_alarm_policy,
)


def inference_rows(signal, timestamps, batch_size=1):
    windows = []
    rr_features = []
    quality = []
    history_complete = []
    for timestamp_sec in timestamps:
        current_pt = int(round(timestamp_sec * TARGET_FS))
        start = max(0, current_pt - API_BUFFER_SIZE)
        available = signal[start:current_pt]
        arr = np.zeros(API_BUFFER_SIZE, dtype=np.float32)
        if len(available):
            arr[-len(available):] = available[-API_BUFFER_SIZE:]
        ws, rs = main.build_window_sequence(arr)
        windows.append(ws)
        rr_features.append(rs)
        quality.append(main.assess_signal_quality(arr))
        history_complete.append(current_pt >= HISTORY_PTS)

    rows = []
    for start in range(0, len(timestamps), batch_size):
        end = min(start + batch_size, len(timestamps))
        bx = torch.from_numpy(np.stack(windows[start:end])).to(
            main.device,
            dtype=torch.float32,
        )
        bx_rr = torch.from_numpy(np.stack(rr_features[start:end])).to(
            main.device,
            dtype=torch.float32,
        )
        out = main.run_ensemble(bx, bx_rr)
        probs_cur = out["probs_cur"].cpu().numpy()
        probs_fut = out["probs_fut"].cpu().numpy()
        cams = out["cam"].cpu().numpy()
        risk_std = out["risk_std"].cpu().numpy()
        for local_idx in range(end - start):
            idx = start + local_idx
            pf = probs_fut[local_idx]
            pc = probs_cur[local_idx]
            afib_score = float(pf[2] / max(pf[2] + pf[0], 1e-8))
            rows.append({
                "timestamp_sec": float(timestamps[idx]),
                "timestamp_min": float(timestamps[idx] / 60.0),
                "current_class": int(np.argmax(pc)),
                "current_name": CLASS_NAMES[int(np.argmax(pc))],
                "future_class": int(np.argmax(pf)),
                "future_name": CLASS_NAMES[int(np.argmax(pf))],
                "probabilities_cur": pc.astype(float).tolist(),
                "probabilities_fut": pf.astype(float).tolist(),
                "overall_risk_raw": float(1.0 - pf[0]),
                "afib_direction_score": afib_score,
                "risk_std": float(risk_std[local_idx]),
                "agreement_level": out["confidence"][local_idx],
                "future_vote_count": int(
                    out["future_vote_count"][local_idx]
                ),
                "signal_quality": quality[idx],
                "history_complete": bool(history_complete[idx]),
                "cam": cams[local_idx].astype(float).tolist(),
            })
    return rows


def add_policy_and_explanation(rows, record_path, src_fs):
    state = AlarmPolicyState()
    for row in rows:
        state, transition = update_alarm_policy(
            state,
            row["overall_risk_raw"],
            DEFAULT_POLICY,
        )
        row["policy"] = {
            "ewma_risk": float(state.ewma_risk),
            "consecutive_count": int(state.consecutive_count),
            "status": state.status,
            "alert_active": state.alert_active,
            "transition": transition,
        }
        cam = np.asarray(row["cam"], dtype=float)
        focus_idx = int(np.argmax(cam)) if len(cam) else 0
        history_start = row["timestamp_sec"] - 600.0
        focus_start = history_start + focus_idx * 15.0
        focus_end = focus_start + 30.0
        row["cam_focus"] = {
            "window_index": focus_idx,
            "start_sec": float(focus_start),
            "end_sec": float(focus_end),
            "ventricular_annotation_count": int(
                count_ventricular_annotations(
                    record_path,
                    src_fs,
                    max(focus_start, 0.0),
                    max(focus_end, 0.0),
                )
            ),
        }
    return rows


def generate(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(__file__).resolve().parent
    manifest = {
        "manifest_version": "round8-demo-v1",
        "generated_with": {
            "ensemble": "Round5 seed0+seed1+seed4 logits ensemble",
            "policy": policy_config_dict(),
            "inference_interval_sec": 10,
            "serving_batch_size": 1,
        },
        "cases": {},
    }
    for case_id, case in DEMO_CASES.items():
        signal, src_fs, record_path = load_case_signal(case, str(base_dir))
        start_sec = case["start_min"] * 60.0
        end_sec = case["end_min"] * 60.0
        timestamps = np.arange(start_sec, end_sec + 0.1, 10.0)
        rows = inference_rows(signal, timestamps)
        rows = add_policy_and_explanation(rows, record_path, src_fs)
        annotations = rhythm_annotations(
            record_path,
            src_fs,
            start_sec,
            end_sec,
        )

        target = int(case["target_class"])
        summary = {
            "points": len(rows),
            "official_alarm_points": int(
                sum(row["policy"]["alert_active"] for row in rows)
            ),
            "current_target_argmax_points": int(
                sum(row["current_class"] == target for row in rows)
            ),
            "future_target_argmax_points": int(
                sum(row["future_class"] == target for row in rows)
            ),
            "max_target_probability_cur": float(
                max(row["probabilities_cur"][target] for row in rows)
            ),
            "max_target_probability_fut": float(
                max(row["probabilities_fut"][target] for row in rows)
            ),
            "max_afib_direction_score": float(
                max(row["afib_direction_score"] for row in rows)
            ),
            "rhythm_annotation_count": len(annotations),
        }
        case_output = {
            **case,
            "record_path": record_path,
            "source_fs": src_fs,
            "annotations": annotations,
            "summary": summary,
            "rows": rows,
        }
        case_path = output_dir / f"mitdb_{case_id}.json"
        with case_path.open("w", encoding="utf-8") as f:
            json.dump(case_output, f, indent=2, ensure_ascii=False)
        manifest["cases"][case_id] = {
            **case,
            "replay_file": case_path.name,
            "summary": summary,
        }
        print(case_id, summary)

    manifest_path = output_dir / "replay_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results_v7_round8/demo_replays",
    )
    args = parser.parse_args()
    print(generate(args.output_dir))


if __name__ == "__main__":
    main_cli()
