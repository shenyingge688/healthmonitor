"""Acceptance checks for frozen competition demo replays."""
import json
from pathlib import Path

import numpy as np

import healthmonitor.main as main
from healthmonitor.constants import TARGET_FS
from healthmonitor.demo_replay import DEMO_CASES, load_case_signal
from healthmonitor.paths import DEMO_DIR


BASE_DIR = Path(__file__).resolve().parents[1]
REPLAY_DIR = DEMO_DIR


def main_test():
    manifest = json.loads(
        (REPLAY_DIR / "replay_manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "100": {"official_alarm_points": 0, "future_target_min": 19},
        "119": {"current_target_min": 19, "future_target_min": 19},
        "201": {"future_target_min": 18, "afib_score_min": 0.90},
        "223": {"future_target_max": 0},
        "209": {"current_target_max": 0, "future_target_max": 0},
    }
    max_probability_error = 0.0

    for case_id, metadata in manifest["cases"].items():
        replay = json.loads(
            (REPLAY_DIR / metadata["replay_file"]).read_text(encoding="utf-8")
        )
        summary = replay["summary"]
        checks = expected[case_id]
        if "official_alarm_points" in checks:
            assert summary["official_alarm_points"] == checks["official_alarm_points"]
        if "current_target_min" in checks:
            assert summary["current_target_argmax_points"] >= checks["current_target_min"]
        if "future_target_min" in checks:
            assert summary["future_target_argmax_points"] >= checks["future_target_min"]
        if "current_target_max" in checks:
            assert summary["current_target_argmax_points"] <= checks["current_target_max"]
        if "future_target_max" in checks:
            assert summary["future_target_argmax_points"] <= checks["future_target_max"]
        if "afib_score_min" in checks:
            assert summary["max_afib_direction_score"] >= checks["afib_score_min"]

        signal, _, _ = load_case_signal(DEMO_CASES[case_id], str(BASE_DIR))
        for row in replay["rows"]:
            current_pt = int(row["timestamp_sec"] * TARGET_FS)
            payload_signal = signal[max(0, current_pt - 180000):current_pt]
            if len(payload_signal) < main.HISTORY_PTS:
                payload_signal = np.pad(
                    payload_signal,
                    (main.HISTORY_PTS - len(payload_signal), 0),
                    "constant",
                )
            response = main.predict(main.ECGPayload(ecg=payload_signal.tolist()))
            error = float(
                np.max(
                    np.abs(
                        np.asarray(response["probabilities"])
                        - np.asarray(row["probabilities_fut"])
                    )
                )
            )
            max_probability_error = max(max_probability_error, error)

    if max_probability_error >= 2e-6:
        raise AssertionError(
            f"demo/API probability error={max_probability_error:.3e}"
        )
    print(
        "Demo replay acceptance passed: "
        f"max_probability_error={max_probability_error:.3e}"
    )


if __name__ == "__main__":
    main_test()
