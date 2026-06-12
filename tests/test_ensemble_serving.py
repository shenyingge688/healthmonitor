"""
Regression test for the official ensemble serving path.

Compares one formal evaluation batch against the FastAPI inference helper. The
same batch size is used because CUDA convolution kernels can differ slightly
between batch sizes.
"""
import numpy as np
import pandas as pd
import torch

import pytest

import healthmonitor.main as main
from healthmonitor.demo_replay import DEMO_CASES, load_case_signal
from healthmonitor.paths import PROJECT_ROOT


CLASS_KEYS = ("normal", "pvc", "afib", "vf", "vt", "at_svt")
MEMBER_OUTPUTS = (
    "artifacts/evidence/seed0_val_outputs.csv",
    "artifacts/evidence/seed1_val_outputs.csv",
    "artifacts/evidence/seed4_val_outputs.csv",
)
ENSEMBLE_OUTPUT = "artifacts/evidence/ensemble_seed014_val_outputs.csv"


def main_test():
    batch_size = 64
    shard = torch.load(
        "dataset_v7/val_shard_000.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    bx = shard["X"][:batch_size].to(main.device, dtype=torch.float32)
    bx_rr = shard["X_rr"][:batch_size].to(main.device, dtype=torch.float32)
    output = main.run_ensemble(bx, bx_rr)

    probability_columns = [f"prob_fut_{key}" for key in CLASS_KEYS]
    expected_probs = pd.read_csv(
        ENSEMBLE_OUTPUT,
        nrows=batch_size,
    )[probability_columns].to_numpy(dtype=float)
    served_probs = output["probs_fut"].cpu().numpy()
    probability_error = float(np.max(np.abs(served_probs - expected_probs)))
    if probability_error >= 2e-6:
        raise AssertionError(
            f"ensemble probability regression failed: {probability_error}"
        )

    expected_member_risks = []
    for path in MEMBER_OUTPUTS:
        rows = pd.read_csv(path, nrows=batch_size)
        expected_member_risks.append(
            1.0 - rows["prob_fut_normal"].to_numpy(dtype=float)
        )
    expected_member_risks = np.stack(expected_member_risks, axis=1)
    served_member_risks = output["member_risks"].cpu().numpy()
    member_error = float(
        np.max(np.abs(served_member_risks - expected_member_risks))
    )
    if member_error >= 2e-6:
        raise AssertionError(f"member-risk regression failed: {member_error}")

    if set(output["confidence"]) - {"high", "medium", "low"}:
        raise AssertionError("unexpected confidence label")
    if any(vote < 1 or vote > len(main.models) for vote in output["future_vote_count"]):
        raise AssertionError("invalid ensemble vote count")
    if not torch.isfinite(output["risk_std"]).all():
        raise AssertionError("non-finite risk disagreement")

    health = main.health()
    if health["ensemble_size"] != 3:
        raise AssertionError(f"unexpected ensemble size: {health}")
    if health["policy_config"]["threshold"] != 0.10:
        raise AssertionError(f"unexpected policy config: {health}")

    payload = main.ECGPayload(ecg=[0.0] * main.API_BUFFER_SIZE)
    with pytest.raises(main.HTTPException) as exc_info:
        main.predict(payload)
    assert exc_info.value.status_code == 422

    signal, _, _ = load_case_signal(DEMO_CASES["100"], str(PROJECT_ROOT))
    sample_payload = main.ECGPayload(ecg=signal[: main.HISTORY_PTS].tolist())
    response = main.predict(sample_payload)
    required_new_fields = {
        "overall_risk_raw",
        "afib_direction_score",
        "signal_quality",
        "agreement_level",
        "agreement_note",
        "policy_config_version",
    }
    if not required_new_fields.issubset(response):
        raise AssertionError(
            f"missing API fields: {required_new_fields - set(response)}"
        )
    if abs(response["overall_risk_raw"] - response["risk_score"]) >= 1e-12:
        raise AssertionError("overall_risk_raw must preserve risk_score semantics")
    short_payload = main.ECGPayload(ecg=[0.1] * (main.HISTORY_PTS - 1))
    with pytest.raises(main.HTTPException) as short_exc:
        main.predict(short_payload)
    assert short_exc.value.status_code == 422

    print(
        "Ensemble serving regression passed: "
        f"prob_max_error={probability_error:.3e}, "
        f"member_risk_max_error={member_error:.3e}"
    )


if __name__ == "__main__":
    main_test()
