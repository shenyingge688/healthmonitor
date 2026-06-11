"""
Calibrate descriptive ensemble-disagreement bands from validation outputs.

This does not calibrate clinical probability or guarantee correctness. It sets
display thresholds for model-to-model disagreement using validation quantiles.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_KEYS = ("normal", "pvc", "afib", "vf", "vt", "at_svt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-csv", nargs="+", required=True)
    ap.add_argument("--member-name", nargs="+", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--high-quantile", type=float, default=0.50)
    ap.add_argument("--medium-quantile", type=float, default=0.90)
    args = ap.parse_args()

    if len(args.member_csv) != len(args.member_name):
        raise ValueError("--member-csv and --member-name must have equal length")
    if len(args.member_csv) < 2:
        raise ValueError("at least two ensemble members are required")

    members = [pd.read_csv(path) for path in args.member_csv]
    reference = members[0]
    alignment_columns = ("patient_id", "window_id", "target_cur", "target_fut")
    for member_i, member in enumerate(members[1:], start=1):
        if len(member) != len(reference):
            raise RuntimeError(
                f"row-count mismatch for member {args.member_name[member_i]}"
            )
        for column in alignment_columns:
            if not np.array_equal(
                reference[column].to_numpy(),
                member[column].to_numpy(),
            ):
                raise RuntimeError(
                    f"alignment mismatch in {column} for "
                    f"member {args.member_name[member_i]}"
                )

    probability_columns = [f"prob_fut_{key}" for key in CLASS_KEYS]
    member_probabilities = np.stack(
        [member[probability_columns].to_numpy(dtype=float) for member in members],
        axis=1,
    )
    member_risks = 1.0 - member_probabilities[:, :, 0]
    risk_std = np.std(member_risks, axis=1, ddof=0)
    member_classes = np.argmax(member_probabilities, axis=2)
    vote_count = np.max(
        np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(CLASS_KEYS)),
            1,
            member_classes,
        ),
        axis=1,
    )

    high_threshold = float(np.quantile(risk_std, args.high_quantile))
    medium_threshold = float(np.quantile(risk_std, args.medium_quantile))
    high_mask = (risk_std <= high_threshold) & (vote_count == len(members))
    medium_mask = (
        ~high_mask
        & (risk_std <= medium_threshold)
        & (vote_count >= (len(members) // 2 + 1))
    )
    low_mask = ~(high_mask | medium_mask)

    config = {
        "source_split": "validation",
        "member_names": args.member_name,
        "member_csvs": args.member_csv,
        "ensemble_size": len(members),
        "risk_definition": "1 - future_normal_probability_per_member",
        "risk_std_ddof": 0,
        "confidence_semantics": (
            "descriptive model agreement only; not a clinical confidence "
            "interval or correctness probability"
        ),
        "high": {
            "max_risk_std": high_threshold,
            "required_vote_count": len(members),
            "validation_fraction": float(np.mean(high_mask)),
        },
        "medium": {
            "max_risk_std": medium_threshold,
            "required_vote_count": len(members) // 2 + 1,
            "validation_fraction": float(np.mean(medium_mask)),
        },
        "low": {
            "validation_fraction": float(np.mean(low_mask)),
        },
        "validation_risk_std_quantiles": {
            "q10": float(np.quantile(risk_std, 0.10)),
            "q25": float(np.quantile(risk_std, 0.25)),
            "q50": float(np.quantile(risk_std, 0.50)),
            "q75": float(np.quantile(risk_std, 0.75)),
            "q90": float(np.quantile(risk_std, 0.90)),
            "q95": float(np.quantile(risk_std, 0.95)),
        },
        "validation_vote_fractions": {
            "unanimous": float(np.mean(vote_count == len(members))),
            "majority_only": float(
                np.mean(
                    (vote_count >= (len(members) // 2 + 1))
                    & (vote_count < len(members))
                )
            ),
            "no_majority": float(
                np.mean(vote_count < (len(members) // 2 + 1))
            ),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(
        f"Saved uncertainty config: {output_path} "
        f"(high<={high_threshold:.4f}, medium<={medium_threshold:.4f})"
    )


if __name__ == "__main__":
    main()
