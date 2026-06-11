"""
Evaluate a validation-selected checkpoint ensemble under the existing V7 protocol.

This script reuses final_validation_analysis.py so the ensemble is evaluated with
the same patient-wise bootstrap, calibration, threshold sweep, event lead-time,
and locked-test machinery as a single checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_validation_analysis import (
    CLASS_KEYS,
    calibration_analysis,
    event_lead_time_analysis,
    load_dataset_config,
    patient_bootstrap,
    patient_robustness_analysis,
    plot_patient_timeline,
    reconstruct_val_metadata,
    record_order_from_split_plan,
    run_inference,
    save_outputs,
    threshold_analysis,
    write_summary,
)


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def combine_outputs(outputs_list, mode):
    first = outputs_list[0]
    combined = {
        "target_cur": first["target_cur"],
        "target_fut_soft": first["target_fut_soft"],
        "target_fut": first["target_fut"],
        "t_weight": first["t_weight"],
        "transition_flag": first["transition_flag"],
        "checkpoint_epoch": -1,
        "checkpoint_metrics": {},
        "checkpoint_path": "checkpoint_ensemble",
        "shard_paths": first.get("shard_paths", []),
        "dataset_dir": first.get("dataset_dir", ""),
        "split_name": first.get("split_name", ""),
    }

    for other in outputs_list[1:]:
        for key in ["target_cur", "target_fut", "transition_flag"]:
            if not np.array_equal(first[key], other[key]):
                raise RuntimeError(f"Cannot ensemble checkpoints: mismatched {key}")

    if mode == "logits":
        combined["logits_cur"] = np.mean([o["logits_cur"] for o in outputs_list], axis=0)
        combined["logits_fut"] = np.mean([o["logits_fut"] for o in outputs_list], axis=0)
        combined["probs_cur"] = softmax_np(combined["logits_cur"])
        combined["probs_fut"] = softmax_np(combined["logits_fut"])
    elif mode == "probs":
        combined["probs_cur"] = np.mean([o["probs_cur"] for o in outputs_list], axis=0)
        combined["probs_fut"] = np.mean([o["probs_fut"] for o in outputs_list], axis=0)
        # Use averaged logits only as an approximate calibration input. Selection is
        # still validation-only and locked test applies the validation temperature.
        eps = 1e-8
        combined["logits_cur"] = np.log(np.clip(combined["probs_cur"], eps, 1.0))
        combined["logits_fut"] = np.log(np.clip(combined["probs_fut"], eps, 1.0))
    else:
        raise ValueError(f"unsupported ensemble mode: {mode}")

    combined["pred_cur"] = combined["probs_cur"].argmax(axis=1)
    combined["pred_fut"] = combined["probs_fut"].argmax(axis=1)
    return combined


def run_full_evaluation(outputs, out_dir, args, reference_calibration, reference_ops):
    import final_validation_analysis as fva

    dataset_config = load_dataset_config(args.dataset_dir)
    history_sec = dataset_config.get("history_sec")
    predict_sec = dataset_config.get("predict_sec")
    if predict_sec is not None:
        fva.EARLY_WARNING_HORIZON_SEC = float(predict_sec)
        print(f"[config] dataset predict_sec={predict_sec}; event horizon={float(predict_sec):.0f}s")
    records = (
        record_order_from_split_plan(args.split_plan, args.split)
        if args.split_plan else None
    )
    metadata = reconstruct_val_metadata(
        len(outputs["target_fut"]),
        records=records,
        use_db_stride=bool(args.split_plan),
        history_sec=history_sec,
        predict_sec=predict_sec,
    )
    outputs_df, outputs_csv, outputs_npz = save_outputs(outputs, metadata, out_dir)

    observed, ci_df, boot_df = patient_bootstrap(outputs_df, n_boot=args.bootstrap, seed=args.seed)
    ci_df.to_csv(out_dir / "patient_bootstrap_ci.csv", index=False)
    boot_df.to_csv(out_dir / "patient_bootstrap_samples.csv", index=False)

    patient_df, loo_df = patient_robustness_analysis(outputs_df, out_dir)
    calibration, outputs_df = calibration_analysis(
        outputs_df,
        out_dir,
        n_bins=args.bins,
        reference_calibration=reference_calibration,
    )
    outputs_df.to_csv(outputs_csv, index=False, encoding="utf-8")
    _, _, op_df, _dca_df, dca_summary_df = threshold_analysis(
        outputs_df,
        out_dir,
        selected_source=calibration["selected_threshold_probability_source"],
        reference_ops=reference_ops,
    )
    event_summary_df, _event_df = event_lead_time_analysis(
        outputs_df,
        out_dir,
        selected_source=calibration["selected_threshold_probability_source"],
        op_df=op_df,
    )
    plot_patient_timeline(outputs_df, out_dir)

    risk_flags = {}
    with open(out_dir / "observed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(observed, f, indent=2, ensure_ascii=False)
    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    with open(out_dir / "validation_risk_flags.json", "w", encoding="utf-8") as f:
        json.dump(risk_flags, f, indent=2, ensure_ascii=False)
    event_summary_json = event_summary_df.replace({np.nan: None}).to_dict(orient="records")
    with open(out_dir / "event_lead_time_summary.json", "w", encoding="utf-8") as f:
        json.dump(event_summary_json, f, indent=2, ensure_ascii=False)

    summary_path = write_summary(
        out_dir,
        " + ".join(args.checkpoint),
        outputs,
        observed,
        ci_df,
        calibration,
        op_df,
        patient_df,
        loo_df,
        event_summary_df,
        dca_summary_df,
        risk_flags,
        outputs_csv,
        outputs_npz,
    )
    return observed, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="dataset_v7")
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-plan", default=None)
    parser.add_argument("--reference-calibration", default=None)
    parser.add_argument("--reference-operating-points", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--ensemble-mode", choices=["logits", "probs"], default="logits")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_calibration = None
    reference_ops = None
    if args.reference_calibration:
        with open(args.reference_calibration, "r", encoding="utf-8") as f:
            reference_calibration = json.load(f)
    if args.reference_operating_points:
        reference_ops = pd.read_csv(args.reference_operating_points)

    outputs_list = []
    for checkpoint in args.checkpoint:
        print(f"[ensemble] running inference: {checkpoint}")
        outputs_list.append(
            run_inference(
                checkpoint,
                batch_size=args.batch_size,
                dataset_dir=args.dataset_dir,
                split_name=args.split,
            )
        )

    print(f"[ensemble] combining {len(outputs_list)} checkpoints via {args.ensemble_mode}")
    outputs = combine_outputs(outputs_list, mode=args.ensemble_mode)
    observed, summary_path = run_full_evaluation(
        outputs,
        out_dir,
        args,
        reference_calibration=reference_calibration,
        reference_ops=reference_ops,
    )
    print(
        "Done. "
        f"future_macro_f1={observed['future_macro_f1']:.4f} "
        f"future_macro_auroc={observed['future_macro_auroc']:.4f} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
