"""
Evaluate lightweight RR gating for AFib-dominant alert-entry candidates.

The gate is deliberately narrow: it never suppresses PVC, VT, VF, or AT/SVT
dominant candidates. Selection is validation-only and preserves the accepted
Round5 risk smoothing, global threshold, and consecutive-trigger settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alert_policy import event_target_for_task, score_for_task
from evaluate_adaptive_alert_policy import (
    build_event_contexts,
    evaluate_alert_array,
    hysteresis_alert_by_patient,
    json_value,
    patient_index_groups,
)
from evaluate_smoothed_alert_policy import (
    causal_smooth_by_patient,
    infer_horizon_sec,
)


ARRHYTHMIA_PROB_COLUMNS = (
    "prob_fut_pvc",
    "prob_fut_afib",
    "prob_fut_vf",
    "prob_fut_vt",
    "prob_fut_at_svt",
)


def parse_float_list(text):
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def afib_gate_applicable(df, min_afib_share):
    arr_probs = df.loc[:, ARRHYTHMIA_PROB_COLUMNS].to_numpy(dtype=float)
    afib_index = ARRHYTHMIA_PROB_COLUMNS.index("prob_fut_afib")
    dominant = np.argmax(arr_probs, axis=1) == afib_index
    arrhythmia_mass = np.maximum(
        1.0 - df["prob_fut_normal"].to_numpy(dtype=float),
        1e-8,
    )
    afib_share = (
        df["prob_fut_afib"].to_numpy(dtype=float) / arrhythmia_mass
    )
    return dominant & (afib_share >= float(min_afib_share))


def rr_support_mask(df, mode, rmssd_min, cv_min):
    if mode == "none":
        return np.ones(len(df), dtype=bool)

    rmssd_pass = (
        df["rr_rmssd_norm"].to_numpy(dtype=float) >= float(rmssd_min)
    )
    cv_pass = df["rr_cv"].to_numpy(dtype=float) >= float(cv_min)
    if mode == "rmssd":
        return rmssd_pass
    if mode == "cv":
        return cv_pass
    if mode == "either":
        return rmssd_pass | cv_pass
    raise ValueError(f"unsupported RR gate mode: {mode}")


def evaluate_policy(
    df,
    score,
    event_target,
    patient_groups,
    event_contexts,
    task,
    source,
    threshold,
    consecutive_k,
    smoothing_method,
    smoothing_window,
    smoothing_alpha,
    gate_mode,
    min_afib_share,
    rmssd_min,
    cv_min,
    horizon_sec,
    selection,
):
    base_entry = score >= float(threshold)
    applicable = afib_gate_applicable(df, min_afib_share)
    support = rr_support_mask(df, gate_mode, rmssd_min, cv_min)
    entry = base_entry & (~applicable | support)
    alert = hysteresis_alert_by_patient(
        df=df,
        score=score,
        entry_mask=entry,
        exit_threshold=threshold,
        consecutive_k=consecutive_k,
        patient_groups=patient_groups,
    )
    metrics = evaluate_alert_array(
        df=df,
        event_target=event_target,
        alert=alert,
        horizon_sec=horizon_sec,
        event_contexts=event_contexts,
    )
    return {
        "task": task,
        "selection": selection,
        "probability_source": source,
        "threshold": float(threshold),
        "consecutive_k": int(consecutive_k),
        "smoothing_method": smoothing_method,
        "smoothing_window": int(smoothing_window),
        "smoothing_alpha": float(smoothing_alpha),
        "rr_gate_mode": gate_mode,
        "min_afib_share": float(min_afib_share),
        "rr_rmssd_norm_min": float(rmssd_min),
        "rr_cv_min": float(cv_min),
        "gated_candidate_windows": int(np.sum(base_entry & applicable & ~support)),
        "event_horizon_sec": float(horizon_sec),
        **metrics,
    }


def policy_configs(args):
    yield {
        "gate_mode": "none",
        "min_afib_share": 0.0,
        "rmssd_min": 0.0,
        "cv_min": 0.0,
    }
    shares = parse_float_list(args.min_afib_shares)
    rmssd_values = parse_float_list(args.rmssd_thresholds)
    cv_values = parse_float_list(args.cv_thresholds)
    for share in shares:
        for rmssd_min in rmssd_values:
            yield {
                "gate_mode": "rmssd",
                "min_afib_share": share,
                "rmssd_min": rmssd_min,
                "cv_min": 0.0,
            }
        for cv_min in cv_values:
            yield {
                "gate_mode": "cv",
                "min_afib_share": share,
                "rmssd_min": 0.0,
                "cv_min": cv_min,
            }
        for rmssd_min in rmssd_values:
            for cv_min in cv_values:
                yield {
                    "gate_mode": "either",
                    "min_afib_share": share,
                    "rmssd_min": rmssd_min,
                    "cv_min": cv_min,
                }


def choose_policy(grid, args):
    anchor_rows = grid[grid["rr_gate_mode"] == "none"]
    if len(anchor_rows) != 1:
        raise RuntimeError("RR gate grid must contain exactly one ungated anchor")
    anchor = anchor_rows.iloc[0]
    event_floor = max(
        float(args.min_event_recall),
        float(anchor["event_recall"]) - float(args.max_event_recall_drop),
    )
    incident_floor = max(
        float(args.min_incident_recall),
        float(anchor["incident_event_recall"])
        - float(args.max_incident_recall_drop),
    )
    false_window_limit = (
        float(anchor["false_alert_windows_per_patient_hour"])
        * (1.0 + float(args.max_false_window_increase))
    )
    feasible = grid[
        (grid["event_recall"] >= event_floor)
        & (grid["incident_event_recall"] >= incident_floor)
        & (grid["median_lead_time_sec"] >= float(args.min_median_lead_sec))
        & (
            grid["false_alert_windows_per_patient_hour"]
            <= false_window_limit
        )
    ].copy()

    if len(feasible):
        selected = feasible.sort_values(
            [
                "false_alert_episodes_per_patient_hour",
                "false_alert_windows_per_patient_hour",
                "event_recall",
                "incident_event_recall",
                "median_lead_time_sec",
                "gated_candidate_windows",
            ],
            ascending=[True, True, False, False, False, True],
        ).iloc[0].copy()
        selected["selection"] = "rr_gate_val_selected"
    else:
        selected = anchor.copy()
        selected["selection"] = "rr_gate_val_fallback_anchor"

    selected["validation_event_recall_floor"] = event_floor
    selected["validation_incident_recall_floor"] = incident_floor
    selected["validation_min_median_lead_sec"] = float(args.min_median_lead_sec)
    selected["validation_false_window_limit"] = false_window_limit
    selected["anchor_event_recall"] = float(anchor["event_recall"])
    selected["anchor_incident_event_recall"] = float(
        anchor["incident_event_recall"]
    )
    selected["anchor_median_lead_time_sec"] = float(
        anchor["median_lead_time_sec"]
    )
    selected["anchor_false_windows_per_patient_hour"] = float(
        anchor["false_alert_windows_per_patient_hour"]
    )
    selected["anchor_false_episodes_per_patient_hour"] = float(
        anchor["false_alert_episodes_per_patient_hour"]
    )
    return anchor.to_dict(), selected.to_dict(), feasible


def load_reference_policy(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = [rows]
    if len(rows) != 1:
        raise ValueError("reference policy must contain exactly one row")
    return rows[0]


def write_summary(path, anchor, selected, grid_count, feasible_count, locked):
    lines = [
        "# Round6 RR Gating",
        "",
        f"- mode: {'fixed policy application' if locked else 'validation search'}",
        f"- evaluated policies: {grid_count}",
        f"- feasible policies: {feasible_count}",
        "",
    ]
    if anchor is not None:
        lines.extend([
            "## Ungated Round5 Anchor",
            "",
            f"- event recall: {anchor['event_recall']:.3f}",
            f"- incident recall: {anchor['incident_event_recall']:.3f}",
            f"- median lead: {anchor['median_lead_time_sec']:.0f} s",
            f"- false windows/hr: {anchor['false_alert_windows_per_patient_hour']:.2f}",
            f"- false episodes/hr: {anchor['false_alert_episodes_per_patient_hour']:.2f}",
            "",
        ])
    lines.extend([
        "## Selected Policy",
        "",
        f"- RR gate mode: {selected['rr_gate_mode']}",
        f"- minimum AFib share: {selected['min_afib_share']:.2f}",
        f"- RMSSD minimum: {selected['rr_rmssd_norm_min']:.2f}",
        f"- RR-CV minimum: {selected['rr_cv_min']:.2f}",
        f"- gated candidate windows: {int(selected['gated_candidate_windows'])}",
        f"- event recall: {selected['event_recall']:.3f}",
        f"- incident recall: {selected['incident_event_recall']:.3f}",
        f"- median lead: {selected['median_lead_time_sec']:.0f} s",
        f"- false windows/hr: {selected['false_alert_windows_per_patient_hour']:.2f}",
        f"- false episodes/hr: {selected['false_alert_episodes_per_patient_hour']:.2f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task", default="future_arrhythmia")
    ap.add_argument("--probability-source", default="uncalibrated")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--consecutive-k", type=int, default=2)
    ap.add_argument("--smoothing-method", default="ewma")
    ap.add_argument("--smoothing-window", type=int, default=1)
    ap.add_argument("--smoothing-alpha", type=float, default=0.65)
    ap.add_argument("--min-afib-shares", default="0.50,0.65,0.80,0.90")
    ap.add_argument("--rmssd-thresholds", default="1.50,2.00,2.50,3.00")
    ap.add_argument("--cv-thresholds", default="0.25,0.30,0.35,0.40,0.45")
    ap.add_argument("--min-event-recall", type=float, default=0.90)
    ap.add_argument("--min-incident-recall", type=float, default=0.88)
    ap.add_argument("--min-median-lead-sec", type=float, default=260.0)
    ap.add_argument("--max-event-recall-drop", type=float, default=0.02)
    ap.add_argument("--max-incident-recall-drop", type=float, default=0.03)
    ap.add_argument("--max-false-window-increase", type=float, default=0.05)
    ap.add_argument("--horizon-sec", type=float, default=None)
    ap.add_argument("--reference-policy", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.outputs_csv).reset_index(drop=True)
    required_rr = {"rr_rmssd_norm", "rr_cv"}
    missing = sorted(required_rr - set(df.columns))
    if missing:
        raise ValueError(f"outputs CSV is missing RR columns: {missing}")

    horizon_sec = infer_horizon_sec(df, override=args.horizon_sec)
    score = causal_smooth_by_patient(
        df,
        score_for_task(df, args.task, args.probability_source),
        method=args.smoothing_method,
        window=args.smoothing_window,
        alpha=args.smoothing_alpha,
    )
    event_target = event_target_for_task(df, args.task)
    patient_groups = patient_index_groups(df)
    event_contexts = build_event_contexts(
        df,
        event_target,
        horizon_sec,
        patient_groups,
    )

    common = {
        "df": df,
        "score": score,
        "event_target": event_target,
        "patient_groups": patient_groups,
        "event_contexts": event_contexts,
        "task": args.task,
        "source": args.probability_source,
        "threshold": args.threshold,
        "consecutive_k": args.consecutive_k,
        "smoothing_method": args.smoothing_method,
        "smoothing_window": args.smoothing_window,
        "smoothing_alpha": args.smoothing_alpha,
        "horizon_sec": horizon_sec,
    }
    if args.reference_policy:
        ref = load_reference_policy(args.reference_policy)
        selected = evaluate_policy(
            **common,
            gate_mode=ref["rr_gate_mode"],
            min_afib_share=float(ref["min_afib_share"]),
            rmssd_min=float(ref["rr_rmssd_norm_min"]),
            cv_min=float(ref["rr_cv_min"]),
            selection="fixed_rr_gate_val_policy",
        )
        grid = pd.DataFrame([selected])
        anchor = None
        feasible_count = 0
    else:
        rows = []
        for config in policy_configs(args):
            rows.append(
                evaluate_policy(
                    **common,
                    selection="rr_gate_grid",
                    **config,
                )
            )
        grid = pd.DataFrame(rows)
        anchor, selected, feasible = choose_policy(grid, args)
        feasible_count = len(feasible)

    selected_df = pd.DataFrame([selected])
    grid.to_csv(out_dir / "rr_gating_grid.csv", index=False)
    selected_df.to_csv(out_dir / "rr_gating_selected.csv", index=False)
    with (out_dir / "rr_gating_selected.json").open("w", encoding="utf-8") as f:
        json.dump([json_value(selected)], f, indent=2, ensure_ascii=False)
    write_summary(
        out_dir / "rr_gating_summary.md",
        anchor,
        selected,
        len(grid),
        feasible_count,
        args.reference_policy is not None,
    )
    print(
        f"Selected RR gate: mode={selected['rr_gate_mode']} "
        f"share={selected['min_afib_share']:.2f} "
        f"rmssd={selected['rr_rmssd_norm_min']:.2f} "
        f"cv={selected['rr_cv_min']:.2f} "
        f"event_recall={selected['event_recall']:.3f} "
        f"incident_recall={selected['incident_event_recall']:.3f} "
        f"lead={selected['median_lead_time_sec']:.0f}s "
        f"false_ep/hr={selected['false_alert_episodes_per_patient_hour']:.2f} "
        f"false_win/hr={selected['false_alert_windows_per_patient_hour']:.2f}"
    )


if __name__ == "__main__":
    main()
