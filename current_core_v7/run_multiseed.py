"""
run_multiseed.py
Run train_trajectory.py across several fixed seeds (fresh subprocess each) and
aggregate the per-seed summaries into mean +/- std. Resolves the run-to-run
variance question: which signals are stable (AUROC) vs unstable (argmax-F1)?

Usage:  python run_multiseed.py --seeds 0 1 2 3 4
"""
import os, sys, json, subprocess, argparse
import numpy as np

KEYS = ["f1_cur", "f1_fut", "auroc_fut", "recall_rare", "transition_recall", "ece_fut"]


def agg(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std())


def load_summaries(seeds, output_root="models"):
    out = []
    for s in seeds:
        p = os.path.join(output_root, "seeds", f"seed{s}", "summary.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        else:
            print(f"  [warn] missing {p}")
    return out


def report(summaries, output_root="models"):
    if not summaries:
        print("No summaries found.")
        return
    print("\n" + "=" * 72)
    print(f"MULTI-SEED AGGREGATE  (n={len(summaries)} seeds: "
          f"{[s['seed'] for s in summaries]})")
    print("=" * 72)

    for view, label in [("best_composite_epoch", "BEST-COMPOSITE checkpoint (selection rule)"),
                        ("final_epoch", "FINAL epoch (ep60)")]:
        print(f"\n--- {label} ---")
        print(f"{'metric':<20}{'mean':>8}{'std':>8}   per-seed")
        for k in KEYS:
            vals = [s[view][k] for s in summaries]
            m, sd = agg(vals)
            per = " ".join(f"{v:.3f}" for v in vals)
            print(f"{k:<20}{m:>8.3f}{sd:>8.3f}   [{per}]")

    print("\n--- PER-METRIC PEAK across the 60-epoch run (best epoch each seed) ---")
    print(f"{'metric':<20}{'mean':>8}{'std':>8}   per-seed (value@epoch)")
    peak_keys = [("auroc_fut", "auroc_fut"), ("f1_fut", "f1_fut"),
                 ("f1_cur", "f1_cur"), ("recall_rare", "recall_rare"),
                 ("transition_recall", "transition_recall"), ("ece_fut_min", "ece_fut_min")]
    for label, pk in peak_keys:
        vals = [s["peaks"][pk][0] for s in summaries]
        eps = [s["peaks"][pk][1] for s in summaries]
        m, sd = agg(vals)
        per = " ".join(f"{v:.3f}@{e}" for v, e in zip(vals, eps))
        print(f"{label:<20}{m:>8.3f}{sd:>8.3f}   [{per}]")

    agg_out = {
        "seeds": [s["seed"] for s in summaries],
        "best_composite": {k: agg([s["best_composite_epoch"][k] for s in summaries]) for k in KEYS},
        "final_epoch": {k: agg([s["final_epoch"][k] for s in summaries]) for k in KEYS},
        "peaks": {pk: agg([s["peaks"][pk][0] for s in summaries])
                  for pk in ["auroc_fut", "f1_fut", "f1_cur", "recall_rare",
                             "transition_recall", "ece_fut_min"]},
    }
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, "multiseed_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg_out, f, indent=2)
    print(f"\nAggregate -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--report-only", action="store_true",
                    help="skip training, just aggregate existing summaries")
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--output-root", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    if not args.report_only:
        for s in args.seeds:
            print(f"\n{'#'*72}\n# SEED {s}\n{'#'*72}", flush=True)
            r = subprocess.run([
                sys.executable, "train_trajectory.py",
                "--seed", str(s),
                "--dataset-dir", args.dataset_dir,
                "--output-root", args.output_root,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ])
            if r.returncode != 0:
                print(f"[error] seed {s} exited with {r.returncode}")

    report(load_summaries(args.seeds, output_root=args.output_root), output_root=args.output_root)


if __name__ == "__main__":
    main()
