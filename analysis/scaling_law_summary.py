"""
Aggregate scaling law multi-seed experiments.

Reads epoch_metrics.csv from each (fraction, seed) experiment in
experiments/scaling_law_multiseed/, computes:
  1. last_5_avg crit (more stable than best epoch)
  2. mean ± std across seeds for each fraction

Output:
  - JSON summary
  - Markdown table
"""
import os
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_BASE = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[1])) / "experiments" / "scaling_law_multiseed"
OUT_DIR = EXP_BASE


def compute_crit(row):
    """L3D criterion: mean of 4 stable side-level AUCs."""
    return (
        float(row.get("auc_left_abnormal", 0))
        + float(row.get("auc_right_abnormal", 0))
        + float(row.get("auc_left_cyst", 0))
        + float(row.get("auc_right_cyst", 0))
    ) / 4


def parse_exp_name(name):
    """Extract fraction and seed from exp name like '..._frac10_seed42_TIMESTAMP'."""
    frac_match = re.search(r"frac(\d+)_seed(\d+)", name)
    if not frac_match:
        return None, None
    return int(frac_match.group(1)), int(frac_match.group(2))


def load_experiment(exp_dir):
    csv_path = exp_dir / "epoch_metrics.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    crits = [compute_crit(r) for r in rows]
    return {
        "n_epochs": len(rows),
        "best_crit": float(max(crits)),
        "best_epoch": int(rows[crits.index(max(crits))]["epoch"]),
        "final_crit": float(crits[-1]),
        "last5_avg": float(np.mean(crits[-5:])),
        "last5_std": float(np.std(crits[-5:])),
    }


def main():
    # Collect all experiments
    by_frac = defaultdict(list)  # frac → list of {seed, ...metrics}
    for exp_dir in sorted(EXP_BASE.iterdir()):
        if not exp_dir.is_dir():
            continue
        frac, seed = parse_exp_name(exp_dir.name)
        if frac is None:
            continue
        result = load_experiment(exp_dir)
        if result is None:
            continue
        result["seed"] = seed
        result["exp_name"] = exp_dir.name
        by_frac[frac].append(result)

    # Aggregate
    summary = []
    for frac in sorted(by_frac):
        runs = by_frac[frac]
        last5_vals = [r["last5_avg"] for r in runs]
        best_vals = [r["best_crit"] for r in runs]
        summary.append({
            "fraction": frac,
            "n_seeds": len(runs),
            "last5_mean": float(np.mean(last5_vals)),
            "last5_std": float(np.std(last5_vals)),
            "best_mean": float(np.mean(best_vals)),
            "best_std": float(np.std(best_vals)),
            "runs": runs,
        })

    # Save JSON
    out_json = OUT_DIR / "scaling_law_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    # Print markdown table
    print()
    print("| Fraction | Seeds | Last5 Avg (mean ± std) | Best Crit (mean ± std) |")
    print("|----------|-------|------------------------|------------------------|")
    for s in summary:
        print(
            f"| {s['fraction']}% | {s['n_seeds']} | "
            f"{s['last5_mean']:.4f} ± {s['last5_std']:.4f} | "
            f"{s['best_mean']:.4f} ± {s['best_std']:.4f} |"
        )


if __name__ == "__main__":
    main()
