"""Aggregate detection-training results across all experiments × seeds.

Reads experiments/detection_training/*/seed*/epoch_metrics.csv and saved
predictions, computes per-class AP (Method D) and writes a summary table.
"""
import os
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[2]))
BASE_DIR = ROOT / "experiments" / "detection_training"

sys.path.insert(0, str(ROOT / "analysis"))
from lesion_level_eval import evaluate_method_D, compute_class_AP_coco
from recompute_metrics import compute_metrics_from_predictions


def crit_v3(metrics):
    return (metrics.get("auc_left_abnormal", 0) + metrics.get("auc_right_abnormal", 0) +
            metrics.get("auc_left_cyst", 0) + metrics.get("auc_right_cyst", 0)) / 4


def evaluate_seed(seed_dir: Path):
    """For one seed, find best epoch by v3 crit and compute Method D."""
    csv_path = seed_dir / "epoch_metrics.csv"
    pred_dir = seed_dir / "predictions"
    if not csv_path.exists() or not pred_dir.exists():
        return None

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None

    # Best by v3 crit
    def crit_row(r):
        try:
            return (float(r.get("auc_left_abnormal", 0)) + float(r.get("auc_right_abnormal", 0)) +
                    float(r.get("auc_left_cyst", 0)) + float(r.get("auc_right_cyst", 0))) / 4
        except (ValueError, TypeError):
            return 0.0
    best_row = max(rows, key=crit_row)
    best_ep = int(best_row["epoch"])

    npz_path = pred_dir / f"epoch_{best_ep}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    predictions = {k: data[k] for k in data.files}

    # Method D: per-class AP (COCO-greedy, default)
    method_d = evaluate_method_D(predictions, size_tol=1.0, matching="coco")
    # Also compute for other tolerances (consistent: use COCO for all)
    ap_per_tol = {}
    for tol in [0.5, 1.0, 2.0]:
        ap_per_tol[f"{tol}cm"] = {
            c: compute_class_AP_coco(predictions, c, size_tol=tol)["AP"]
            for c in ["cyst", "mass", "tumor", "solid"]
        }

    return {
        "best_epoch": best_ep,
        "side_crit_v3": crit_row(best_row),
        "AP_cyst": method_d["per_class"]["cyst"]["AP"],
        "AP_mass": method_d["per_class"]["mass"]["AP"],
        "AP_tumor": method_d["per_class"]["tumor"]["AP"],
        "AP_solid": method_d["per_class"]["solid"]["AP"],
        "mAP": method_d["mAP_unweighted"],
        "ap_per_tol": ap_per_tol,
    }


def aggregate_experiment(exp_dir: Path):
    """Aggregate over seeds within one experiment."""
    seed_dirs = sorted(exp_dir.glob("seed*"))
    per_seed = []
    for sd in seed_dirs:
        r = evaluate_seed(sd)
        if r is not None:
            m = re.search(r"seed(\d+)", sd.name)
            r["seed"] = int(m.group(1)) if m else -1
            per_seed.append(r)

    if not per_seed:
        return None

    def agg(key):
        vals = [r[key] for r in per_seed if key in r]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    return {
        "exp": exp_dir.name,
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "side_crit_v3": agg("side_crit_v3"),
        "AP_cyst": agg("AP_cyst"),
        "AP_mass": agg("AP_mass"),
        "AP_tumor": agg("AP_tumor"),
        "AP_solid": agg("AP_solid"),
        "mAP": agg("mAP"),
    }


def main():
    results = []
    for exp_dir in sorted(BASE_DIR.iterdir()):
        if not exp_dir.is_dir(): continue
        r = aggregate_experiment(exp_dir)
        if r is not None:
            results.append(r)
            print(f"✓ {exp_dir.name}: n_seeds={r['n_seeds']}, "
                  f"mAP={r['mAP']['mean']:.3f}±{r['mAP']['std']:.3f}, "
                  f"AP_cyst={r['AP_cyst']['mean']:.3f}±{r['AP_cyst']['std']:.3f}")

    out_json = BASE_DIR / "results_summary.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Markdown table
    md_lines = [
        "# Detection-Training Results Summary",
        "",
        f"Total experiments: {len(results)}",
        "",
        "## Per-class AP (mean ± std over 3 seeds, size_tol=1cm)",
        "",
        "| Experiment | side_crit | AP_cyst | AP_mass | AP_tumor | AP_solid | **mAP** |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        def f(k):
            a = r[k]
            return f"{a['mean']:.3f}±{a['std']:.3f}"
        md_lines.append(f"| {r['exp']} | {f('side_crit_v3')} | {f('AP_cyst')} | "
                        f"{f('AP_mass')} | {f('AP_tumor')} | {f('AP_solid')} | **{f('mAP')}** |")

    out_md = BASE_DIR / "results_summary.md"
    out_md.write_text("\n".join(md_lines))
    print(f"\nSaved: {out_json}\nSaved: {out_md}")


if __name__ == "__main__":
    main()
