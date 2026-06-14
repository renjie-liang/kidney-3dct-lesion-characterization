"""Compute per-class AP noise floor using random scoring.

Takes a val predictions npz (from any trained model), replaces the
confidence values with uniform random, and computes AP. Averaged over
multiple random seeds for stability.

Usage:
    python random_baseline.py <path/to/epoch_N.npz> [--n-trials 100]

Also computes two additional baselines:
  - "prior"  : confidence = class prior rate (constant, same for every pred)
  - "size"   : confidence = normalized size_pred (size-only proxy)

Writes a summary table to stdout.
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np

STAGE1_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(STAGE1_ROOT / "detection_training"))
from analysis.lesion_level_eval import compute_class_AP_coco


def set_conf(predictions, class_name, new_conf):
    """Return a shallow copy of predictions with class_c_pred replaced by new_conf.

    new_conf has shape (N, 6) and is combined with exists_pred to form
    the final confidence (exists * class). So to effectively set the full
    confidence, we set exists_pred = 1 and class_pred = new_conf.
    """
    d = dict(predictions)
    d["exists_pred"] = np.ones_like(d["exists_pred"])
    d[f"{class_name}_pred"] = new_conf
    return d


def run(predictions, class_name, trials=100, rng=None):
    rng = rng or np.random.default_rng(42)
    base_shape = predictions["exists_pred"].shape
    aps_random = []
    for t in range(trials):
        rc = rng.uniform(0, 1, size=base_shape)
        d = set_conf(predictions, class_name, rc)
        aps_random.append(compute_class_AP_coco(d, class_name, size_tol=1.0)["AP"])

    # Prior baseline: constant confidence = class positive rate
    gt = predictions[f"{class_name}_gt"]
    valid = predictions[f"{class_name}_valid"]
    exists_gt = predictions["exists_gt"]
    positive_rate = float(((gt > 0.5) & (valid > 0.5) & (exists_gt > 0.5)).mean())
    const_conf = np.full(base_shape, 0.5)
    d = set_conf(predictions, class_name, const_conf)
    ap_prior = compute_class_AP_coco(d, class_name, size_tol=1.0)["AP"]

    # Size-only proxy: confidence = normalized size_pred
    size_pred = predictions["size_pred"]
    size_norm = size_pred / (size_pred.max() + 1e-6)
    d = set_conf(predictions, class_name, size_norm)
    ap_size = compute_class_AP_coco(d, class_name, size_tol=1.0)["AP"]

    return {
        "class": class_name,
        "positive_rate": positive_rate,
        "random_mean": float(np.mean(aps_random)),
        "random_std": float(np.std(aps_random)),
        "random_ci_low": float(np.percentile(aps_random, 2.5)),
        "random_ci_high": float(np.percentile(aps_random, 97.5)),
        "prior_baseline": ap_prior,
        "size_only_baseline": ap_size,
        "n_trials": trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str, help="path to epoch_*.npz")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--classes", nargs="+", default=["cyst", "solid"])
    args = parser.parse_args()

    predictions = dict(np.load(args.npz_path))
    print(f"Val N = {predictions['exists_pred'].shape[0]} patients × 6 slots")
    print(f"Trials = {args.n_trials}\n")

    print(f"{'class':8} {'pos_rate':>10} {'random_AP':>12} {'±SD':>8} {'95% CI':>16} {'prior=0.5':>11} {'size_only':>11}")
    print("-" * 86)
    for cls in args.classes:
        r = run(predictions, cls, trials=args.n_trials)
        ci = f"[{r['random_ci_low']:.3f},{r['random_ci_high']:.3f}]"
        print(f"{r['class']:8} {r['positive_rate']:>10.3f} {r['random_mean']:>12.3f} "
              f"{r['random_std']:>8.3f} {ci:>16} {r['prior_baseline']:>11.3f} {r['size_only_baseline']:>11.3f}")


if __name__ == "__main__":
    main()
