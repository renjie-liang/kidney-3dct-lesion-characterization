"""Compare Hungarian-on-size vs COCO-greedy-by-confidence matching protocols.

Uses existing saved predictions (no re-training). Compares per-class AP and mAP
across both protocols on core_matrix L3D experiments + head ablations.
"""
import os
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from lesion_level_eval import (
    crit_v3,
    evaluate_method_D,
    evaluate_method_D_coco,
)

PROJECT_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[1]))


def evaluate_experiment(exp_dir: Path):
    """Run both matchings on the best-epoch predictions of one experiment."""
    v3 = exp_dir / "epoch_metrics_v3.csv"
    if not v3.exists():
        return None
    rows = list(csv.DictReader(open(v3)))
    if not rows: return None
    best = max(rows, key=crit_v3)
    ep = int(best["epoch"])
    npz = exp_dir / "predictions" / f"epoch_{ep}.npz"
    if not npz.exists(): return None

    d = np.load(npz)
    preds = {k: d[k] for k in d.files}

    hung = evaluate_method_D(preds, size_tol=1.0)
    coco = evaluate_method_D_coco(preds, size_tol=1.0)

    return {
        "exp": exp_dir.name,
        "epoch": ep,
        "hungarian": hung,
        "coco": coco,
    }


def fmt_cell(hung_v, coco_v):
    diff = coco_v - hung_v
    sign = "+" if diff >= 0 else ""
    return f"{hung_v:.3f} / {coco_v:.3f} ({sign}{diff:.3f})"


def main():
    # All L3D experiments we have v3 predictions for
    candidate_dirs = [
        PROJECT_ROOT / "experiments/core_matrix_v2/suprem_whole_L3D_flip_hier0.3l1_0408_0151",
        PROJECT_ROOT / "experiments/core_matrix_v2/suprem_crop_L3D_flip_hier0.3l1_0408_0254",
        PROJECT_ROOT / "experiments/core_matrix_v2/suprem_crop+kmask_L3D_flip_hier0.3l1_0408_0306",
        PROJECT_ROOT / "experiments/core_matrix_v2/suprem_crop+fmask_L3D_flip_hier0.3l1_0408_0846",
        PROJECT_ROOT / "experiments/l3_head_ablation_v2/suprem_crop+kmask_L3B_flip_hier0.3l1_0413_1714",
        PROJECT_ROOT / "experiments/l3_head_ablation_v2/suprem_crop+kmask_L3C_flip_hier0.3l1_0413_1729",
    ]

    results = []
    for d in candidate_dirs:
        r = evaluate_experiment(d)
        if r is not None:
            results.append(r)
            h = r["hungarian"]["per_class"]
            c = r["coco"]["per_class"]
            print(f"✓ {d.name[:60]}")
            print(f"    cyst: Hung={h['cyst']['AP']:.3f}  COCO={c['cyst']['AP']:.3f}")
            print(f"    mass: Hung={h['mass']['AP']:.3f}  COCO={c['mass']['AP']:.3f}")
            print(f"    tumor: Hung={h['tumor']['AP']:.3f}  COCO={c['tumor']['AP']:.3f}")
            print(f"    solid: Hung={h['solid']['AP']:.3f}  COCO={c['solid']['AP']:.3f}")
            print(f"    mAP (cyst+mass+tumor): Hung={r['hungarian']['mAP_unweighted']:.3f}  COCO={r['coco']['mAP_unweighted']:.3f}")

    # Write comparison table
    out_dir = PROJECT_ROOT / "experiments/analysis/lesion_level_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_md = out_dir / "hungarian_vs_coco.md"
    lines = [
        "# Hungarian vs COCO-greedy matching protocol comparison",
        "",
        "Same predictions, two matching protocols:",
        "- **Hungarian**: per-side bipartite matching on size cost",
        "- **COCO-greedy**: global sort by confidence, greedy match to best GT in same side",
        "",
        "Each cell: `Hung_AP / COCO_AP (diff = COCO - Hung)`. Positive diff = COCO > Hungarian.",
        "",
        "| Experiment | AP_cyst | AP_mass | AP_tumor | AP_solid | **mAP** |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        h = r["hungarian"]["per_class"]
        c = r["coco"]["per_class"]
        name = r["exp"][:45]
        lines.append(
            f"| {name} | "
            f"{fmt_cell(h['cyst']['AP'], c['cyst']['AP'])} | "
            f"{fmt_cell(h['mass']['AP'], c['mass']['AP'])} | "
            f"{fmt_cell(h['tumor']['AP'], c['tumor']['AP'])} | "
            f"{fmt_cell(h['solid']['AP'], c['solid']['AP'])} | "
            f"{fmt_cell(r['hungarian']['mAP_unweighted'], r['coco']['mAP_unweighted'])} |"
        )

    # Summary: avg diff across experiments
    lines.append("")
    lines.append("## Average difference (COCO - Hungarian) across experiments")
    lines.append("")
    avg_diffs = {}
    for cls in ["cyst", "mass", "tumor", "solid"]:
        diffs = [r["coco"]["per_class"][cls]["AP"] - r["hungarian"]["per_class"][cls]["AP"] for r in results]
        avg_diffs[cls] = {"mean": float(np.mean(diffs)), "max": float(np.max(diffs)), "min": float(np.min(diffs))}
    map_diffs = [r["coco"]["mAP_unweighted"] - r["hungarian"]["mAP_unweighted"] for r in results]
    avg_diffs["mAP"] = {"mean": float(np.mean(map_diffs)), "max": float(np.max(map_diffs)), "min": float(np.min(map_diffs))}

    lines.append("| Class | mean Δ | min Δ | max Δ |")
    lines.append("|---|---|---|---|")
    for cls, d in avg_diffs.items():
        lines.append(f"| {cls} | {d['mean']:+.4f} | {d['min']:+.4f} | {d['max']:+.4f} |")

    out_md.write_text("\n".join(lines))
    (out_dir / "hungarian_vs_coco.json").write_text(json.dumps(results, indent=2, default=str))

    print()
    print("=" * 70)
    print("Summary of (COCO - Hungarian):")
    for cls, d in avg_diffs.items():
        print(f"  {cls:<8}: mean Δ = {d['mean']:+.4f}  (range [{d['min']:+.4f}, {d['max']:+.4f}])")
    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()
