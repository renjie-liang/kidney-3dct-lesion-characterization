"""
Recompute L3D metrics from saved predictions with fixed eval logic.

Fix: side-level cyst/mass/tumor AUC now includes normal sides (no lesion → GT=0).
Previously, normal sides were excluded, leaving only 2 negatives for cyst AUC.

Reads: predictions/epoch_*.npz from each experiment directory
Writes: epoch_metrics_v3.csv (same columns as epoch_metrics.csv but with corrected AUCs)

Usage:
    python analysis/recompute_metrics.py --exp_dirs <dir1> <dir2> ...
    python analysis/recompute_metrics.py --scan_group core_matrix_v2
"""
import os
import argparse
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[1]))
EXP_BASE = PROJECT_ROOT / "experiments"


def compute_metrics_from_predictions(npz_path):
    """Compute all L3D metrics from a saved prediction file."""
    d = np.load(npz_path)
    exists_pred = d["exists_pred"]  # (N, 6)
    exists_gt = d["exists_gt"]      # (N, 6)
    metrics = {}

    # ---- L1: side-level abnormality AUC ----
    left_abn_pred = exists_pred[:, :3].max(axis=1)
    right_abn_pred = exists_pred[:, 3:].max(axis=1)
    left_abn_gt = (exists_gt[:, :3].max(axis=1) > 0.5).astype(float)
    right_abn_gt = (exists_gt[:, 3:].max(axis=1) > 0.5).astype(float)
    try:
        metrics["auc_left_abnormal"] = float(roc_auc_score(left_abn_gt, left_abn_pred))
    except ValueError:
        metrics["auc_left_abnormal"] = 0.5
    try:
        metrics["auc_right_abnormal"] = float(roc_auc_score(right_abn_gt, right_abn_pred))
    except ValueError:
        metrics["auc_right_abnormal"] = 0.5

    # ---- L2: side-level per-type AUC (FIXED: includes normal sides) ----
    # Also computes bilateral AUC (L+R concatenated) to boost statistical power
    # for low-prevalence attributes (mass, tumor).
    side_agg_cache = {}  # {feat: {"L": (gt, pred, mask), "R": (...)}}
    for feat in ["cyst", "mass", "tumor"]:
        pred = d[f"{feat}_pred"]    # (N, 6)
        gt = d[f"{feat}_gt"]        # (N, 6)
        valid = d[f"{feat}_valid"]  # (N, 6)

        side_agg_cache[feat] = {}
        for side_name, short, sl in [("left", "L", slice(0, 3)), ("right", "R", slice(3, 6))]:
            side_exist = exists_gt[:, sl]
            side_valid = valid[:, sl]
            side_gt_vals = gt[:, sl]
            side_pred_vals = pred[:, sl]

            no_lesion = (side_exist.max(axis=1) < 0.5)
            slot_mask = (side_exist > 0.5) & (side_valid > 0.5)
            has_known = slot_mask.any(axis=1)
            include_mask = no_lesion | has_known

            side_gt_agg = np.where(no_lesion, 0.0,
                np.where(has_known,
                    np.where(slot_mask, side_gt_vals, 0).max(axis=1),
                    np.nan))
            side_pred_agg = side_pred_vals.max(axis=1)

            valid_final = include_mask & ~np.isnan(side_gt_agg)
            side_agg_cache[feat][short] = (side_gt_agg, side_pred_agg, valid_final)

            if valid_final.sum() > 10:
                try:
                    metrics[f"auc_{side_name}_{feat}"] = float(
                        roc_auc_score(side_gt_agg[valid_final], side_pred_agg[valid_final]))
                except ValueError:
                    metrics[f"auc_{side_name}_{feat}"] = 0.5
            metrics[f"n_{side_name}_{feat}_valid"] = int(valid_final.sum())
            metrics[f"npos_{side_name}_{feat}"] = int(((side_gt_agg > 0.5) & valid_final).sum())

        # Bilateral AUC: concatenate L+R samples (doubles effective positives)
        gt_L, pred_L, mask_L = side_agg_cache[feat]["L"]
        gt_R, pred_R, mask_R = side_agg_cache[feat]["R"]
        bi_gt = np.concatenate([gt_L[mask_L], gt_R[mask_R]])
        bi_pred = np.concatenate([pred_L[mask_L], pred_R[mask_R]])
        if len(bi_gt) > 10 and (bi_gt > 0.5).sum() > 0 and (bi_gt < 0.5).sum() > 0:
            try:
                metrics[f"auc_bilateral_{feat}"] = float(roc_auc_score(bi_gt, bi_pred))
            except ValueError:
                metrics[f"auc_bilateral_{feat}"] = 0.5
        metrics[f"n_bilateral_{feat}"] = len(bi_gt)
        metrics[f"npos_bilateral_{feat}"] = int((bi_gt > 0.5).sum())

    # ---- Solid = mass ∪ tumor (combines both low-prevalence classes) ----
    for side_name, short in [("left", "L"), ("right", "R")]:
        mass_gt, mass_pred, mass_mask = side_agg_cache["mass"][short]
        tumor_gt, tumor_pred, tumor_mask = side_agg_cache["tumor"][short]
        # Include sample if either mass or tumor GT is known (i.e., in either valid set)
        combined_mask = mass_mask | tumor_mask
        # GT: solid = 1 if (known mass=1) OR (known tumor=1)
        solid_gt = np.zeros(len(mass_gt))
        solid_gt[mass_mask & (mass_gt > 0.5)] = 1
        solid_gt[tumor_mask & (tumor_gt > 0.5)] = 1
        # Pred: noisy-OR-ish — max of mass/tumor predictions
        solid_pred = np.maximum(mass_pred, tumor_pred)
        if combined_mask.sum() > 10 and (solid_gt[combined_mask] > 0.5).sum() > 0 and (solid_gt[combined_mask] < 0.5).sum() > 0:
            try:
                metrics[f"auc_{side_name}_solid"] = float(
                    roc_auc_score(solid_gt[combined_mask], solid_pred[combined_mask]))
            except ValueError:
                metrics[f"auc_{side_name}_solid"] = 0.5
        metrics[f"n_{side_name}_solid"] = int(combined_mask.sum())
        metrics[f"npos_{side_name}_solid"] = int((solid_gt[combined_mask] > 0.5).sum())

    # Bilateral solid (L+R concatenated)
    L_solid_gt = np.zeros(len(side_agg_cache["mass"]["L"][0]))
    m_gt_L, m_mask_L = side_agg_cache["mass"]["L"][0], side_agg_cache["mass"]["L"][2]
    t_gt_L, t_mask_L = side_agg_cache["tumor"]["L"][0], side_agg_cache["tumor"]["L"][2]
    L_solid_gt[m_mask_L & (m_gt_L > 0.5)] = 1
    L_solid_gt[t_mask_L & (t_gt_L > 0.5)] = 1
    L_solid_pred = np.maximum(side_agg_cache["mass"]["L"][1], side_agg_cache["tumor"]["L"][1])
    L_combined = m_mask_L | t_mask_L

    R_solid_gt = np.zeros(len(side_agg_cache["mass"]["R"][0]))
    m_gt_R, m_mask_R = side_agg_cache["mass"]["R"][0], side_agg_cache["mass"]["R"][2]
    t_gt_R, t_mask_R = side_agg_cache["tumor"]["R"][0], side_agg_cache["tumor"]["R"][2]
    R_solid_gt[m_mask_R & (m_gt_R > 0.5)] = 1
    R_solid_gt[t_mask_R & (t_gt_R > 0.5)] = 1
    R_solid_pred = np.maximum(side_agg_cache["mass"]["R"][1], side_agg_cache["tumor"]["R"][1])
    R_combined = m_mask_R | t_mask_R

    bi_solid_gt = np.concatenate([L_solid_gt[L_combined], R_solid_gt[R_combined]])
    bi_solid_pred = np.concatenate([L_solid_pred[L_combined], R_solid_pred[R_combined]])
    if len(bi_solid_gt) > 10 and (bi_solid_gt > 0.5).sum() > 0 and (bi_solid_gt < 0.5).sum() > 0:
        try:
            metrics["auc_bilateral_solid"] = float(roc_auc_score(bi_solid_gt, bi_solid_pred))
        except ValueError:
            metrics["auc_bilateral_solid"] = 0.5
    metrics["n_bilateral_solid"] = len(bi_solid_gt)
    metrics["npos_bilateral_solid"] = int((bi_solid_gt > 0.5).sum())

    # Bilateral abnormal (L+R concatenated)
    bi_abn_gt = np.concatenate([left_abn_gt, right_abn_gt])
    bi_abn_pred = np.concatenate([left_abn_pred, right_abn_pred])
    if (bi_abn_gt > 0.5).sum() > 0 and (bi_abn_gt < 0.5).sum() > 0:
        try:
            metrics["auc_bilateral_abnormal"] = float(roc_auc_score(bi_abn_gt, bi_abn_pred))
        except ValueError:
            metrics["auc_bilateral_abnormal"] = 0.5
    metrics["n_bilateral_abnormal"] = len(bi_abn_gt)
    metrics["npos_bilateral_abnormal"] = int((bi_abn_gt > 0.5).sum())

    # ---- L3D: slot-level metrics ----
    # Count MAE
    metrics["count_mae"] = float(np.abs(
        (exists_pred > 0.5).sum(axis=1) - exists_gt.sum(axis=1)
    ).mean())

    # Size MAE
    exist_mask = exists_gt.flatten() > 0.5
    size_pred = d["size_pred"].flatten()
    size_gt = d["size_gt"].flatten()
    size_valid = d["size_valid"].flatten()
    size_mask = (size_valid > 0.5) & exist_mask
    if size_mask.sum() > 0:
        metrics["size_mae"] = float(np.abs(size_pred[size_mask] - size_gt[size_mask]).mean())

    # Enhancement accuracy
    enh_pred = d["enh_pred"].flatten()
    enh_gt = d["enh_gt"].flatten()
    enh_valid = d["enh_valid"].flatten()
    enh_mask = (enh_valid > 0.5) & exist_mask
    if enh_mask.sum() > 0:
        metrics["enh_acc"] = float((enh_pred[enh_mask] == enh_gt[enh_mask].round()).mean())

    # Attenuation accuracy
    att_pred = d["att_pred"].flatten()
    att_gt = d["att_gt"].flatten()
    att_valid = d["att_valid"].flatten()
    att_mask = (att_valid > 0.5) & exist_mask
    if att_mask.sum() > 0:
        metrics["att_acc"] = float((att_pred[att_mask] == att_gt[att_mask].round()).mean())

    # auc_mean
    auc_values = [v for k, v in metrics.items()
                  if k.startswith("auc_") and isinstance(v, float)]
    metrics["auc_mean"] = float(np.mean(auc_values)) if auc_values else 0.5

    return metrics


def process_experiment(exp_dir):
    """Recompute metrics for all epochs in an experiment directory."""
    pred_dir = exp_dir / "predictions"
    if not pred_dir.exists():
        print(f"  SKIP (no predictions/): {exp_dir.name}")
        return

    npz_files = sorted(pred_dir.glob("epoch_*.npz"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not npz_files:
        print(f"  SKIP (no epoch_*.npz): {exp_dir.name}")
        return

    # Only process L3D experiments (must have exists_pred key)
    test_d = np.load(npz_files[0])
    if "exists_pred" not in test_d.files:
        print(f"  SKIP (not L3D): {exp_dir.name}")
        return

    rows = []
    for npz_path in npz_files:
        epoch_num = int(npz_path.stem.split("_")[1])
        metrics = compute_metrics_from_predictions(npz_path)
        metrics["epoch"] = epoch_num
        rows.append(metrics)

    # Also read train_loss and val_loss from original csv if available
    orig_csv = exp_dir / "epoch_metrics.csv"
    if orig_csv.exists():
        orig_rows = {int(r["epoch"]): r for r in csv.DictReader(open(orig_csv))}
        for row in rows:
            orig = orig_rows.get(row["epoch"], {})
            if "train_loss" in orig:
                row["train_loss"] = float(orig["train_loss"])
            if "val_loss" in orig:
                row["val_loss"] = float(orig["val_loss"])

    # Column ordering
    priority = ["epoch", "train_loss", "val_loss", "auc_mean",
                "auc_left_abnormal", "auc_right_abnormal", "auc_bilateral_abnormal",
                "auc_left_cyst", "auc_right_cyst", "auc_bilateral_cyst",
                "auc_left_mass", "auc_right_mass", "auc_bilateral_mass",
                "auc_left_tumor", "auc_right_tumor", "auc_bilateral_tumor",
                "auc_left_solid", "auc_right_solid", "auc_bilateral_solid",
                "count_mae", "size_mae", "enh_acc", "att_acc"]
    all_cols = set()
    for row in rows:
        all_cols.update(row.keys())
    ordered = []
    for col in priority:
        if col in all_cols:
            ordered.append(col)
            all_cols.discard(col)
    ordered.extend(sorted(all_cols))

    out_path = exp_dir / "epoch_metrics_v3.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    # Report best
    def crit4(r):
        return (r.get("auc_left_abnormal", 0) + r.get("auc_right_abnormal", 0) +
                r.get("auc_left_cyst", 0) + r.get("auc_right_cyst", 0)) / 4
    best = max(rows, key=crit4)
    print(f"  {exp_dir.name}: {len(rows)} epochs → {out_path.name}, best ep{best['epoch']} crit={crit4(best):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", help="Specific experiment directories")
    parser.add_argument("--scan_group", nargs="+", help="Scan experiment groups (e.g., core_matrix_v2)")
    args = parser.parse_args()

    dirs = []
    if args.exp_dirs:
        dirs = [Path(d) for d in args.exp_dirs]
    if args.scan_group:
        for group in args.scan_group:
            group_dir = EXP_BASE / group
            if group_dir.exists():
                dirs.extend(sorted(d for d in group_dir.iterdir() if d.is_dir()))

    if not dirs:
        print("No directories specified. Use --exp_dirs or --scan_group.")
        return

    print(f"Processing {len(dirs)} experiment directories...\n")
    for d in dirs:
        process_experiment(d)


if __name__ == "__main__":
    main()
