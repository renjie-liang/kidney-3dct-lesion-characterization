"""
Lesion-level evaluation for L3D models — detection + per-lesion attributes.

Moves beyond side-level AUC (which aggregates all slots via max) to proper
object-detection-style metrics where prediction and GT lesion sets are matched
via Hungarian assignment on size cost.

Three protocols:
  Method A: Hungarian + fixed threshold (conf=0.5, |Δsize|<1cm tolerance)
  Method B: Hungarian + AP curve (sweep confidence threshold, integrate PR)
  Method C: Full mAP framework (multiple tolerances, per-attribute AP)

Inputs: saved predictions/epoch_*.npz
Outputs: JSON + markdown table

Usage:
  python analysis/lesion_level_eval.py --exp_dirs <dir1> <dir2> ...
"""
import os
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def crit_v3(r):
    return (float(r["auc_left_abnormal"]) + float(r["auc_right_abnormal"])
            + float(r["auc_left_cyst"]) + float(r["auc_right_cyst"])) / 4


DEFAULT_COST_WEIGHTS = {
    "w_size":  1.0,
    "w_cyst":  1.0,
    "w_mass":  2.0,
    "w_tumor": 3.0,
    "size_scale": 10.0,   # cm → [0,1] normalizer for size diff
}


def compute_class_AP(predictions, class_name, size_tol=1.0):
    """COCO-style per-class Average Precision.

    For class c ∈ {cyst, mass, tumor}:
      - Candidate preds = all 6 slots per patient, with confidence = exists_pred * type_c_pred
      - GT positives = slots where exists_gt=1 AND type_c_gt=1 AND type_c_valid=1
      - Slots with type_c_valid=0 are excluded from GT (unknown)
      - Match per side (3 slots) via Hungarian on size cost
      - TP iff matched AND |Δsize| < size_tol
      - Sort all (pred_conf, is_tp) pairs descending, compute 11-point AP

    Special 'solid' class = mass ∪ tumor (union of positive labels).

    Returns: dict with AP, n_pos, n_pred, etc.
    """
    ex_p = predictions["exists_pred"]
    ex_g = predictions["exists_gt"]
    sz_p = predictions["size_pred"]
    sz_g = predictions["size_gt"]
    N = ex_p.shape[0]

    if class_name == "solid":
        c_p = np.maximum(predictions["mass_pred"], predictions["tumor_pred"])
        g_mass = predictions["mass_gt"] * predictions["mass_valid"]
        g_tumor = predictions["tumor_gt"] * predictions["tumor_valid"]
        c_g = np.maximum(g_mass, g_tumor)
        c_v = np.maximum(predictions["mass_valid"], predictions["tumor_valid"])
    else:
        c_p = predictions[f"{class_name}_pred"]
        c_g = predictions[f"{class_name}_gt"]
        c_v = predictions[f"{class_name}_valid"]

    confidences = []
    is_tp = []
    total_gt = 0

    for n in range(N):
        for sl in [slice(0, 3), slice(3, 6)]:
            p_ex = ex_p[n, sl]
            p_sz = sz_p[n, sl]
            p_c = c_p[n, sl]
            g_ex = ex_g[n, sl]
            g_sz = sz_g[n, sl]
            g_c = c_g[n, sl]
            g_v = c_v[n, sl]

            # Confidence per slot = P(exists) × P(type=c)
            slot_conf = p_ex * p_c

            # GT positives for class c: exists AND is-class-c AND valid label
            gt_mask = (g_ex > 0.5) & (g_c > 0.5) & (g_v > 0.5)
            gt_idx = np.where(gt_mask)[0]
            total_gt += len(gt_idx)

            pred_idx = np.arange(3)

            if len(gt_idx) == 0:
                # No GT of this class on this side → all preds are FP candidates
                for i in pred_idx:
                    confidences.append(float(slot_conf[i]))
                    is_tp.append(False)
                continue

            # Hungarian match on size cost (class already filtered at GT level)
            cost = np.abs(p_sz[pred_idx][:, None] - g_sz[gt_idx][None, :])
            r, c = linear_sum_assignment(cost)

            matched_p = set()
            for i, j in zip(r, c):
                p_i = pred_idx[i]
                g_j = gt_idx[j]
                size_diff = abs(p_sz[p_i] - g_sz[g_j])
                tp_flag = size_diff < size_tol
                confidences.append(float(slot_conf[p_i]))
                is_tp.append(tp_flag)
                matched_p.add(p_i)
            for i in pred_idx:
                if i not in matched_p:
                    confidences.append(float(slot_conf[i]))
                    is_tp.append(False)

    if total_gt == 0 or not confidences:
        return {"AP": 0.0, "n_pos": total_gt, "n_pred": len(confidences),
                "F1_max": 0.0, "P_at_F1": 0.0, "R_at_F1": 0.0}

    order = np.argsort(-np.array(confidences))
    tp_sorted = np.array(is_tp)[order].astype(int)
    fp_sorted = 1 - tp_sorted
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        t = t - 1e-9
        m = recalls >= t
        ap += (precisions[m].max() if m.any() else 0.0) / 11.0

    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-10)
    if len(f1s) > 0:
        best = int(np.argmax(f1s))
        f1_max = float(f1s[best])
        p_at = float(precisions[best])
        r_at = float(recalls[best])
    else:
        f1_max = p_at = r_at = 0.0

    return {
        "AP": float(ap),
        "n_pos": int(total_gt),
        "n_pred": int(len(confidences)),
        "F1_max": f1_max,
        "P_at_F1": p_at,
        "R_at_F1": r_at,
    }


def evaluate_method_D(predictions, size_tol=1.0, matching="coco"):
    """Per-class AP, COCO-greedy by default (main paper protocol).

    Args:
        matching: "coco" (default, standard COCO protocol) or "hungarian"
                  (per-side bipartite; used for comparison/ablation only)

    Classes: cyst, mass, tumor (main mAP), plus solid (= mass ∪ tumor) as extra.
    mAP = unweighted mean of AP_cyst, AP_mass, AP_tumor. Classes with no
    positives (AP=NaN) are excluded from mAP, per COCO convention.
    """
    compute_fn = compute_class_AP_coco if matching == "coco" else compute_class_AP
    per_class = {
        c: compute_fn(predictions, c, size_tol=size_tol)
        for c in ["cyst", "mass", "tumor", "solid"]
    }
    # mAP: exclude NaN (n_pos=0) classes
    valid_aps = [per_class[c]["AP"] for c in ["cyst", "mass", "tumor"]
                 if not np.isnan(per_class[c]["AP"])]
    mAP = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return {
        "method": "D",
        "matching": matching,
        "size_tol_cm": size_tol,
        "per_class": per_class,
        "mAP_unweighted": mAP,
        "mAP_classes_counted": len(valid_aps),
    }


def compute_class_AP_coco(predictions, class_name, size_tol=1.0):
    """COCO-style per-class AP using greedy matching by confidence.

    Unlike compute_class_AP (Hungarian per-side), this sorts all predictions by
    confidence descending, then greedily matches each to the best-matching GT
    (by size) in the SAME (patient, side) that hasn't been claimed yet.

    This is the standard COCO protocol and is preferred in detection papers.
    """
    ex_p = predictions["exists_pred"]
    ex_g = predictions["exists_gt"]
    sz_p = predictions["size_pred"]
    sz_g = predictions["size_gt"]
    N = ex_p.shape[0]

    if class_name == "solid":
        c_p = np.maximum(predictions["mass_pred"], predictions["tumor_pred"])
        g_mass = predictions["mass_gt"] * predictions["mass_valid"]
        g_tumor = predictions["tumor_gt"] * predictions["tumor_valid"]
        c_g = np.maximum(g_mass, g_tumor)
        c_v = np.maximum(predictions["mass_valid"], predictions["tumor_valid"])
    else:
        c_p = predictions[f"{class_name}_pred"]
        c_g = predictions[f"{class_name}_gt"]
        c_v = predictions[f"{class_name}_valid"]

    # Collect all candidate preds with global confidence ordering
    # Each candidate = (confidence, patient_idx, side_idx, slot_idx_within_side, abs_slot_idx)
    candidates = []
    # Per (patient, side), track class-c GT and which preds' abs_slot are available
    # Then greedy: high-conf preds claim best-matching GT first
    gt_by_side = {}  # (patient_idx, side_key) → list of (gt_abs_slot, gt_size)
    total_gt = 0

    for n in range(N):
        for side_key, sl in [("L", slice(0, 3)), ("R", slice(3, 6))]:
            # GT of class c in this side
            g_ex = ex_g[n, sl]
            g_c = c_g[n, sl]
            g_v = c_v[n, sl]
            gt_mask = (g_ex > 0.5) & (g_c > 0.5) & (g_v > 0.5)
            gt_abs_slots = [sl.start + int(i) for i in np.where(gt_mask)[0]]
            gt_sizes = [float(sz_g[n, abs_idx]) for abs_idx in gt_abs_slots]
            gt_by_side[(n, side_key)] = list(zip(gt_abs_slots, gt_sizes))
            total_gt += len(gt_abs_slots)

            # All 6 slots' worth on this side
            p_ex = ex_p[n, sl]
            p_c = c_p[n, sl]
            for i in range(3):
                abs_slot = sl.start + i
                conf = float(p_ex[i] * p_c[i])
                candidates.append({
                    "conf": conf,
                    "patient": n,
                    "side": side_key,
                    "abs_slot": abs_slot,
                    "pred_size": float(sz_p[n, abs_slot]),
                })

    # Sort candidates by confidence descending (global, across all images)
    candidates.sort(key=lambda x: -x["conf"])

    # For each candidate in sorted order, find best unmatched GT in same (patient, side)
    # Matched GT tracker: set of (patient, side, abs_slot)
    matched_gt_slots = set()

    is_tp_list = []
    confidences_list = []

    for cand in candidates:
        key = (cand["patient"], cand["side"])
        gts = gt_by_side[key]   # list of (gt_abs_slot, gt_size)
        # Find best (smallest |Δsize|) GT that isn't matched and is within tolerance
        best_gt = None
        best_diff = float("inf")
        for gt_abs_slot, gt_size in gts:
            if (cand["patient"], cand["side"], gt_abs_slot) in matched_gt_slots:
                continue
            diff = abs(cand["pred_size"] - gt_size)
            if diff < size_tol and diff < best_diff:
                best_diff = diff
                best_gt = gt_abs_slot
        if best_gt is not None:
            matched_gt_slots.add((cand["patient"], cand["side"], best_gt))
            is_tp_list.append(True)
        else:
            is_tp_list.append(False)
        confidences_list.append(cand["conf"])

    if total_gt == 0:
        # COCO convention: AP is undefined when there are no positive examples.
        # Return NaN so it can be filtered out of mAP (not counted as 0).
        return {"AP": float("nan"), "n_pos": 0, "n_pred": len(confidences_list),
                "F1_max": float("nan"), "P_at_F1": float("nan"), "R_at_F1": float("nan")}
    if not confidences_list:
        return {"AP": 0.0, "n_pos": total_gt, "n_pred": 0,
                "F1_max": 0.0, "P_at_F1": 0.0, "R_at_F1": 0.0}

    # Note: candidates are already sorted by confidence desc
    tp_arr = np.array(is_tp_list).astype(int)
    fp_arr = 1 - tp_arr
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # 11-point interpolated AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        t = t - 1e-9
        m = recalls >= t
        ap += (precisions[m].max() if m.any() else 0.0) / 11.0

    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-10)
    if len(f1s) > 0:
        best = int(np.argmax(f1s))
        f1_max = float(f1s[best])
        p_at = float(precisions[best])
        r_at = float(recalls[best])
    else:
        f1_max = p_at = r_at = 0.0

    return {
        "AP": float(ap),
        "n_pos": int(total_gt),
        "n_pred": int(len(confidences_list)),
        "F1_max": f1_max,
        "P_at_F1": p_at,
        "R_at_F1": r_at,
    }


def evaluate_method_D_coco(predictions, size_tol=1.0):
    """COCO-greedy variant of Method D."""
    per_class = {
        c: compute_class_AP_coco(predictions, c, size_tol=size_tol)
        for c in ["cyst", "mass", "tumor", "solid"]
    }
    mAP = float(np.mean([per_class[c]["AP"] for c in ["cyst", "mass", "tumor"]]))
    return {
        "method": "D_coco",
        "matching": "coco_greedy",
        "size_tol_cm": size_tol,
        "per_class": per_class,
        "mAP_unweighted": mAP,
        "mAP_classes": ["cyst", "mass", "tumor"],
    }


def compute_cost_matrix(pred_size, gt_size, pred_preds, gt_preds, gt_valids, weights):
    """Compute cost matrix with size + type components.

    Size cost is normalized by size_scale (cm). Type costs are absolute differences
    of sigmoid outputs for cyst/mass/tumor, weighted and only applied where GT valid.
    """
    n_pred = len(pred_size)
    n_gt = len(gt_size)
    cost = weights["w_size"] / weights["size_scale"] * np.abs(
        pred_size[:, None] - gt_size[None, :]
    )
    for feat, w_key in [("cyst", "w_cyst"), ("mass", "w_mass"), ("tumor", "w_tumor")]:
        p = pred_preds[feat]  # (n_pred,) sigmoid score
        g = gt_preds[feat]    # (n_gt,) 0/1
        v = gt_valids[feat]   # (n_gt,) 0/1
        diff = np.abs(p[:, None] - g[None, :])   # (n_pred, n_gt)
        cost = cost + weights[w_key] * diff * v[None, :]
    return cost


def hungarian_match_one_side(pred_ex, pred_sz, pred_type, gt_ex, gt_sz, gt_type, gt_valid,
                              conf_thresh=0.5, weights=None):
    """Match predicted lesions to GT lesions within one side (3 slots) via Hungarian.

    pred_ex, pred_sz: (3,) scalars
    pred_type: dict{cyst, mass, tumor} → (3,) sigmoid scores
    gt_ex, gt_sz: (3,) scalars
    gt_type: dict{cyst, mass, tumor} → (3,) 0/1
    gt_valid: dict{cyst, mass, tumor} → (3,) 0/1

    Returns: matches list [(p_i, g_j, size_diff, total_cost, pred_conf)],
             unmatched_pred, unmatched_gt
    """
    if weights is None:
        weights = DEFAULT_COST_WEIGHTS
    pred_mask = pred_ex > conf_thresh
    gt_mask = gt_ex > 0.5
    pred_idx = np.where(pred_mask)[0]
    gt_idx = np.where(gt_mask)[0]

    if len(pred_idx) == 0 and len(gt_idx) == 0:
        return [], [], []
    if len(pred_idx) == 0:
        return [], [], list(gt_idx)
    if len(gt_idx) == 0:
        return [], [(i, float(pred_ex[i])) for i in pred_idx], []

    pred_type_sel = {k: v[pred_idx] for k, v in pred_type.items()}
    gt_type_sel = {k: v[gt_idx] for k, v in gt_type.items()}
    gt_valid_sel = {k: v[gt_idx] for k, v in gt_valid.items()}

    cost = compute_cost_matrix(pred_sz[pred_idx], gt_sz[gt_idx],
                                pred_type_sel, gt_type_sel, gt_valid_sel, weights)
    r, c = linear_sum_assignment(cost)

    matched_pred = set()
    matches = []
    for i, j in zip(r, c):
        p_i = pred_idx[i]
        g_j = gt_idx[j]
        size_diff = float(abs(pred_sz[p_i] - gt_sz[g_j]))
        matches.append((p_i, g_j, size_diff, float(cost[i, j]), float(pred_ex[p_i])))
        matched_pred.add(p_i)
    unmatched_pred = [(i, float(pred_ex[i])) for i in pred_idx if i not in matched_pred]
    unmatched_gt = [g for g in gt_idx if g not in {m[1] for m in matches}]
    return matches, unmatched_pred, unmatched_gt


def evaluate_method_A(predictions, conf_thresh=0.5, size_tol=1.0, weights=None):
    """Method A: Hungarian + fixed thresholds.

    Match pred-GT via Hungarian within each side. TP iff matched and |Δsize|<tolerance.
    Unmatched pred → FP. Match with |Δsize|>=tolerance → split into FP+FN.
    Unmatched GT → FN.
    """
    if weights is None:
        weights = DEFAULT_COST_WEIGHTS
    ex_p = predictions["exists_pred"]  # (N, 6)
    ex_g = predictions["exists_gt"]
    sz_p = predictions["size_pred"]
    sz_g = predictions["size_gt"]
    pred_type = {k: predictions[f"{k}_pred"] for k in ["cyst", "mass", "tumor"]}
    gt_type   = {k: predictions[f"{k}_gt"]    for k in ["cyst", "mass", "tumor"]}
    gt_valid  = {k: predictions[f"{k}_valid"] for k in ["cyst", "mass", "tumor"]}
    N = ex_p.shape[0]

    tp = fp = fn = 0
    # Per-attribute accuracy on TP
    attr_totals = {a: {"correct": 0, "total": 0} for a in ["cyst", "mass", "tumor"]}
    size_errors = []
    enh_correct = 0; enh_total = 0
    att_correct = 0; att_total = 0

    for n in range(N):
        for sl, side_name in [(slice(0, 3), "L"), (slice(3, 6), "R")]:
            p_t = {k: pred_type[k][n, sl] for k in ["cyst", "mass", "tumor"]}
            g_t = {k: gt_type[k][n, sl]   for k in ["cyst", "mass", "tumor"]}
            g_v = {k: gt_valid[k][n, sl]  for k in ["cyst", "mass", "tumor"]}
            matches, un_p, un_g = hungarian_match_one_side(
                ex_p[n, sl], sz_p[n, sl], p_t,
                ex_g[n, sl], sz_g[n, sl], g_t, g_v,
                conf_thresh=conf_thresh, weights=weights)
            for p_i, g_j, sd, _total_cost, conf in matches:
                if sd < size_tol:
                    tp += 1
                    slot_p = sl.start + p_i
                    slot_g = sl.start + g_j
                    size_errors.append(sd)
                    for attr in ["cyst", "mass", "tumor"]:
                        v = predictions[f"{attr}_valid"][n, slot_g]
                        if v > 0.5:
                            pred_lab = predictions[f"{attr}_pred"][n, slot_p] > 0.5
                            gt_lab = predictions[f"{attr}_gt"][n, slot_g] > 0.5
                            attr_totals[attr]["correct"] += int(pred_lab == gt_lab)
                            attr_totals[attr]["total"] += 1
                    # enh / att
                    if predictions["enh_valid"][n, slot_g] > 0.5:
                        enh_correct += int(predictions["enh_pred"][n, slot_p] ==
                                           int(round(predictions["enh_gt"][n, slot_g])))
                        enh_total += 1
                    if predictions["att_valid"][n, slot_g] > 0.5:
                        att_correct += int(predictions["att_pred"][n, slot_p] ==
                                           int(round(predictions["att_gt"][n, slot_g])))
                        att_total += 1
                else:
                    fp += 1
                    fn += 1
            fp += len(un_p)
            fn += len(un_g)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "method": "A",
        "conf_thresh": conf_thresh,
        "size_tol_cm": size_tol,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "size_mae_on_tp": float(np.mean(size_errors)) if size_errors else None,
        "cyst_acc_on_tp": (attr_totals["cyst"]["correct"] / attr_totals["cyst"]["total"]
                           if attr_totals["cyst"]["total"] > 0 else None),
        "mass_acc_on_tp": (attr_totals["mass"]["correct"] / attr_totals["mass"]["total"]
                           if attr_totals["mass"]["total"] > 0 else None),
        "tumor_acc_on_tp": (attr_totals["tumor"]["correct"] / attr_totals["tumor"]["total"]
                             if attr_totals["tumor"]["total"] > 0 else None),
        "enh_acc_on_tp": enh_correct / enh_total if enh_total > 0 else None,
        "att_acc_on_tp": att_correct / att_total if att_total > 0 else None,
    }


def evaluate_method_B(predictions, size_tol=1.0, weights=None):
    """Method B: Hungarian match at low threshold, then sweep confidence → AP.

    Matching uses type-aware cost. TP flag uses size tolerance only (to keep the
    TP criterion interpretable and independent of cost weights).
    """
    if weights is None:
        weights = DEFAULT_COST_WEIGHTS
    ex_p = predictions["exists_pred"]
    ex_g = predictions["exists_gt"]
    sz_p = predictions["size_pred"]
    sz_g = predictions["size_gt"]
    pred_type = {k: predictions[f"{k}_pred"] for k in ["cyst", "mass", "tumor"]}
    gt_type   = {k: predictions[f"{k}_gt"]    for k in ["cyst", "mass", "tumor"]}
    gt_valid  = {k: predictions[f"{k}_valid"] for k in ["cyst", "mass", "tumor"]}
    N = ex_p.shape[0]

    confidences = []
    is_tp = []
    total_gt = 0

    for n in range(N):
        for sl, _ in [(slice(0, 3), "L"), (slice(3, 6), "R")]:
            p_ex = ex_p[n, sl]; p_sz = sz_p[n, sl]
            g_ex = ex_g[n, sl]; g_sz = sz_g[n, sl]
            p_t = {k: pred_type[k][n, sl] for k in ["cyst", "mass", "tumor"]}
            g_t = {k: gt_type[k][n, sl]   for k in ["cyst", "mass", "tumor"]}
            g_v = {k: gt_valid[k][n, sl]  for k in ["cyst", "mass", "tumor"]}

            pred_idx = np.arange(3)
            gt_idx = np.where(g_ex > 0.5)[0]
            total_gt += len(gt_idx)

            if len(gt_idx) == 0:
                for i in pred_idx:
                    confidences.append(float(p_ex[i]))
                    is_tp.append(False)
                continue

            p_t_sel = {k: v[pred_idx] for k, v in p_t.items()}
            g_t_sel = {k: v[gt_idx]   for k, v in g_t.items()}
            g_v_sel = {k: v[gt_idx]   for k, v in g_v.items()}
            cost = compute_cost_matrix(p_sz[pred_idx], g_sz[gt_idx],
                                         p_t_sel, g_t_sel, g_v_sel, weights)
            r, c = linear_sum_assignment(cost)

            matched_p = set()
            for i, j in zip(r, c):
                p_i = pred_idx[i]
                g_j = gt_idx[j]
                size_diff = abs(p_sz[p_i] - g_sz[g_j])
                tp_flag = size_diff < size_tol
                confidences.append(float(p_ex[p_i]))
                is_tp.append(tp_flag)
                matched_p.add(p_i)
            for i in pred_idx:
                if i not in matched_p:
                    confidences.append(float(p_ex[i]))
                    is_tp.append(False)

    order = np.argsort(-np.array(confidences))
    conf = np.array(confidences)[order]
    tp = np.array(is_tp)[order].astype(int)
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # 11-point interpolated AP (Pascal VOC)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        t = t - 1e-9
        mask = recalls >= t
        p_at = precisions[mask].max() if mask.any() else 0.0
        ap += p_at / 11.0

    # F1 at each threshold
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-10)
    if len(f1s) > 0:
        best = np.argmax(f1s)
        f1_max = float(f1s[best])
        best_thresh = float(conf[best])
        p_at_f1 = float(precisions[best])
        r_at_f1 = float(recalls[best])
    else:
        f1_max = best_thresh = p_at_f1 = r_at_f1 = 0.0

    return {
        "method": "B",
        "size_tol_cm": size_tol,
        "total_gt": int(total_gt),
        "total_pred": int(len(confidences)),
        "AP_11point": float(ap),
        "F1_max": f1_max,
        "best_thresh": best_thresh,
        "P_at_F1max": p_at_f1,
        "R_at_F1max": r_at_f1,
    }


def evaluate_method_C(predictions, weights=None):
    """Method C: full mAP framework — multiple size tolerances."""
    out = {"method": "C", "per_tolerance": {}}
    for tol in [0.5, 1.0, 2.0]:
        out["per_tolerance"][f"{tol:.1f}cm"] = evaluate_method_B(predictions, size_tol=tol, weights=weights)
    ap_values = [out["per_tolerance"][k]["AP_11point"] for k in out["per_tolerance"]]
    out["mAP_over_tolerances"] = float(np.mean(ap_values))
    return out


def evaluate_method_A_threshold_sweep(predictions, size_tol=1.0, weights=None,
                                       thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Method A with multiple conf thresholds for side-by-side comparison."""
    return {
        "method": "A_sweep",
        "size_tol_cm": size_tol,
        "per_threshold": {
            f"conf={t:.1f}": evaluate_method_A(predictions, conf_thresh=t,
                                                 size_tol=size_tol, weights=weights)
            for t in thresholds
        },
    }


def build_random_baseline(reference_predictions, seed=42, variant="uniform",
                           model_exists_dist=None):
    """Build random baseline predictions with same shape as reference.

    Variants:
      - "uniform": exists~U(0,1), size~U(0, 10)cm — fully uninformed
      - "prevalence": exists ~ Bernoulli(prior), prior = GT positive rate
      - "model_calibrated": exists from reference model's distribution (given)
        + uniform size over [0, 10]cm — removes empirical size shortcut but
        keeps model's prediction rate
    GT, valid masks: copied from reference (these are real labels)
    """
    rng = np.random.default_rng(seed)
    N = reference_predictions["exists_pred"].shape[0]
    shape = (N, 6)

    ex_gt = reference_predictions["exists_gt"]
    sz_gt = reference_predictions["size_gt"]

    # Size pred: uniform over the range observed in GT (fair, no empirical bias)
    real_sizes = sz_gt[ex_gt > 0.5]
    real_sizes = real_sizes[real_sizes > 0]
    if len(real_sizes) == 0:
        sz_min, sz_max = 0.5, 8.0
    else:
        sz_min = float(real_sizes.min())
        sz_max = float(real_sizes.max())

    out = {}

    if variant == "uniform":
        out["exists_pred"] = rng.uniform(0, 1, size=shape).astype(np.float32)
        out["size_pred"] = rng.uniform(sz_min, sz_max, size=shape).astype(np.float32)
    elif variant == "prevalence":
        prior = float((ex_gt > 0.5).mean())   # ≈ 0.237
        # Center exists_pred around prior, with noise
        bernoulli = rng.binomial(1, prior, size=shape).astype(np.float32)
        noise = rng.uniform(-0.1, 0.1, size=shape)
        out["exists_pred"] = np.clip(bernoulli * 0.7 + 0.1 + noise, 0, 1).astype(np.float32)
        out["size_pred"] = rng.uniform(sz_min, sz_max, size=shape).astype(np.float32)
    elif variant == "model_calibrated":
        # Use model's actual exists_pred distribution but shuffle across slots
        # → same rate as model, but no correlation with GT
        model_flat = reference_predictions["exists_pred"].flatten()
        shuffled = rng.permutation(model_flat).reshape(shape)
        out["exists_pred"] = shuffled.astype(np.float32)
        out["size_pred"] = rng.uniform(sz_min, sz_max, size=shape).astype(np.float32)
    else:
        raise ValueError(variant)

    for feat in ["cyst", "mass", "tumor"]:
        out[f"{feat}_pred"] = rng.uniform(0, 1, size=shape).astype(np.float32)
        out[f"{feat}_gt"] = reference_predictions[f"{feat}_gt"]
        out[f"{feat}_valid"] = reference_predictions[f"{feat}_valid"]
    out["exists_gt"] = reference_predictions["exists_gt"]
    out["size_gt"] = reference_predictions["size_gt"]
    out["size_valid"] = reference_predictions["size_valid"]
    out["enh_pred"] = rng.integers(0, 2, size=shape).astype(np.int64)
    out["enh_gt"] = reference_predictions["enh_gt"]
    out["enh_valid"] = reference_predictions["enh_valid"]
    out["att_pred"] = rng.integers(0, 4, size=shape).astype(np.int64)
    out["att_gt"] = reference_predictions["att_gt"]
    out["att_valid"] = reference_predictions["att_valid"]
    return out


WEIGHT_PRESETS = {
    # Current (size-only): for reference
    "size_only":      {"w_size": 1.0, "w_cyst": 0.0, "w_mass": 0.0, "w_tumor": 0.0, "size_scale": 10.0},
    # Balanced: size + type equally, tumor most important
    "balanced":       {"w_size": 1.0, "w_cyst": 1.0, "w_mass": 2.0, "w_tumor": 3.0, "size_scale": 10.0},
    # Type-heavy: emphasize type accuracy (clinical: tumor is what matters)
    "type_heavy":     {"w_size": 0.5, "w_cyst": 1.0, "w_mass": 3.0, "w_tumor": 5.0, "size_scale": 10.0},
    # Size-heavy: emphasize size accuracy (radiologic: measurement accuracy)
    "size_heavy":     {"w_size": 3.0, "w_cyst": 0.5, "w_mass": 1.0, "w_tumor": 1.5, "size_scale": 10.0},
}


def evaluate_experiment(exp_dir, weights=None, weight_name="balanced"):
    """Evaluate one experiment using the best v3 epoch."""
    exp_dir = Path(exp_dir)
    v3 = exp_dir / "epoch_metrics_v3.csv"
    if not v3.exists():
        return None
    rows = list(csv.DictReader(open(v3)))
    if not rows:
        return None
    best_row = max(rows, key=crit_v3)
    ep = int(best_row["epoch"])
    npz = exp_dir / "predictions" / f"epoch_{ep}.npz"
    if not npz.exists():
        return None

    d = np.load(npz)
    predictions = {k: d[k] for k in d.files}

    if weights is None:
        weights = DEFAULT_COST_WEIGHTS

    return {
        "exp": exp_dir.name,
        "epoch": ep,
        "side_crit_v3": crit_v3(best_row),
        "n_patients": int(d["exists_pred"].shape[0]),
        "weight_preset": weight_name,
        "weights": weights,
        "method_A": evaluate_method_A(predictions, conf_thresh=0.5, size_tol=1.0, weights=weights),
        "method_B": evaluate_method_B(predictions, size_tol=1.0, weights=weights),
        "method_C": evaluate_method_C(predictions, weights=weights),
        "method_D": evaluate_method_D(predictions, size_tol=1.0),
    }


def format_markdown(results, out_path):
    """Write results as markdown doc."""
    lines = []
    lines.append("# Lesion-Level Evaluation Results")
    lines.append("")
    lines.append("Three object-detection-style evaluation protocols for L3D per-lesion prediction.")
    lines.append("All metrics computed on UF validation set (452 patients = 904 sides).")
    lines.append("Matching uses Hungarian assignment on size distance within each side (3 slots).")
    lines.append("")
    lines.append("## Protocols")
    lines.append("")
    lines.append("- **Method A**: fixed confidence threshold 0.5, fixed size tolerance 1.0 cm. Simple P/R/F1 + attribute accuracy on matched TP.")
    lines.append("- **Method B**: Hungarian match, sweep confidence → PR curve → 11-point AP. Reports AP + F1_max.")
    lines.append("- **Method C**: mAP over 3 size tolerances {0.5, 1.0, 2.0} cm.")
    lines.append("")

    # Method A table
    lines.append("## Method A — Fixed threshold (conf=0.5, |Δsize|<1.0 cm)")
    lines.append("")
    lines.append("| Experiment | TP | FP | FN | P | R | F1 | Size MAE (cm) | Cyst acc | Mass acc | Tumor acc | Enh acc | Att acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        a = r["method_A"]
        def f(v):
            return "—" if v is None else f"{v:.3f}"
        lines.append(f"| {r['exp']} | {a['tp']} | {a['fp']} | {a['fn']} | "
                     f"{a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | "
                     f"{f(a['size_mae_on_tp'])} | {f(a['cyst_acc_on_tp'])} | "
                     f"{f(a['mass_acc_on_tp'])} | {f(a['tumor_acc_on_tp'])} | "
                     f"{f(a['enh_acc_on_tp'])} | {f(a['att_acc_on_tp'])} |")
    lines.append("")

    # Method B table
    lines.append("## Method B — Hungarian + AP curve (|Δsize|<1.0 cm)")
    lines.append("")
    lines.append("| Experiment | total GT | total pred | **AP** | F1_max | P@F1max | R@F1max | best conf |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        b = r["method_B"]
        lines.append(f"| {r['exp']} | {b['total_gt']} | {b['total_pred']} | "
                     f"**{b['AP_11point']:.3f}** | {b['F1_max']:.3f} | "
                     f"{b['P_at_F1max']:.3f} | {b['R_at_F1max']:.3f} | {b['best_thresh']:.3f} |")
    lines.append("")

    # Method C table
    lines.append("## Method C — mAP over size tolerances {0.5, 1.0, 2.0} cm")
    lines.append("")
    lines.append("| Experiment | AP@0.5cm | AP@1.0cm | AP@2.0cm | **mAP** |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        c = r["method_C"]["per_tolerance"]
        mAP = r["method_C"]["mAP_over_tolerances"]
        lines.append(f"| {r['exp']} | {c['0.5cm']['AP_11point']:.3f} | {c['1.0cm']['AP_11point']:.3f} | "
                     f"{c['2.0cm']['AP_11point']:.3f} | **{mAP:.3f}** |")
    lines.append("")

    # Method D: per-class AP (COCO-style)
    lines.append("## Method D — Per-class AP (COCO-style, size_tol=1.0 cm)")
    lines.append("")
    lines.append("Confidence for class c = P(exists) × P(type=c). GT: slots with exists AND class-c AND valid.")
    lines.append("Hungarian match per side on size cost, TP iff matched AND |Δsize| < 1 cm.")
    lines.append("**mAP = unweighted mean of AP_cyst, AP_mass, AP_tumor.** AP_solid (=mass∪tumor) is extra.")
    lines.append("")
    lines.append("| Experiment | AP_cyst (n=528) | AP_mass (n=20) | AP_tumor (n=5) | AP_solid (n=25) | **mAP** |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        pc = r["method_D"]["per_class"]
        mAP = r["method_D"]["mAP_unweighted"]
        lines.append(f"| {r['exp']} | {pc['cyst']['AP']:.3f} | {pc['mass']['AP']:.3f} | "
                     f"{pc['tumor']['AP']:.3f} | {pc['solid']['AP']:.3f} | **{mAP:.3f}** |")
    lines.append("")
    lines.append("### Per-class F1_max (best PR-curve F1 threshold)")
    lines.append("")
    lines.append("| Experiment | F1_cyst | F1_mass | F1_tumor | F1_solid |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        pc = r["method_D"]["per_class"]
        lines.append(f"| {r['exp']} | {pc['cyst']['F1_max']:.3f} | {pc['mass']['F1_max']:.3f} | "
                     f"{pc['tumor']['F1_max']:.3f} | {pc['solid']['F1_max']:.3f} |")
    lines.append("")

    # Comparison to side-level crit
    lines.append("## Side-level crit_v3 (for reference)")
    lines.append("")
    lines.append("| Experiment | Ep | side_crit_v3 | F1 (method A) | AP (method B) | mAP_tol (method C) | **mAP_class (method D)** |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['exp']} | {r['epoch']} | {r['side_crit_v3']:.3f} | "
                     f"{r['method_A']['f1']:.3f} | {r['method_B']['AP_11point']:.3f} | "
                     f"{r['method_C']['mAP_over_tolerances']:.3f} | "
                     f"**{r['method_D']['mAP_unweighted']:.3f}** |")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", help="experiment directories")
    parser.add_argument("--weight_preset", default="balanced",
                        choices=list(WEIGHT_PRESETS.keys()),
                        help="Cost weight preset (default: balanced)")
    args = parser.parse_args()

    weight_preset = getattr(args, "weight_preset", "balanced")
    weights = WEIGHT_PRESETS.get(weight_preset, DEFAULT_COST_WEIGHTS)
    results = []
    ref_preds = None
    for exp_dir in args.exp_dirs:
        r = evaluate_experiment(exp_dir, weights=weights, weight_name=weight_preset)
        if r is not None:
            results.append(r)
            md = r["method_D"]
            print(f"✓ {r['exp']}: AP_cyst={md['per_class']['cyst']['AP']:.3f}, "
                  f"AP_mass={md['per_class']['mass']['AP']:.3f}, "
                  f"AP_tumor={md['per_class']['tumor']['AP']:.3f}, "
                  f"AP_solid={md['per_class']['solid']['AP']:.3f}, "
                  f"mAP={md['mAP_unweighted']:.3f}")
            # Cache first experiment's GT for random baseline
            if ref_preds is None:
                exp = Path(exp_dir)
                v3 = exp / "epoch_metrics_v3.csv"
                rows = list(csv.DictReader(open(v3)))
                best_row = max(rows, key=crit_v3)
                ep = int(best_row["epoch"])
                npz = np.load(exp / "predictions" / f"epoch_{ep}.npz")
                ref_preds = {k: npz[k] for k in npz.files}
        else:
            print(f"✗ SKIP: {exp_dir}")

    # Random baselines: 3 variants × 3 seeds
    if ref_preds is not None:
        for variant in ["uniform", "prevalence", "model_calibrated"]:
            for seed in [42, 43, 44]:
                rand_pred = build_random_baseline(ref_preds, seed=seed, variant=variant)
                r = {
                    "exp": f"RANDOM [{variant}] (seed={seed})",
                    "epoch": -1,
                    "side_crit_v3": 0.5,
                    "n_patients": int(ref_preds["exists_pred"].shape[0]),
                    "weight_preset": weight_preset,
                    "method_A": evaluate_method_A(rand_pred, conf_thresh=0.5, size_tol=1.0, weights=weights),
                    "method_B": evaluate_method_B(rand_pred, size_tol=1.0, weights=weights),
                    "method_C": evaluate_method_C(rand_pred, weights=weights),
                    "method_D": evaluate_method_D(rand_pred, size_tol=1.0),
                }
                results.append(r)
                md = r["method_D"]
                print(f"✓ RANDOM[{variant}] s{seed}: "
                      f"AP_cyst={md['per_class']['cyst']['AP']:.3f}, "
                      f"AP_mass={md['per_class']['mass']['AP']:.3f}, "
                      f"AP_tumor={md['per_class']['tumor']['AP']:.3f}, "
                      f"mAP={md['mAP_unweighted']:.3f}")

    OUT_DIR = Path(os.environ.get("KIDNEY3DCT_ROOT", Path(__file__).resolve().parents[1])) / "experiments" / "analysis" / "lesion_level_eval"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    format_markdown(results, OUT_DIR / "results.md")
    print(f"\nSaved:\n  {OUT_DIR / 'results.json'}\n  {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
