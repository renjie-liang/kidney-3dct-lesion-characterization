"""Synthetic golden tests for lesion_level_eval metrics.

For each scenario we construct predictions with known properties and verify
AP_cyst / AP_mass / AP_tumor / mAP are as expected.

Scenarios:
  A. Perfect:            every GT perfectly detected + correctly classified  → AP = 1.0
  B. Perfect detection only: exists/size correct, types random               → AP ≈ class prevalence
  C. Half recall, perfect precision: detect half of GTs, correctly           → AP ≈ 0.5
  D. Inverted confidence: everything correct but high conf on false preds   → AP ≈ 0
  E. Random everything:  uniform random predictions                          → AP near chance
  F. Zero positives:     class has no GT                                     → AP = NaN
  G. Size too far:       detection correct but |Δsize| > tolerance           → AP ≈ 0
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lesion_level_eval import compute_class_AP_coco, evaluate_method_D


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def make_empty(N=100):
    """Blank prediction dict with N patients, no lesions."""
    shape = (N, 6)
    return {
        "exists_pred": np.zeros(shape, dtype=np.float32),
        "exists_gt":   np.zeros(shape, dtype=np.float32),
        "size_pred":   np.zeros(shape, dtype=np.float32),
        "size_gt":     np.zeros(shape, dtype=np.float32),
        "size_valid":  np.zeros(shape, dtype=np.float32),
        **{f"{c}_pred":  np.zeros(shape, dtype=np.float32) for c in ["cyst", "mass", "tumor"]},
        **{f"{c}_gt":    np.zeros(shape, dtype=np.float32) for c in ["cyst", "mass", "tumor"]},
        **{f"{c}_valid": np.zeros(shape, dtype=np.float32) for c in ["cyst", "mass", "tumor"]},
    }


def add_gt_lesion(data, patient, side, slot_idx_in_side, size, cls):
    """Add a GT lesion to specified slot. cls ∈ {'cyst', 'mass', 'tumor'}."""
    abs_slot = (3 if side == "R" else 0) + slot_idx_in_side
    data["exists_gt"][patient, abs_slot] = 1.0
    data["size_gt"][patient, abs_slot] = size
    data["size_valid"][patient, abs_slot] = 1.0
    data[f"{cls}_gt"][patient, abs_slot] = 1.0
    for c in ["cyst", "mass", "tumor"]:
        data[f"{c}_valid"][patient, abs_slot] = 1.0


def add_pred_lesion(data, patient, side, slot_idx_in_side,
                    exists_prob, size, type_probs):
    """Add a predicted lesion to specified slot. type_probs is dict {cyst, mass, tumor} → prob."""
    abs_slot = (3 if side == "R" else 0) + slot_idx_in_side
    data["exists_pred"][patient, abs_slot] = exists_prob
    data["size_pred"][patient, abs_slot] = size
    for c, p in type_probs.items():
        data[f"{c}_pred"][patient, abs_slot] = p


def check(name, got, expected, tol=0.05):
    """Assert metric within tolerance."""
    if isinstance(expected, float) and np.isnan(expected):
        ok = np.isnan(got)
        result = "✓" if ok else "✗"
        print(f"  {result} {name}: expected NaN, got {got}")
        return ok
    diff = abs(got - expected)
    ok = diff < tol
    result = "✓" if ok else "✗"
    print(f"  {result} {name}: expected {expected:.3f} ± {tol}, got {got:.3f} (diff {diff:.3f})")
    return ok


# ──────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────

def test_A_perfect():
    """Every GT perfectly predicted, nothing else. AP = 1.0 expected."""
    print("\n═══ Scenario A: Perfect predictions ═══")
    N = 50
    d = make_empty(N)

    # Generate: patient i has cyst on left (slot 0) size=2+i*0.1
    # Patient i has mass on right (slot 0) size=3+i*0.1 if i < 20
    # Patient i has tumor on right (slot 1) size=4 if i < 10
    for i in range(N):
        size_c = 2.0 + i * 0.05
        add_gt_lesion(d, i, "L", 0, size_c, "cyst")
        add_pred_lesion(d, i, "L", 0,
                        exists_prob=0.95, size=size_c,
                        type_probs={"cyst": 0.95, "mass": 0.05, "tumor": 0.05})
        if i < 20:
            size_m = 3.0 + i * 0.05
            add_gt_lesion(d, i, "R", 0, size_m, "mass")
            add_pred_lesion(d, i, "R", 0,
                            exists_prob=0.90, size=size_m,
                            type_probs={"cyst": 0.05, "mass": 0.90, "tumor": 0.05})
        if i < 10:
            size_t = 4.5 + i * 0.01
            add_gt_lesion(d, i, "R", 1, size_t, "tumor")
            add_pred_lesion(d, i, "R", 1,
                            exists_prob=0.85, size=size_t,
                            type_probs={"cyst": 0.05, "mass": 0.05, "tumor": 0.85})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    ok = all([
        check(f"AP_cyst (n_pos={pc['cyst']['n_pos']})",   pc["cyst"]["AP"],   1.00, 0.02),
        check(f"AP_mass (n_pos={pc['mass']['n_pos']})",   pc["mass"]["AP"],   1.00, 0.02),
        check(f"AP_tumor (n_pos={pc['tumor']['n_pos']})", pc["tumor"]["AP"],  1.00, 0.02),
        check(f"mAP",   r["mAP_unweighted"],    1.00, 0.02),
    ])
    return ok


def test_B_confidence_inversion():
    """Everything correct except confidence is inverted: low conf on true positives.

    Insight: AP is robust to confidence ranking — even if TPs are ranked below FPs,
    as long as all GTs are eventually reached (recall=1), AP has a lower bound of
    TP / (TP + FP). Here: 30 TPs, 60 FPs → AP >= 30/90 ≈ 0.333.
    """
    print("\n═══ Scenario B: Inverted confidence (TP has low, FP has high) ═══")
    print("  [Insight: AP penalizes missed GTs more than rank errors.]")
    N = 30
    d = make_empty(N)

    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        # Correct pred but low confidence (TP)
        add_pred_lesion(d, i, "L", 0,
                        exists_prob=0.1, size=3.0,
                        type_probs={"cyst": 0.1, "mass": 0.05, "tumor": 0.05})
        # FP slots (left 1, 2 have no GT) — high conf → rank first
        for s in [1, 2]:
            add_pred_lesion(d, i, "L", s,
                            exists_prob=0.9, size=5.0,
                            type_probs={"cyst": 0.9, "mass": 0.05, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # Expected: 11-point AP = 30/(30+60) = 0.333 (TPs accumulate at end of ranking,
    # interpolated precision = 1/3 across all 11 recall levels)
    expected_ap = 30 / 90
    ok = check(f"AP_cyst (expected TP/(TP+FP)={expected_ap:.3f})",
               pc["cyst"]["AP"], expected_ap, 0.02)
    return ok


def test_C_half_recall_perfect_precision():
    """Detect 50% of GTs with high confidence, other 50% not predicted at all.
    Expected: AP = 0.5 (Pascal 11-point, since precision stays at 1.0 up to R=0.5)."""
    print("\n═══ Scenario C: 50% recall at perfect precision ═══")
    N = 40
    d = make_empty(N)

    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        if i < N // 2:   # detect first half only
            add_pred_lesion(d, i, "L", 0,
                            exists_prob=0.9, size=3.0,
                            type_probs={"cyst": 0.9, "mass": 0.05, "tumor": 0.05})
        # No pred for second half → FN
        # Other slots: no pred → no FPs

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # Pascal 11-point: precision=1 up to recall=0.5, then 0 → AP = sum(1.0 at points where R>=t, for t in [0,0.1,...0.5])
    # That's 6 points with precision=1, 5 points with 0 → AP = 6/11 ≈ 0.545
    ok = check("AP_cyst (half recall)", pc["cyst"]["AP"], 6/11, 0.05)
    return ok


def test_D_empty_class():
    """No tumors in GT. AP_tumor should be NaN (not 0)."""
    print("\n═══ Scenario D: Empty class (n_pos=0 for tumor) ═══")
    N = 30
    d = make_empty(N)

    # GT has cyst only, no tumor, no mass
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        add_pred_lesion(d, i, "L", 0, 0.9, 3.0, {"cyst": 0.9, "mass": 0.1, "tumor": 0.1})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    ok = all([
        check(f"AP_cyst (n_pos={pc['cyst']['n_pos']})",   pc["cyst"]["AP"],   1.00, 0.05),
        check(f"AP_mass (n_pos={pc['mass']['n_pos']})",   pc["mass"]["AP"],   float("nan")),
        check(f"AP_tumor (n_pos={pc['tumor']['n_pos']})", pc["tumor"]["AP"],  float("nan")),
        check(f"mAP (only cyst counted, {r['mAP_classes_counted']} classes)",
              r["mAP_unweighted"], 1.00, 0.05),
    ])
    assert r["mAP_classes_counted"] == 1, f"Expected 1 class counted, got {r['mAP_classes_counted']}"
    return ok


def test_E_random_baseline():
    """Uniform random predictions on sparse GT. AP should approximate prevalence."""
    print("\n═══ Scenario E: Random predictions ═══")
    N = 200
    d = make_empty(N)
    rng = np.random.default_rng(42)

    # Sparse GT: 30% patients have cyst on left slot 0
    n_cyst = 0
    for i in range(N):
        if rng.random() < 0.3:
            add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
            n_cyst += 1

    # Random predictions
    d["exists_pred"] = rng.uniform(0, 1, (N, 6)).astype(np.float32)
    d["size_pred"]   = rng.uniform(0, 5, (N, 6)).astype(np.float32)
    for c in ["cyst", "mass", "tumor"]:
        d[f"{c}_pred"] = rng.uniform(0, 1, (N, 6)).astype(np.float32)

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # Expected: very small, since random Hungarian assignment with size cost still might match
    # but confidence is uncorrelated with correctness
    ok = all([
        check(f"AP_cyst (random, n_pos={pc['cyst']['n_pos']})",
              pc["cyst"]["AP"], 0.10, 0.15),   # could be 0 to ~0.2
    ])
    return ok


def test_F_size_beyond_tolerance():
    """Exists correct, type correct, but size wildly off (|Δsize| > tolerance).
    Expected: AP low because matched pairs fail TP criterion."""
    print("\n═══ Scenario F: Size beyond tolerance (|Δsize| > 1cm) ═══")
    N = 30
    d = make_empty(N)

    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        # Predict with very wrong size (Δ = 5cm)
        add_pred_lesion(d, i, "L", 0,
                        exists_prob=0.9, size=8.0,   # 5cm off
                        type_probs={"cyst": 0.9, "mass": 0.05, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # All preds fail size_tol → no TP → AP = 0
    ok = check("AP_cyst (size off by 5cm)", pc["cyst"]["AP"], 0.00, 0.05)
    return ok


def test_G_partial_class_confusion():
    """Detection right, but model predicts cyst confidence 0.5 vs mass confidence 0.5 for
    all slots. If GT is cyst and the class-c confidence uses P(exists) × P(type=c),
    the cyst TP and mass FP both have confidence 0.9*0.5=0.45 → ambiguous ranking."""
    print("\n═══ Scenario G: Ambiguous type confidence ═══")
    N = 20
    d = make_empty(N)
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        add_pred_lesion(d, i, "L", 0,
                        exists_prob=0.9, size=3.0,
                        # Ambiguous: both cyst and mass at 0.5
                        type_probs={"cyst": 0.5, "mass": 0.5, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # AP_cyst should still be ~1 because there's 1 cyst pred per patient and no competing FP from other slots.
    ok = check("AP_cyst (ambiguous type ok because no competing FP)",
                pc["cyst"]["AP"], 1.00, 0.05)
    return ok


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────

def test_H_decent_detector_mixed_conf():
    """Realistic: decent model with mixed confidence.

    Setup: 20 patients, each with 1 cyst GT (size=3).
      - 10 high-conf TPs: exists=0.95, cyst=0.95 → conf ≈ 0.90
      - 10 medium-conf FPs: exists=0.7, cyst=0.7 → conf ≈ 0.49 (on slot 1, wrong size)
      - 10 low-conf TPs: exists=0.3, cyst=0.3 → conf ≈ 0.09

    Ranking: 10 TPs, 10 FPs, 10 TPs.
    At rank 10: P=1.0, R=0.5
    At rank 20: P=0.5, R=0.5
    At rank 30: P=0.667, R=1.0

    11-point AP: 6 levels at P=1.0 (recall 0..0.5), 5 levels at P=0.667 (recall 0.6..1.0)
    → AP = (6×1.0 + 5×0.667) / 11 ≈ 0.848
    """
    print("\n═══ Scenario H: Decent detector, mixed confidence (realistic) ═══")
    N = 20
    d = make_empty(N)
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        if i < 10:
            # High-conf TP
            add_pred_lesion(d, i, "L", 0, exists_prob=0.95, size=3.0,
                            type_probs={"cyst": 0.95, "mass": 0.05, "tumor": 0.05})
        else:
            # Low-conf TP at slot 0
            add_pred_lesion(d, i, "L", 0, exists_prob=0.3, size=3.0,
                            type_probs={"cyst": 0.3, "mass": 0.05, "tumor": 0.05})
        # Medium-conf FP at slot 1 (wrong size so it can't match GT)
        if i < 10:
            add_pred_lesion(d, i, "L", 1, exists_prob=0.7, size=8.0,
                            type_probs={"cyst": 0.7, "mass": 0.05, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    expected_ap = (6 * 1.0 + 5 * (20 / 30)) / 11
    ok = check(f"AP_cyst (realistic mixed, expected 6/11 + 5/11 × 20/30 = {expected_ap:.3f})",
               pc["cyst"]["AP"], expected_ap, 0.02)
    return ok


def test_I_overpredicting_model():
    """Realistic: model over-predicts heavily (like our L3D baseline).

    Setup: 15 cyst GTs, model outputs ~45 predictions at similar confidence.
      - 15 TPs: cyst class conf 0.6 (just barely above)
      - 30 FPs: cyst class conf 0.59 (just below)

    TPs ranked first, then FPs. After TPs: P=1, R=1.0.
    As FPs add: P drops but R stays 1.
    At R=1.0: max P = 1.0 (achieved at item 15).
    11-point AP ≈ 1.0.

    BUT the point of this scenario is: even though model over-predicts, AP
    doesn't penalize it because the TPs have the highest confidences.
    This is an important property to understand.
    """
    print("\n═══ Scenario I: Over-predicting model (TPs barely outrank FPs) ═══")
    N = 15
    d = make_empty(N)
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        # TP with barely-higher confidence
        add_pred_lesion(d, i, "L", 0, exists_prob=0.8, size=3.0,
                        type_probs={"cyst": 0.75, "mass": 0.05, "tumor": 0.05})
        # 2 FPs per patient with slightly lower conf
        for s in [1, 2]:
            add_pred_lesion(d, i, "L", s, exists_prob=0.8, size=8.0,
                            type_probs={"cyst": 0.74, "mass": 0.05, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    ok = check("AP_cyst (TPs barely outrank FPs ⇒ near 1.0)",
               pc["cyst"]["AP"], 1.0, 0.02)
    return ok


def test_J_underpredicting_conservative():
    """Realistic: conservative model — predicts few, but what it predicts is correct.

    Setup: 30 cyst GTs, model predicts only 15 TPs at high confidence, misses the other 15.
      - 15 TPs (conf 0.85, correct size/type)
      - 0 FPs (model doesn't predict anywhere else)

    Ranking: 15 TPs (conf 0.85), then 75 zero-conf predictions (FPs since other slots aren't matched).
    After 15 TPs: P=1.0, R=0.5
    After 90 items: P=15/90 ≈ 0.167, R=0.5

    At 11-point interpolation:
      t=0 to 0.5: max P at R>=t = 1.0 (from rank 1-15)
      t=0.6 to 1.0: max P at R>=t = 0 (no points reach R>=0.6)

    AP = (6 × 1.0 + 5 × 0) / 11 ≈ 0.545
    """
    print("\n═══ Scenario J: Conservative model, 50% recall, 100% precision ═══")
    N = 30
    d = make_empty(N)
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        if i < 15:
            add_pred_lesion(d, i, "L", 0, exists_prob=0.85, size=3.0,
                            type_probs={"cyst": 0.85, "mass": 0.05, "tumor": 0.05})
        # Other 15 patients: no prediction (all zeros)

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    expected = (6 * 1.0 + 5 * 0) / 11   # same as scenario C, but different setup
    ok = check(f"AP_cyst (50% recall, perfect precision, expected {expected:.3f})",
               pc["cyst"]["AP"], expected, 0.03)
    return ok


def test_K_gradual_calibration():
    """Realistic: model with gradual confidence calibration.

    Setup: 20 cyst GTs. For each, TP confidence is uniformly spaced in [0.2, 0.9].
    No FPs (clean). Intermingled TP confidences just rank them.
    Expected: AP = 1.0 (all TPs, no FPs regardless of order).
    """
    print("\n═══ Scenario K: Calibrated confidence, no FPs ═══")
    N = 20
    d = make_empty(N)
    for i in range(N):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        conf = 0.2 + (i / (N - 1)) * 0.7    # uniformly 0.2..0.9
        add_pred_lesion(d, i, "L", 0, exists_prob=conf, size=3.0,
                        type_probs={"cyst": conf, "mass": 0.05, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    ok = check("AP_cyst (all TPs, varied conf ⇒ 1.0)",
               pc["cyst"]["AP"], 1.0, 0.02)
    return ok


def test_L_realistic_two_class():
    """Realistic: model good at cyst, weak at mass (mirrors our real setup).

    Setup: 30 cyst GTs (common) + 10 mass GTs (rare).
      - Cyst: model predicts 30 TPs correctly at conf 0.8 + 10 extra FPs at conf 0.7
      - Mass: model predicts 5 TPs correctly at conf 0.6 + 15 random FPs at conf 0.5

    Cyst ranking: 30 TPs then 10 FPs.
      After 30 TPs: P=1, R=1.0 → all good → AP=1.0

    Mass ranking: 5 TPs (0.6) + 15 FPs (0.5):
      After 5 TPs: P=1, R=5/10=0.5
      After 20 items: P=5/20=0.25, R=0.5
      At R>0.5 no points → interpolated precision = 0 for R>=0.6.
      AP = (6 × 1.0 + 5 × 0) / 11 = 0.545
    """
    print("\n═══ Scenario L: Realistic — strong cyst, weak mass ═══")
    N = 40
    d = make_empty(N)

    # Patients 0-29: cyst GT + TP + one FP (left slot 1)
    for i in range(30):
        add_gt_lesion(d, i, "L", 0, 3.0, "cyst")
        add_pred_lesion(d, i, "L", 0, exists_prob=0.9, size=3.0,
                        type_probs={"cyst": 0.9, "mass": 0.05, "tumor": 0.05})
    # Some cyst FPs on slot 1 for patients 0-9 (wrong size so |Δsize|>1)
    for i in range(10):
        add_pred_lesion(d, i, "L", 1, exists_prob=0.8, size=8.0,
                        type_probs={"cyst": 0.85, "mass": 0.05, "tumor": 0.05})

    # Patients 30-39: mass GT
    for i in range(30, 40):
        add_gt_lesion(d, i, "L", 0, 2.5, "mass")
        if i < 35:    # 5 TPs out of 10
            add_pred_lesion(d, i, "L", 0, exists_prob=0.7, size=2.5,
                            type_probs={"cyst": 0.1, "mass": 0.8, "tumor": 0.05})
        # 15 FPs for mass: spread across remaining slots
        for s in [1, 2]:
            add_pred_lesion(d, i, "L", s, exists_prob=0.6, size=7.0,
                            type_probs={"cyst": 0.1, "mass": 0.75, "tumor": 0.05})

    r = evaluate_method_D(d, size_tol=1.0, matching="coco")
    pc = r["per_class"]
    # cyst: FPs have wrong size so shouldn't match. 30 TPs from correct patients.
    ok = all([
        check(f"AP_cyst (n_pos={pc['cyst']['n_pos']})",  pc["cyst"]["AP"],  1.0, 0.05),
        # mass: 5 TPs from 10 GTs at 100% precision → AP ≈ 6/11 ≈ 0.545
        check(f"AP_mass (n_pos={pc['mass']['n_pos']}, 50% recall)",
              pc["mass"]["AP"], 6/11, 0.08),
    ])
    return ok


def main():
    results = {
        "A perfect":                        test_A_perfect(),
        "B confidence_inversion":           test_B_confidence_inversion(),
        "C half_recall_perfect_prec":       test_C_half_recall_perfect_precision(),
        "D empty_class (NaN)":              test_D_empty_class(),
        "E random_baseline":                test_E_random_baseline(),
        "F size_beyond_tolerance":          test_F_size_beyond_tolerance(),
        "G ambiguous_type":                 test_G_partial_class_confusion(),
        "H mixed_conf_decent":              test_H_decent_detector_mixed_conf(),
        "I overpredicting_barely":          test_I_overpredicting_model(),
        "J conservative_50%_recall":        test_J_underpredicting_conservative(),
        "K calibrated_conf":                test_K_gradual_calibration(),
        "L two_class_strong_weak":          test_L_realistic_two_class(),
    }
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for name, ok in results.items():
        mark = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {mark}  {name}")
    total = sum(results.values())
    print(f"\n{total}/{len(results)} scenarios passed")
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
