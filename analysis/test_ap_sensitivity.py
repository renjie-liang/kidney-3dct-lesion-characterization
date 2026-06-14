"""Sensitivity tests: does AP respond to performance changes as expected?

Unlike test_ap_synthetic.py which verifies exact AP values for specific inputs,
this file parametrically sweeps each "dimension of performance" and checks that
AP responds monotonically and with sensible magnitudes.

Dimensions tested:
  1. Recall rate (vary fraction of GT detected, precision=1)
  2. FP count — low-conf (TPs rank higher) vs interleaved (mixed)
  3. Size error magnitude (cliff at size_tol)
  4. Type prediction quality (fraction with correct type probability)
  5. Ranking quality (conf gap between TP and FP)
  6. Gaussian noise injection into a perfect model
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lesion_level_eval import compute_class_AP_coco, evaluate_method_D


# ──────────────────────────────────────────────────────────────
# Helpers (reuse from synthetic test)
# ──────────────────────────────────────────────────────────────

def make_empty(N):
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


def add_gt(d, pt, side, slot_local, size, cls):
    abs_slot = (3 if side == "R" else 0) + slot_local
    d["exists_gt"][pt, abs_slot] = 1.0
    d["size_gt"][pt, abs_slot] = size
    d["size_valid"][pt, abs_slot] = 1.0
    d[f"{cls}_gt"][pt, abs_slot] = 1.0
    for c in ["cyst", "mass", "tumor"]:
        d[f"{c}_valid"][pt, abs_slot] = 1.0


def add_pred(d, pt, side, slot_local, exists_p, size, type_probs):
    abs_slot = (3 if side == "R" else 0) + slot_local
    d["exists_pred"][pt, abs_slot] = exists_p
    d["size_pred"][pt, abs_slot] = size
    for c, p in type_probs.items():
        d[f"{c}_pred"][pt, abs_slot] = p


def cyst_ap(d):
    return compute_class_AP_coco(d, "cyst", size_tol=1.0)["AP"]


def fmt_row(label, results):
    return f"  {label:<12}  " + "  ".join(f"{v:.3f}" for v in results)


def assert_monotonic(values, increasing=True, name=""):
    """Check values are monotonic (allowing small tolerance for ties)."""
    tol = 0.01
    for i in range(len(values) - 1):
        if increasing:
            if values[i+1] < values[i] - tol:
                print(f"  ✗ FAIL monotonicity ({name}): values not increasing at index {i}: {values[i]:.3f} → {values[i+1]:.3f}")
                return False
        else:
            if values[i+1] > values[i] + tol:
                print(f"  ✗ FAIL monotonicity ({name}): values not decreasing at index {i}: {values[i]:.3f} → {values[i+1]:.3f}")
                return False
    return True


# ──────────────────────────────────────────────────────────────
# Sweeps
# ──────────────────────────────────────────────────────────────

def sweep_1_recall():
    """Fix precision=1 (no FPs), vary recall.
    Expected: AP monotonically increasing, ≈ recall rate."""
    print("\n═══ Sweep 1: Recall rate (precision=100%, no FPs) ═══")
    N = 50
    recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
    aps = []
    for r in recalls:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            if i < int(N * r):
                add_pred(d, i, "L", 0, 0.9, 3.0,
                         {"cyst": 0.9, "mass": 0.05, "tumor": 0.05})
        aps.append(cyst_ap(d))
    print(f"    recall:  " + "  ".join(f"{r:.1f}  " for r in recalls))
    print(fmt_row("AP:", aps))
    ok_mono = assert_monotonic(aps, increasing=True, name="recall")
    # Expected: AP ≈ (k+1)/11 for recall = k/10 (11-point interpolation)
    # At recall=1.0: AP = 1.0. At recall=0.5: AP = 6/11 ≈ 0.545 etc.
    expected = [(int(r * 10) + 1) / 11 for r in recalls]
    print(f"  expect:   " + "  ".join(f"{e:.3f}" for e in expected))
    ok_match = all(abs(a - e) < 0.05 for a, e in zip(aps, expected))
    return ok_mono and ok_match


def sweep_2a_fp_low_conf():
    """Fix recall=100%, add FPs with low conf on NON-TP slots (TPs still rank first).
    Expected: AP stays ≈ 1.0 regardless of FP count."""
    print("\n═══ Sweep 2a: FP count with LOW conf (TPs rank first) ═══")
    N = 20
    fp_counts = [0, 1, 3, 5]
    aps = []
    # FPs go on slots 1-5 (skip slot 0 which is TP)
    fp_slot_list = [("L", 1), ("L", 2), ("R", 0), ("R", 1), ("R", 2)]
    for fp_per in fp_counts:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            # TP on left slot 0
            add_pred(d, i, "L", 0, 0.9, 3.0, {"cyst": 0.9, "mass": 0, "tumor": 0})
            # FPs on non-TP slots
            for j in range(min(fp_per, 5)):
                side, loc = fp_slot_list[j]
                add_pred(d, i, side, loc, 0.3, 8.0,
                         {"cyst": 0.3, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  FP/patient: " + "  ".join(f"{c:3d}    " for c in fp_counts))
    print(fmt_row("AP:", aps))
    ok = all(a > 0.95 for a in aps)
    print(f"  Insight: Low-conf FPs don't hurt AP (all ≥ 0.95): {'✓' if ok else '✗'}")
    return ok


def sweep_2b_fp_interleaved():
    """Fix recall=100%, but TP conf and FP conf overlap (interleaved ranking).
    Expected: AP should decrease as more FPs interleaved with TPs."""
    print("\n═══ Sweep 2b: FP count INTERLEAVED with TPs (realistic) ═══")
    N = 20
    fp_counts = [0, 1, 2, 5]
    aps = []
    rng = np.random.default_rng(42)
    fp_slot_list = [("L", 1), ("L", 2), ("R", 0), ("R", 1), ("R", 2)]
    for fp_per in fp_counts:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            tp_conf = rng.uniform(0.3, 0.7)
            add_pred(d, i, "L", 0, tp_conf, 3.0,
                     {"cyst": tp_conf, "mass": 0, "tumor": 0})
            for j in range(min(fp_per, 5)):
                side, loc = fp_slot_list[j]
                fp_conf = rng.uniform(0.3, 0.7)
                add_pred(d, i, side, loc, fp_conf, 8.0,
                         {"cyst": fp_conf, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  FP/patient: " + "  ".join(f"{c:3d}    " for c in fp_counts))
    print(fmt_row("AP:", aps))
    ok_mono = assert_monotonic(aps, increasing=False, name="interleaved FPs")
    print(f"  Insight: AP decreases monotonically: {'✓' if ok_mono else '✗'}")
    return ok_mono


def sweep_3_size_cliff():
    """Vary size error Δ around tolerance (1 cm).
    Expected: AP = 1 for Δ < tol, AP = 0 for Δ > tol (sharp cliff)."""
    print("\n═══ Sweep 3: Size error magnitude (tolerance = 1.0 cm) ═══")
    N = 30
    deltas = [0.3, 0.5, 0.8, 0.95, 1.05, 1.5, 2.0]
    aps = []
    for delta in deltas:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            # Predict with size shifted by delta
            add_pred(d, i, "L", 0, 0.9, 3.0 + delta,
                     {"cyst": 0.9, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  Δsize:    " + "  ".join(f"{d:.2f}  " for d in deltas))
    print(fmt_row("AP:", aps))
    # Expected: sharp drop at Δ=1.0
    below = [aps[i] for i, d in enumerate(deltas) if d < 1.0]
    above = [aps[i] for i, d in enumerate(deltas) if d > 1.0]
    ok = all(a > 0.95 for a in below) and all(a < 0.05 for a in above)
    print(f"  Insight: cliff at Δ=1.0 (below all ≥ 0.95, above all ≤ 0.05): {'✓' if ok else '✗'}")
    return ok


def sweep_4_type_quality():
    """Fix exists=0.9, size correct, vary type prediction quality.
    Expected: AP_cyst increases with type_prob for cyst GT."""
    print("\n═══ Sweep 4: Type prediction quality (exists fixed at 0.9) ═══")
    N = 20
    type_qualities = [0.3, 0.5, 0.7, 0.9]
    aps = []
    for q in type_qualities:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            # TP with correct size, varying cyst type prob
            add_pred(d, i, "L", 0, 0.9, 3.0,
                     {"cyst": q, "mass": 0, "tumor": 0})
            # Add a distractor FP at slot 1 with wrong size (so not a TP)
            # and cyst prob 0.5 — so depending on q, TP or FP wins ranking
            add_pred(d, i, "L", 1, 0.9, 8.0,
                     {"cyst": 0.5, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  type_prob:" + "  ".join(f"{q:.2f}  " for q in type_qualities))
    print(fmt_row("AP:", aps))
    ok_mono = assert_monotonic(aps, increasing=True, name="type_prob")
    print(f"  Insight: AP increases with type prob: {'✓' if ok_mono else '✗'}")
    return ok_mono


def sweep_5_ranking_gap():
    """TP conf and FP conf separation matters only in terms of ordering.
    A gap of 0.01 vs 0.5 should give similar AP (both correctly rank TP first)."""
    print("\n═══ Sweep 5: TP/FP conf gap (AP is ranking-invariant) ═══")
    N = 20
    gaps = [0.01, 0.05, 0.1, 0.3, 0.5]   # TP conf = 0.5 + gap, FP conf = 0.5
    aps = []
    for gap in gaps:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            add_pred(d, i, "L", 0, 1.0, 3.0,
                     {"cyst": 0.5 + gap, "mass": 0, "tumor": 0})
            # FP at slot 1 (wrong size)
            add_pred(d, i, "L", 1, 1.0, 8.0,
                     {"cyst": 0.5, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  gap:      " + "  ".join(f"{g:.2f}  " for g in gaps))
    print(fmt_row("AP:", aps))
    # All should be ~1.0 since TP > FP in rank regardless of gap
    variance = max(aps) - min(aps)
    ok = variance < 0.05 and all(a > 0.9 for a in aps)
    print(f"  Insight: AP insensitive to conf gap magnitude (range {variance:.3f}): {'✓' if ok else '✗'}")
    return ok


def sweep_6_noise_injection():
    """Add Gaussian noise to a perfect model's predictions, measure AP degradation."""
    print("\n═══ Sweep 6: Gaussian noise on perfect model ═══")
    N = 40
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    rng = np.random.default_rng(42)
    aps = []
    for sigma in sigmas:
        d = make_empty(N)
        for i in range(N):
            add_gt(d, i, "L", 0, 3.0, "cyst")
            # Perfect pred + noise on exists and size
            exists_p = float(np.clip(0.9 + rng.normal(0, sigma), 0, 1))
            size_p = 3.0 + float(rng.normal(0, sigma * 2))   # size noise scales
            cyst_p = float(np.clip(0.9 + rng.normal(0, sigma), 0, 1))
            add_pred(d, i, "L", 0, exists_p, size_p,
                     {"cyst": cyst_p, "mass": 0, "tumor": 0})
            # Add low-conf "ghost" predictions on slots 1/2 (noise can push these up)
            for s in [1, 2]:
                ghost_exists = float(np.clip(0.0 + rng.normal(0, sigma), 0, 1))
                ghost_size = float(5.0 + rng.normal(0, sigma * 2))
                ghost_cyst = float(np.clip(0.0 + rng.normal(0, sigma), 0, 1))
                add_pred(d, i, "L", s, ghost_exists, ghost_size,
                         {"cyst": ghost_cyst, "mass": 0, "tumor": 0})
        aps.append(cyst_ap(d))
    print(f"  σ:        " + "  ".join(f"{s:.2f}  " for s in sigmas))
    print(fmt_row("AP:", aps))
    # Expected: sigma=0 → AP=1.0, monotonic-ish decrease
    ok_start = aps[0] > 0.95
    ok_end_lower = aps[-1] < aps[0]
    print(f"  Insight: σ=0 → AP≈1.0 ({aps[0]:.3f}), σ={sigmas[-1]} → AP={aps[-1]:.3f} (lower): "
          f"{'✓' if ok_start and ok_end_lower else '✗'}")
    return ok_start and ok_end_lower


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────

def main():
    results = {
        "1 recall_monotone":        sweep_1_recall(),
        "2a fp_low_conf (robust)":  sweep_2a_fp_low_conf(),
        "2b fp_interleaved (hurt)": sweep_2b_fp_interleaved(),
        "3 size_cliff":             sweep_3_size_cliff(),
        "4 type_quality":           sweep_4_type_quality(),
        "5 ranking_gap (invariant)":sweep_5_ranking_gap(),
        "6 noise_injection":        sweep_6_noise_injection(),
    }
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for name, ok in results.items():
        mark = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {mark}  {name}")
    total = sum(results.values())
    print(f"\n{total}/{len(results)} sweeps passed")
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
