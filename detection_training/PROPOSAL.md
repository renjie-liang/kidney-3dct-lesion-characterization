# Detection-Oriented Training Proposal

> Last updated: 2026-04-14 (approved, awaiting implementation)
> Scope: **Follow-up experiments** for improved per-lesion detection. **Does not block the main paper.**
> Status: **Approved** — ready for implementation.

## Approved Design Decisions (2026-04-14)

| Question | Decision |
|---|---|
| C-slot-embed in matrix | Yes, participates; **skip M2 (count reg)** for it (redundant with its count head) |
| Method combinations | Keep as proposed (E9, E11, E13 highest priority) |
| Size tolerances in eval | **Report all 3**: 0.5 / 1.0 / 2.0 cm |
| Training epochs | **30 epochs** (reduced from 50 to fit SLURM time budget and speed up queue backfill) |
| Seeds per experiment | **3 seeds per experiment, sequential within a single SLURM job** (avoid queue contention) |
| Motivation clarity | Approved |
| Method coverage | Approved (will write all 8 methods) |
| Matrix size | Approved (17 experiments) |
| File structure | Approved |
| Budget | Approved |

---

## 1. Motivation

### The core problem
The main paper's L3D training is optimized for **slot-level BCE + noisy-OR side-level supervision**. It achieves strong side-level AUC (0.65 crit) but weak lesion-level AP (mAP = 0.06–0.07).

The per-class AP evaluation (Method D) reveals:
- **Over-prediction**: model predicts ~2700 lesions when GT has 644 → precision ≈ 14% at default threshold
- **No precision incentive**: BCE weights FP and FN equally; noisy-OR max-aggregation lets FP slots contribute to correct side-level label without penalty
- **Positional mismatch**: training assumes `slot i = i-th largest lesion`; evaluation does Hungarian matching on size

### Training–evaluation mismatch (summary)

| Training | Evaluation |
|---|---|
| Positional slot assignment | Hungarian matching |
| Per-slot BCE (symmetric FP/FN) | AP (heavily penalizes FP) |
| exists and type are independent heads | confidence = P(exists) × P(type) |
| No explicit "no-object" supervision | Empty-slot preds become FP |
| Size is just a regression target | Size defines the matching criterion |

### Expected impact of fixing
Back-of-envelope: a proper detection loss should give AP_cyst 0.17 → 0.30–0.40, mAP 0.07 → 0.12–0.18. Still below strong-supervision medical detection (0.6+), but would be a respectable weakly-supervised baseline.

### Positioning
- **Main paper (MedIA submission)**: Keep current side-level-focused story. Acknowledge detection limitation in discussion.
- **This proposal**: Build the infrastructure + run experiments to push per-lesion AP. Results feed into:
  - Rebuttal ammunition (if reviewers challenge per-lesion performance)
  - Future follow-up paper on "weakly-supervised kidney lesion detection from radiology reports"
  - Stage 2 report generation (better per-lesion input → better report)

---

## 2. Current Baselines (v3 eval, Method D at size_tol=1cm)

| Experiment | AP_cyst (n=528) | AP_mass (n=20) | AP_tumor (n=5) | AP_solid (n=25) | mAP |
|---|---|---|---|---|---|
| D (Crop+Kmask, core_matrix) | 0.167 | 0.011 | 0.006 | 0.011 | 0.061 |
| C (slot embed, fixed) | **0.201** | 0.010 | 0.000 | 0.010 | **0.070** |
| B (Hungarian head, fixed) | 0.047 | 0.026 | 0.001 | 0.026 | 0.024 |
| Random (9-run mean) | 0.049 | 0.004 | 0.001 | — | 0.018 |

**Starting points for improvement experiments**: D (most common baseline) and C-slot-embed (best per-lesion).

---

## 3. Methods to Try

Each method is **independent** — can be turned on/off via config. We will test them individually and in combinations.

### M1. Focal Loss on `exists`

**Motivation**: BCE treats easy and hard negatives equally. Most slots are easy negatives (empty); the model sees lots of "it's zero, it's zero, it's zero" gradient. Focal loss down-weights easy examples so training focuses on hard ones (including hard negatives = "I said it's a lesion but it's not").

**Formulation**:
$$FL(p, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t = p$ if $y = 1$ else $1-p$. Default $\alpha = 0.25$ (for the minority positive class), $\gamma = 2.0$.

**Where applied**: exists loss only. Works in both `positional` mode (as a drop-in BCE
replacement, combinable with size_weight / hard_neg as multiplicative modifiers) and
`hungarian` mode (applied to both matched-pair exists loss AND unmatched no-object loss).

**Expected impact**: reduce FP count by 30–50%, AP gain 0.02–0.05.

**Implementation**: `losses/focal.py`, function applied inline within `losses/combined.py`
for positional mode and `losses/hungarian.py` for hungarian mode.

**Risks**: poorly tuned α may hurt; tumor (very rare) might get pushed into even lower confidence.

---

### M2. Count Consistency Regularizer

**Motivation**: current model over-predicts — it outputs `exists > 0.5` on ~6 slots per patient when GT has ~1.4. We can directly penalize this.

**Formulation** (three variants, pick one):
- **Soft L1**: `|sum(sigmoid(exists_logit)) - count_gt|`, per side
- **CE on count**: treat `count_side ∈ {0,1,2,3}` as 4-way classification (similar to C's count head, but applied to all heads)
- **Expected count matching**: `|E[count] - count_gt|^2` using differentiable sum over sigmoid

**Loss weight**: start with 0.5 (hier loss is already 0.3).

**Expected impact**: reduce over-prediction, AP gain 0.02–0.05.

**Implementation**: `losses/count_reg.py`.

**Risks**: if weight too high, suppresses recall on patients with many lesions.

---

### M3. DETR-style Hungarian Matching Loss

**Motivation**: the biggest architectural fix. Instead of "slot i supervises GT slot i" (positional), match pred ↔ GT via Hungarian at training time; unmatched preds get a "no-object" (exists → 0) supervision.

**Formulation**:
1. For each patient/side:
   - Compute cost between pred slots and GT slots: `C[i,j] = λ₁·|Δsize| + λ₂·BCE(exists) + λ₃·type_cost`
   - Hungarian assignment → optimal pairing
2. Loss:
   - Matched pairs: BCE(exists=1) + BCE(cyst/mass/tumor) + MSE(size)
   - Unmatched preds: BCE(exists=0) — the "no-object" supervision
   - Unmatched GTs: already covered by matched requirement

**Expected impact**: this is the biggest gain — AP_cyst may jump from 0.17 to 0.30+, mAP from 0.07 to 0.15+. This is essentially what DETR does and should fix the positional-vs-Hungarian mismatch.

**Implementation**: `losses/hungarian.py`. Reuses `scipy.optimize.linear_sum_assignment` (already in l3_heads.py).

**Risks**: matching cost design matters; DETR uses Hungarian but also needs auxiliary losses at each decoder layer to stabilize. Our version doesn't have decoder layers, so should be simpler but still non-trivial to tune.

---

### M4. Joint Confidence Supervision

**Motivation**: evaluation uses `P(exists) × P(type=c)` as the class-c confidence. If `exists` and `type` are trained independently, their product is poorly calibrated. Add an auxiliary loss that directly supervises the joint score.

**Formulation**:
$$L_{joint} = \sum_c BCE\big(\sigma(\text{exists\_logit}) \cdot \sigma(\text{type}_c\_\text{logit}),\ exists_{gt} \cdot type_c_{gt} \cdot type_c_{valid}\big)$$

Where $c \in \{cyst, mass, tumor\}$.

**Expected impact**: AP gain 0.01–0.03. Fixes calibration without changing the architecture.

**Implementation**: `losses/joint_conf.py`. Adds ~10 lines to the combined loss function.

**Risks**: minor; may interfere with independent exists training if weight is too high.

---

### M5. No-Object Class with Label Smoothing

**Motivation**: slots with no GT lesion should have low exists_pred. Currently they are just "BCE target 0" but with label smoothing we can make this more robust.

**Formulation**: targets for empty slots become $\epsilon$ (small, e.g., 0.05) instead of 0. This prevents overconfidence collapse.

**Expected impact**: small, AP gain ~0.01. Often combined with M3 (DETR-style).

**Implementation**: trivial, one config flag.

---

### M6. Size-aware positive weighting

**Motivation**: larger lesions are more clinically important (more likely to be tumors, malignant). Weight their BCE loss more.

**Formulation**: for each GT slot with size $s$ cm, `weight = 1 + α·s` where α is a hyperparameter.

**Expected impact**: AP_mass/AP_tumor may improve (larger lesions are in mass/tumor categories). AP_cyst may slightly decrease (most cysts are small).

**Implementation**: modify BCE loss to accept per-sample weights.

**Risks**: may trade off cyst performance for mass/tumor. Not recommended as primary method.

---

### M7. Hard Negative Mining

**Motivation**: explicitly mine the hardest-to-classify negative slots each epoch and weight them up.

**Formulation**:
- At each batch, identify top-K negative slots with highest exists_pred (hardest FPs)
- Apply extra BCE loss on these with weight 2x

**Expected impact**: AP gain 0.02–0.04.

**Implementation**: small change to loss computation. Can combine with M1 (focal) and M3 (Hungarian).

**Risks**: can cause instability if K is too large.

---

### M8. Joint Training with Count Head (for heads without one)

**Motivation**: C-slot-embed has a count head and outperforms on per-lesion AP. D-flatten doesn't have one. Adding a count head to D might unlock the same benefit.

**Formulation**: add `self.count_head = nn.Linear(features, 4 * 2)` predicting per-side count (0–3). Supervise with CE.

**Expected impact**: AP gain 0.02–0.05 for D.

**Implementation**: `models/l3_heads_det.py` — new D variant with count head.

---

## 4. Experimental Matrix

We will test each method on two heads (D-flatten and C-slot-embed), plus selected combinations. Total **~20 experiments**, all runnable in parallel on 16 GPUs.

### Single-method experiments (prio: high)

| Exp | Head | Method | Expected mAP |
|-----|------|--------|--------------|
| E1  | D    | M1 (focal γ=2)     | 0.08–0.10 |
| E2  | D    | M2 (count reg)     | 0.08–0.10 |
| E3  | D    | M3 (Hungarian)     | 0.10–0.15 |
| E4  | D    | M4 (joint conf)    | 0.07–0.08 |
| E5  | C-embed | M1 (focal)      | 0.10–0.12 |
| E6  | C-embed | M2 (count reg ×3) | 0.08–0.10 |
| E7  | C-embed | M3 (Hungarian)  | 0.12–0.17 |

### Combination experiments (prio: medium)

| Exp | Head | Methods | Expected mAP |
|-----|------|---------|--------------|
| E8  | D    | M1 + M2 (focal + count) | 0.10–0.13 |
| E9  | D    | M1 + M3 (focal + Hungarian) | 0.13–0.17 |
| E10 | D    | M3 + M4 (Hungarian + joint conf) | 0.12–0.16 |
| E11 | D    | M1 + M2 + M3 + M4 ("kitchen sink") | 0.15–0.20 |
| E12 | C-embed | M1 + M3 | 0.15–0.20 |
| E13 | C-embed | "kitchen sink" | 0.18–0.25 |

### Ablation experiments (prio: low, run only if main results are promising)

| Exp | Variation |
|-----|-----------|
| E14 | M1 focal γ sweep: {1.0, 1.5, 2.0, 3.0} |
| E15 | M2 count reg weight sweep: {0.5, 1.0, 2.0, 5.0} |
| E16 | M3 matching cost weights (λ₁, λ₂, λ₃) sweep |

### Baseline (must run, for fair comparison)

| Exp | Head | Method |
|-----|------|--------|
| E0a | D    | none (current baseline, Crop+Kmask core_matrix) |
| E0b | C-embed | none (current C-slot-embed) |

These are already run; we'll use existing `epoch_metrics_v3.csv` and `predictions/epoch_*.npz`.

### Note on C-slot-embed

E6 (C-embed + M2 count reg) is **skipped** because C already has a count head — adding M2 is redundant.

### Total budget (with 3 seeds per experiment)

- **13 new experiments** (E1–E13) × **3 seeds each** = **39 training runs**
- Each experiment = **1 sbatch job running 3 seeds sequentially** (50 epochs × 3 seeds ≈ 18–22 hours per job)
- With 16 GPUs in parallel: 13 jobs fit → **wall-clock ~20 hours** (assuming no queue wait)
- With queue contention: likely 1–2 days
- Single SLURM job per experiment simplifies queue management

---

## 5. File Structure

```
stage1_classification/
├── train.py, models/, data/, analysis/  # ── FROZEN (main paper) ──
│
└── detection_training/                  # ── EXPERIMENTAL SANDBOX ──
    ├── PROPOSAL.md                      # this document
    ├── README.md                        # how to run
    │
    ├── train_detection.py               # new entry point, imports from main
    │
    ├── losses/
    │   ├── __init__.py
    │   ├── focal.py                     # M1
    │   ├── count_reg.py                 # M2
    │   ├── hungarian.py                 # M3
    │   ├── joint_conf.py                # M4
    │   ├── label_smooth.py              # M5
    │   ├── size_weight.py               # M6
    │   └── hard_neg.py                  # M7
    │
    ├── models/
    │   ├── __init__.py
    │   └── l3_heads_det.py              # optional new variants (e.g. D with count head for M8)
    │
    ├── configs/
    │   ├── base.yaml                    # same as core_matrix default
    │   ├── focal.yaml                   # E1, E5
    │   ├── count.yaml                   # E2, E6
    │   ├── hungarian.yaml               # E3, E7
    │   ├── joint_conf.yaml              # E4
    │   ├── combined_focal_count.yaml    # E8
    │   ├── combined_focal_hungarian.yaml # E9, E12
    │   ├── kitchen_sink.yaml            # E11, E13
    │   └── ...
    │
    ├── scripts/
    │   ├── submit_wave1.sh              # all single-method experiments
    │   ├── submit_wave2.sh              # combinations
    │   ├── submit_wave3.sh              # ablations (optional)
    │   └── aggregate_results.py         # collect & compare results
    │
    └── results/                         # detection-training-specific results
        ├── summary_table.md             # auto-generated comparison
        └── (per-experiment folders populated after runs)
```

---

## 6. Implementation Principles

1. **Isolate from main code** — no changes to `stage1_classification/train.py`, `models/l3_heads.py`, `data/dataset.py`.
2. **Import, don't fork** — share data loading, encoder, base head architectures via `from models.l3_heads import L3HeadD`.
3. **Each loss is a pure function** — clear input/output, independently unit-testable.
4. **Config-driven** — new YAML config per experiment, no hardcoded hyperparameters.
5. **Fail-fast** — use assertions; surface shape/type errors early.
6. **Reuse evaluation** — `analysis/lesion_level_eval.py` is already the canonical Method D evaluator. Don't re-implement.

---

## 7. Timeline

| Day | Milestone |
|---|---|
| D0  | ✅ Review and approve this proposal |
| D1  | Implement `losses/` modules (M1, M2, M3, M4, plus M5-M8 as needed); unit tests |
| D2  | Write `train_detection.py` with 3-seed loop; smoke-test single experiment (E1) for ~2 epochs |
| D3  | Submit **all 12 experiments at once** (E1–E5, E7–E13, excluding skipped E6); each job runs 3 seeds sequentially |
| D4  | Jobs running (~20 hours); monitor for failures |
| D5  | Jobs finish → aggregate results → compare to baseline |
| D6  | Update PROPOSAL.md with actual numbers; decide on ablations (E14–E16) |

---

## 8. Success Criteria

| Level | Criterion | Implication |
|---|---|---|
| **Minimum success** | Best combined method beats E0 (D baseline, mAP=0.061) by > 0.03 | Worth mentioning in paper discussion as "improved with detection loss" |
| **Good success** | Any method reaches mAP > 0.10, AP_cyst > 0.25 | Publishable follow-up paper |
| **Great success** | Best method reaches mAP > 0.15, AP_cyst > 0.35 | Can add results to main paper's appendix |

If **minimum success** is not achieved, we document the negative result (also valuable for future work / rebuttal).

---

## 9. Open Questions (for user review)

1. **E0b reference** — C-slot-embed isn't really a "baseline" for improvements since it already has count head; should we treat it separately?
2. **Which loss combinations are highest priority?** Currently E9 (focal+Hungarian), E12, E13 are flagged as highest-value. Adjust?
3. **Size tolerance** — evaluation uses size_tol=1cm. Should we also eval at 0.5 and 2cm for completeness?
4. **Training budget** — 30 epochs (matching main paper). Would 50 epochs be worth it to check for saturation?
5. **Seed variance** — main paper runs 3 seeds for scaling law. Should detection experiments also run 3 seeds? That would triple the budget to ~50 runs.

---

## 10. Approval Checklist (waiting on user)

- [ ] Motivation clear?
- [ ] Method list comprehensive (anything to add/drop)?
- [ ] Experimental matrix sized correctly (not too few, not too many)?
- [ ] File structure makes sense?
- [ ] Timeline realistic?
- [ ] Budget acceptable (~17 runs, ~15 hours with 16 GPUs)?
- [ ] Open questions answered?

Once approved, implementation starts.

---

## 11. Post-hoc finding: Head–Loss architectural mismatch (2026-04-14)

### Observation
After cancelling the first batch of runs (E9/E11/E13), a forward-pass sanity
check on the `best_mAP_model.pt` of `E9_D_focal_hungarian_s42` revealed that
the **encoder output is essentially constant across patients**: max |Δ| over
three very different val CTs was 0.0015, versus 1.57 for a fresh SuPreM
encoder. The L3 head's per-slot predictions are therefore identical for every
patient — the "rising AP" observed during training was an artefact of a
constant predictor hitting the cyst base rate (≈0.45) modulated by epoch-level
bias drift.

### Root cause
**HeadD (Flatten-sort) is incompatible with Hungarian matching.**

HeadD maps `encoder(x) ∈ ℝ⁷⁶⁸ → MLP → (6, 11)` where the 6 slots are *positional
outputs of a single MLP*. Slot identity is encoded only by the final linear
layer's position-dependent weights. Under Hungarian matching, the GT-slot
assignment permutes every batch; the same encoder output is asked to map to
different slot targets depending on the permutation. The only stable solution
is for the encoder to emit a constant, letting the MLP's per-slot biases fit
the *average* matching pattern (slot priors). That is exactly what was
observed: slot biases (0.72, 0.62, 0.50, 0.46, 0.55, 0.71) correspond to
prior match rates per slot position.

### Architecture-loss compatibility table

| Head | Slot identity mechanism | Compatible with |
|---|---|---|
| **HeadA (Summary)** | Per-side pooled, not per-lesion | L2-like loss only |
| **HeadB (DETR-like)** | Learnable queries + cross-attention | Hungarian — *if* cross-attn attends to **spatial** features, not a pooled token |
| **HeadC (Count→Attr)** | `slot_embed[i]` concatenated into MLP input | Hungarian (slot_embed breaks permutation symmetry) |
| **HeadD (Flatten)** | MLP-position only | **Positional matching only** |

### Current HeadB is also broken
The existing `L3HeadB` cross-attends over `feat_proj(pooled_features).unsqueeze(1)`
— a **single** token of shape (B, 1, H). This collapses cross-attention to a
per-query linear projection of one global vector, losing the spatial reasoning
that justifies DETR-style architectures. To give HeadB a fair chance, the
SwinUNETR encoder must optionally return its unpooled feature map
`(B, C, D', H', W')`, which is then flattened to `(B, N, H)` for cross-attn.

### Remediation plan
1. **HeadB fix (priority 1)**: add `return_spatial=True` path in `SwinUNETREncoder`;
   update `L3HeadB` to cross-attend over spatial tokens. Smoke-test before
   resubmitting full jobs.
2. **HeadC + Hungarian (priority 2)**: unchanged architecture; should train cleanly
   once HeadB is validated.
3. **HeadD experiments (E1/E2/E8)**: keep positional matching only. Drop the
   four `D_*_hungarian` entries (E3/E9/E10/E11) — architecturally broken.
4. **Smoke test protocol**: before any full job, verify encoder output
   max |Δ| across 3 different CTs stays > 0.1 after 1 epoch. If it drops below
   0.01, the run is collapsed and should be killed.
