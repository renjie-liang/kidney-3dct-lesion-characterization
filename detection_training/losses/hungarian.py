"""M3: DETR-style Hungarian matching loss for L3D lesion prediction.

Reference: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020).

The baseline L3D loss uses positional assignment: slot i in predictions supervises
GT slot i (size-sorted). This creates a tight coupling between training and the
size-sorting convention, and breaks when predictions don't match the sort order.

DETR-style solution:
  1. For each (patient, side), compute cost matrix between 3 pred slots and 3 GT
     slots using a composite cost (size + type).
  2. Hungarian algorithm finds optimal 1-to-1 matching.
  3. Matched pred slots receive supervision from their assigned GT.
  4. Unmatched pred slots receive "no-object" supervision (exists=0, attributes
     masked out since there's no GT to supervise).

Cost function (for matching only):
  C[i,j] = w_size * |size_pred_i - size_gt_j| / size_scale
        + sum_c(w_c * BCE(type_c_pred_i, type_c_gt_j))       (only when GT valid)

Exists cost was removed — after filtering to existing GT only, it becomes
row-constant and has no effect on the Hungarian assignment.

Supervision applied to matched pairs:
  - exists BCE (target = 1) [or focal if use_focal=True]
  - cyst / mass / tumor BCE (only on valid)
  - size MSE (only on valid)
  - enhancement CE (only on valid)  [added in v2]
  - attenuation CE (only on valid)  [added in v2]

Unmatched pred slots:
  - exists BCE (target = 0) [or focal] × no_object_weight
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .focal import focal_bce_loss


MAX_SLOTS = 3
TOTAL_SLOTS = 6


def _bce_cost_matrix(pred_logit: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pairwise BCE cost matrix. pred_logit: (n_pred,), gt: (n_gt,). Returns (n_pred, n_gt)."""
    pred_logit = pred_logit.unsqueeze(1)
    gt = gt.unsqueeze(0)
    return F.binary_cross_entropy_with_logits(
        pred_logit.expand(-1, gt.shape[1]),
        gt.expand(pred_logit.shape[0], -1),
        reduction="none",
    )


def hungarian_match(
    pred_size: torch.Tensor,        # (n_pred,) cm
    pred_type: Dict[str, torch.Tensor],   # {cyst, mass, tumor} → (n_pred,) logits
    gt_size: torch.Tensor,          # (n_gt,) cm
    gt_type: Dict[str, torch.Tensor],     # {cyst, mass, tumor} → (n_gt,) 0/1
    gt_valid: Dict[str, torch.Tensor],    # {cyst, mass, tumor} → (n_gt,) 0/1
    cost_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-side Hungarian matching for training.

    Cost = size L1 (normalized) + type BCE (weighted by clinical priority).
    Exists cost omitted (row-constant after filtering).
    """
    if cost_weights is None:
        cost_weights = {
            "w_size": 1.0,
            "w_cyst": 1.0,
            "w_solid": 1.0,
            "size_scale": 10.0,
        }

    with torch.no_grad():
        size_cost = torch.abs(
            pred_size[:, None] - gt_size[None, :]
        ) * cost_weights["w_size"] / cost_weights["size_scale"]

        type_cost = torch.zeros_like(size_cost)
        for feat, w_key in [("cyst", "w_cyst"), ("solid", "w_solid")]:
            bce = _bce_cost_matrix(pred_type[feat], gt_type[feat].float())
            mask = gt_valid[feat].unsqueeze(0).expand_as(bce)
            type_cost = type_cost + cost_weights[w_key] * bce * mask

        cost = (size_cost + type_cost).cpu().numpy()

    row, col = linear_sum_assignment(cost)
    return row, col


def _binary_loss(logits, targets, use_focal, focal_gamma, focal_alpha):
    """Apply focal loss if enabled, else standard BCE. Returns scalar."""
    if use_focal:
        return focal_bce_loss(logits, targets, gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
    return F.binary_cross_entropy_with_logits(logits, targets)


def hungarian_l3_loss(
    output,
    labels: Dict[str, torch.Tensor],
    cost_weights: Optional[Dict[str, float]] = None,
    no_object_weight: float = 0.1,
    size_loss_weight: float = 0.1,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    exists_loss_weight: float = 2.0,   # D3: boost exists vs type
    type_loss_weight: float = 1.0,
    enh_loss_weight: float = 0.3,
    att_loss_weight: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DETR-style Hungarian set-prediction loss for L3.

    Args:
        output: L3Output-like with fields exists, cyst, mass, tumor, size,
                enhancement (B, 6, 2), attenuation (B, 6, 4). Each (B, 6) or (B, 6, C).
        labels: dict with l3_{side}_{feat} and l3_{side}_{feat}_valid tensors.
        use_focal: if True, apply focal BCE to exists loss (matched + unmatched).
        focal_gamma, focal_alpha: focal params.

    Returns:
        total_loss, loss_dict
    """
    device = output.exists.device
    B = output.exists.shape[0]

    if cost_weights is None:
        cost_weights = {"w_size": 1.0, "w_cyst": 1.0, "w_solid": 1.0, "size_scale": 10.0}

    def gt_stack(key):
        return torch.cat([labels[f"l3_left_{key}"], labels[f"l3_right_{key}"]], dim=1).view(B, TOTAL_SLOTS)

    gt_exists = gt_stack("exists")
    gt_size = gt_stack("size")
    gt_size_valid = gt_stack("size_valid")
    gt_type = {k: gt_stack(k) for k in ["cyst", "solid"]}
    gt_valid = {k: gt_stack(f"{k}_valid") for k in ["cyst", "solid"]}
    gt_enh = gt_stack("enhancement")
    gt_enh_valid = gt_stack("enhancement_valid")
    gt_att = gt_stack("attenuation")
    gt_att_valid = gt_stack("attenuation_valid")

    # Accumulators
    # We collect per-slot contributions and reduce once at the end — fewer graph ops.
    matched_pred_logits_for_exists = []   # list of scalars (logits), target=1
    unmatched_pred_logits_for_exists = []  # list of scalars (logits), target=0
    matched_type_terms = {k: [] for k in ["cyst", "solid"]}   # list of (logit, target)
    matched_size_terms = []       # list of (pred, target)
    matched_enh_terms = []        # list of (pred(2,), target int)
    matched_att_terms = []        # list of (pred(4,), target int)

    n_matched_pos = 0
    n_unmatched_pred = 0

    for b in range(B):
        for sl in [slice(0, 3), slice(3, 6)]:
            p_exists = output.exists[b, sl]
            p_size = output.size[b, sl]
            p_type = {k: getattr(output, k)[b, sl] for k in ["cyst", "solid"]}

            g_exists = gt_exists[b, sl]
            g_size = gt_size[b, sl]
            g_size_valid = gt_size_valid[b, sl]
            g_type = {k: gt_type[k][b, sl] for k in ["cyst", "solid"]}
            g_valid = {k: gt_valid[k][b, sl] for k in ["cyst", "solid"]}
            g_enh = gt_enh[b, sl]; g_enh_valid = gt_enh_valid[b, sl]
            g_att = gt_att[b, sl]; g_att_valid = gt_att_valid[b, sl]

            # Only match against existing GT slots
            gt_exists_mask = (g_exists > 0.5)
            n_gt = int(gt_exists_mask.sum().item())
            n_pred = 3

            if n_gt > 0:
                gt_idx = torch.nonzero(gt_exists_mask, as_tuple=False).squeeze(1)

                # Filtered GT for matching
                g_size_f = g_size[gt_idx]
                g_type_f = {k: g_type[k][gt_idx] for k in ["cyst", "solid"]}
                g_valid_f = {k: g_valid[k][gt_idx] for k in ["cyst", "solid"]}

                row, col = hungarian_match(
                    p_size, p_type, g_size_f, g_type_f, g_valid_f,
                    cost_weights=cost_weights,
                )

                matched_pred_set = set()
                for r_i, c_i in zip(row, col):
                    p_i = int(r_i)
                    gt_slot = int(gt_idx[int(c_i)].item())
                    matched_pred_set.add(p_i)
                    n_matched_pos += 1

                    # Matched exists (target=1)
                    matched_pred_logits_for_exists.append(p_exists[p_i])

                    # Type losses (only on valid)
                    for feat in ["cyst", "solid"]:
                        if g_valid[feat][gt_slot] > 0.5:
                            matched_type_terms[feat].append(
                                (p_type[feat][p_i], g_type[feat][gt_slot].float())
                            )

                    # Size loss (only on valid)
                    if g_size_valid[gt_slot] > 0.5:
                        matched_size_terms.append(
                            (p_size[p_i], g_size[gt_slot].float())
                        )

                    # Enhancement (only on valid)
                    if g_enh_valid[gt_slot] > 0.5:
                        matched_enh_terms.append(
                            (output.enhancement[b, sl.start + p_i],
                             int(round(float(g_enh[gt_slot].item()))))
                        )

                    # Attenuation (only on valid)
                    if g_att_valid[gt_slot] > 0.5:
                        matched_att_terms.append(
                            (output.attenuation[b, sl.start + p_i],
                             int(round(float(g_att[gt_slot].item()))))
                        )

                # Unmatched preds → no-object (exists=0)
                for p_i in range(n_pred):
                    if p_i not in matched_pred_set:
                        unmatched_pred_logits_for_exists.append(p_exists[p_i])
                        n_unmatched_pred += 1
            else:
                # No GT → all preds are no-object
                for p_i in range(n_pred):
                    unmatched_pred_logits_for_exists.append(p_exists[p_i])
                    n_unmatched_pred += 1

    # ─────── Compute each loss component as a single batched op ───────
    comps = {"n_matched_pos": n_matched_pos, "n_unmatched_pred": n_unmatched_pred}

    # Exists (matched pairs: target=1, unmatched: target=0, downweighted)
    exists_loss_terms = []
    if matched_pred_logits_for_exists:
        logits_matched = torch.stack(matched_pred_logits_for_exists)
        targets_matched = torch.ones_like(logits_matched)
        loss_m = _binary_loss(logits_matched, targets_matched, use_focal, focal_gamma, focal_alpha)
        exists_loss_terms.append(loss_m)
        comps["exists_matched"] = loss_m.item()
    if unmatched_pred_logits_for_exists:
        logits_un = torch.stack(unmatched_pred_logits_for_exists)
        targets_un = torch.zeros_like(logits_un)
        loss_u = _binary_loss(logits_un, targets_un, use_focal, focal_gamma, focal_alpha)
        exists_loss_terms.append(no_object_weight * loss_u)
        comps["exists_unmatched"] = loss_u.item()
    exists_loss = sum(exists_loss_terms) if exists_loss_terms else torch.tensor(0.0, device=device)

    # Type losses (mean over valid matched pairs, per feat)
    type_loss_total = torch.tensor(0.0, device=device)
    for feat in ["cyst", "solid"]:
        if matched_type_terms[feat]:
            p = torch.stack([t[0] for t in matched_type_terms[feat]])
            g = torch.stack([t[1] for t in matched_type_terms[feat]])
            loss = F.binary_cross_entropy_with_logits(p, g)
            type_loss_total = type_loss_total + loss
            comps[feat] = loss.item()
        else:
            comps[feat] = 0.0

    # Size loss
    if matched_size_terms:
        p_sz = torch.stack([t[0] for t in matched_size_terms])
        g_sz = torch.stack([t[1] for t in matched_size_terms])
        size_loss = F.mse_loss(p_sz, g_sz)
        comps["size"] = size_loss.item()
    else:
        size_loss = torch.tensor(0.0, device=device)
        comps["size"] = 0.0

    # Enhancement (categorical CE, 2 classes)
    if matched_enh_terms:
        p_enh = torch.stack([t[0] for t in matched_enh_terms])       # (N, 2)
        g_enh_idx = torch.tensor([t[1] for t in matched_enh_terms], device=device).clamp(0, 1)
        enh_loss = F.cross_entropy(p_enh, g_enh_idx)
        comps["enhancement"] = enh_loss.item()
    else:
        enh_loss = torch.tensor(0.0, device=device)
        comps["enhancement"] = 0.0

    # Attenuation (categorical CE, 4 classes)
    if matched_att_terms:
        p_att = torch.stack([t[0] for t in matched_att_terms])       # (N, 4)
        g_att_idx = torch.tensor([t[1] for t in matched_att_terms], device=device).clamp(0, 3)
        att_loss = F.cross_entropy(p_att, g_att_idx)
        comps["attenuation"] = att_loss.item()
    else:
        att_loss = torch.tensor(0.0, device=device)
        comps["attenuation"] = 0.0

    # D3: weighted loss assembly. exists gets extra boost since detection is
    # usually under-weighted relative to type classification (3 classes × BCE).
    total = (
        exists_loss_weight * exists_loss
        + type_loss_weight * type_loss_total
        + size_loss_weight * size_loss
        + enh_loss_weight * enh_loss
        + att_loss_weight * att_loss
    )
    return total, comps


if __name__ == "__main__":
    from types import SimpleNamespace
    B = 4

    out = SimpleNamespace(
        exists=torch.randn(B, 6, requires_grad=True),
        cyst=torch.randn(B, 6, requires_grad=True),
        solid=torch.randn(B, 6, requires_grad=True),
        size=torch.rand(B, 6, requires_grad=True) * 5,
        enhancement=torch.randn(B, 6, 2, requires_grad=True),
        attenuation=torch.randn(B, 6, 4, requires_grad=True),
    )
    labels = {}
    for side in ["left", "right"]:
        labels[f"l3_{side}_exists"] = torch.randint(0, 2, (B, 3)).float()
        for feat in ["cyst", "solid"]:
            labels[f"l3_{side}_{feat}"] = torch.randint(0, 2, (B, 3)).float()
            labels[f"l3_{side}_{feat}_valid"] = torch.ones(B, 3)
        labels[f"l3_{side}_size"] = torch.rand(B, 3) * 5
        labels[f"l3_{side}_size_valid"] = torch.ones(B, 3)
        labels[f"l3_{side}_enhancement"] = torch.randint(0, 2, (B, 3)).float()
        labels[f"l3_{side}_enhancement_valid"] = torch.ones(B, 3)
        labels[f"l3_{side}_attenuation"] = torch.randint(0, 4, (B, 3)).float()
        labels[f"l3_{side}_attenuation_valid"] = torch.ones(B, 3)

    print("=== Without focal ===")
    total, d = hungarian_l3_loss(out, labels, use_focal=False)
    print(f"Total: {total.item():.4f}")
    print(f"Components: {d}")

    print("\n=== With focal ===")
    total, d = hungarian_l3_loss(out, labels, use_focal=True, focal_gamma=2.0, focal_alpha=0.25)
    print(f"Total: {total.item():.4f}")
    print(f"Components: {d}")

    total.backward()
    print("\nBackprop works.")
    print("✓ hungarian.py smoke test passed (with focal + enh + att)")
