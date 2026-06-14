"""Combined L3 loss that assembles M1-M7 based on a config dict.

This is the "recipe" file: given config flags, it mixes base BCE + any enabled
methods, weighted by user-specified coefficients.

Two modes:
  mode="positional": baseline-style positional BCE + optional M1/M2/M4/M6/M7
                     (M1 focal, M6 size weighting, M7 hard-neg mining can be
                      COMBINED as multiplicative weight modifiers on BCE)
  mode="hungarian":  DETR-style Hungarian matching loss (M3) — replaces the
                     positional BCE. M1 (focal) applies to matched + unmatched
                     exists loss. M2/M4 are auxiliaries.

Hier supervision (noisy-OR side-level, from main paper) can be toggled via
hier_weight config; defaults to 0.3 with "l1" mode to match core_matrix.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .focal import focal_bce_loss
from .count_reg import count_reg_loss
from .hungarian import hungarian_l3_loss
from .joint_conf import joint_conf_loss
from .label_smooth import smooth_labels

TOTAL_SLOTS = 6


def _gt_stack(labels, key, B):
    """Stack left/right per-side labels into (B, 6)."""
    return torch.cat([labels[f"l3_left_{key}"], labels[f"l3_right_{key}"]], dim=1).view(B, TOTAL_SLOTS)


def _positional_exists_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gt_size: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    """Compute positional exists loss combining optional modifiers.

    Modifiers (independent, combinable as multiplicative weights on per-sample BCE):
      - M1 focal: weight = alpha_t * (1-p_t)^gamma
      - M6 size weight: weight *= (1 + size_alpha * gt_size * targets)
      - M7 hard negative mining: weight *= (1 + extra_weight * hard_neg_mask)
      - M5 label smoothing: applied to targets, not a weight

    If no modifier is enabled → plain BCE(reduction=mean).
    """
    # M5: label smoothing first (modifies targets)
    if cfg.get("use_label_smooth", False):
        targets = smooth_labels(targets, cfg.get("label_smooth_epsilon", 0.05))

    # Base BCE (per-element, no reduction)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # M1: focal modifier = alpha_t * (1 - p_t)^gamma
    weight = torch.ones_like(bce)
    if cfg.get("use_focal_exists", False):
        gamma = cfg.get("focal_gamma", 2.0)
        alpha = cfg.get("focal_alpha", 0.25)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        weight = weight * alpha_t * (1 - p_t) ** gamma

    # M6: size-aware weight (lifts positive examples proportional to size)
    if cfg.get("use_size_weight", False):
        size_alpha = cfg.get("size_weight_alpha", 0.3)
        weight = weight * (1.0 + size_alpha * gt_size * targets)

    # M7: hard-negative mining — boost weight on top-K hardest negatives per sample
    if cfg.get("use_hard_neg", False):
        top_k = cfg.get("hard_neg_top_k", 3)
        extra_w = cfg.get("hard_neg_extra_weight", 1.0)
        neg_mask = (targets < 0.5)
        # find hardest negatives per row (highest BCE among negatives)
        neg_bce = bce.clone()
        neg_bce[~neg_mask] = -1.0
        k_eff = min(top_k, bce.shape[1])
        _, topk_idx = neg_bce.topk(k_eff, dim=1)
        hard_mask = torch.zeros_like(bce)
        hard_mask.scatter_(1, topk_idx, 1.0)
        hard_mask = hard_mask * neg_mask.float()
        weight = weight * (1.0 + extra_w * hard_mask)

    return (bce * weight).mean()


def _noisy_or_hier_loss(output, labels, hier_mode: str):
    """Noisy-OR side-level supervision, copied from main paper's compute_l3_loss."""
    eps = 1e-7
    exists_probs = torch.sigmoid(output.exists)
    left_abn_prob = (1 - (1 - exists_probs[:, :3]).prod(dim=1)).clamp(eps, 1 - eps)
    right_abn_prob = (1 - (1 - exists_probs[:, 3:]).prod(dim=1)).clamp(eps, 1 - eps)
    gt_left_abn = labels["left_abnormal"]
    gt_right_abn = labels["right_abnormal"]
    hier_l1 = (
        -(gt_left_abn * torch.log(left_abn_prob) + (1 - gt_left_abn) * torch.log(1 - left_abn_prob)).mean()
        - (gt_right_abn * torch.log(right_abn_prob) + (1 - gt_right_abn) * torch.log(1 - right_abn_prob)).mean()
    ) / 2

    total = hier_l1
    comps = {"hier_l1": hier_l1.item()}

    if hier_mode != "l1":
        for feat_name, feat_logits in [("cyst", output.cyst), ("solid", output.solid)]:
            probs = torch.sigmoid(feat_logits)
            for side_name, sl in [("left", slice(0, 3)), ("right", slice(3, 6))]:
                side_prob = (1 - (1 - probs[:, sl]).prod(dim=1)).clamp(eps, 1 - eps)
                gt_side = labels[f"{side_name}_has_{feat_name}"]
                loss_val = -(gt_side * torch.log(side_prob) +
                             (1 - gt_side) * torch.log(1 - side_prob)).mean()
                total = total + loss_val / 2
        comps["hier_type"] = (total - hier_l1).item()

    return total, comps


def compute_detection_loss(
    output,
    labels: Dict,
    cfg: Dict,
) -> Tuple[torch.Tensor, Dict]:
    """Assemble detection-oriented loss from config.

    cfg keys:
      mode: "positional" | "hungarian"
      hier_weight: float (0 disables hier loss)
      hier_mode: "l1" | "l1l2"

      # Methods applicable to both modes:
      use_focal_exists: bool
      focal_gamma, focal_alpha

      # Auxiliary losses (both modes):
      use_count_reg: bool
      count_reg_variant, count_reg_weight
      use_joint_conf: bool
      joint_conf_weight

      # Positional mode only (combinable as multiplicative modifiers):
      use_label_smooth: bool
      label_smooth_epsilon
      use_size_weight: bool
      size_weight_alpha
      use_hard_neg: bool
      hard_neg_top_k, hard_neg_extra_weight

      # Hungarian mode only:
      hungarian_cost_weights: dict
      hungarian_no_object_weight, hungarian_size_loss_weight
    """
    device = output.exists.device
    B = output.exists.shape[0]
    total = torch.tensor(0.0, device=device)
    comps = {}

    gt_exists = _gt_stack(labels, "exists", B)
    gt_size = _gt_stack(labels, "size", B)
    gt_size_valid = _gt_stack(labels, "size_valid", B)
    gt_type = {k: _gt_stack(labels, k, B) for k in ["cyst", "solid"]}
    gt_valid = {k: _gt_stack(labels, f"{k}_valid", B) for k in ["cyst", "solid"]}

    mode = cfg.get("mode", "positional")

    # ─────── Core loss (mode-dependent) ───────
    if mode == "hungarian":
        # D7: In Hungarian partition setup, focal alpha doesn't provide class balance
        # (matched = all pos, unmatched = all neg). Use alpha=1.0 by default so that
        # alpha has no effect; only gamma's focusing applies.
        focal_alpha_hung = cfg.get("focal_alpha_hungarian", 1.0)
        hungarian_total, h_comps = hungarian_l3_loss(
            output, labels,
            cost_weights=cfg.get("hungarian_cost_weights"),
            no_object_weight=cfg.get("hungarian_no_object_weight", 0.1),
            size_loss_weight=cfg.get("hungarian_size_loss_weight", 0.1),
            use_focal=cfg.get("use_focal_exists", False),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            focal_alpha=focal_alpha_hung,
            # D3: rebalance loss scales
            exists_loss_weight=cfg.get("exists_loss_weight", 2.0),
            type_loss_weight=cfg.get("type_loss_weight", 1.0),
            enh_loss_weight=cfg.get("enh_loss_weight", 0.3),
            att_loss_weight=cfg.get("att_loss_weight", 0.3),
        )
        total = total + hungarian_total
        # D9: unified naming — drop hung_ prefix
        comps.update(h_comps)

    elif mode == "positional":
        # D3 weights for rebalancing
        w_exists = cfg.get("exists_loss_weight", 2.0)
        w_type = cfg.get("type_loss_weight", 1.0)
        w_size = cfg.get("size_loss_weight_pos", 0.1)
        w_enh = cfg.get("enh_loss_weight", 0.3)
        w_att = cfg.get("att_loss_weight", 0.3)

        # Combinable exists loss (focal/size_weight/hard_neg/label_smooth modifiers)
        exists_loss = _positional_exists_loss(output.exists, gt_exists, gt_size, cfg)
        total = total + w_exists * exists_loss
        comps["exists"] = exists_loss.item()

        # Type losses (masked)
        exist_mask = gt_exists.bool()
        for name, pred, gt, valid in [
            ("cyst", output.cyst, gt_type["cyst"], gt_valid["cyst"]),
            ("solid", output.solid, gt_type["solid"], gt_valid["solid"]),
        ]:
            m = exist_mask & (valid > 0.5)
            if m.any():
                loss = F.binary_cross_entropy_with_logits(pred[m], gt[m].float())
                total = total + w_type * loss
                comps[name] = loss.item()

        # Size loss (MSE, masked)
        size_mask = exist_mask & (gt_size_valid > 0.5)
        if size_mask.any():
            size_loss = F.mse_loss(output.size[size_mask], gt_size[size_mask].float())
            total = total + w_size * size_loss
            comps["size"] = size_loss.item()

        # Enhancement / attenuation (masked CE, from main paper)
        gt_enh = _gt_stack(labels, "enhancement", B)
        gt_enh_valid = _gt_stack(labels, "enhancement_valid", B)
        enh_mask = exist_mask & (gt_enh_valid > 0.5)
        if enh_mask.any():
            enh_pred = output.enhancement[enh_mask]
            enh_gt = gt_enh[enh_mask].round().long().clamp(0, 1)
            if enh_pred.shape[0] > 0 and enh_pred.dim() == 2:
                enh_loss = F.cross_entropy(enh_pred, enh_gt)
                total = total + w_enh * enh_loss
                comps["enhancement"] = enh_loss.item()

        gt_att = _gt_stack(labels, "attenuation", B)
        gt_att_valid = _gt_stack(labels, "attenuation_valid", B)
        att_mask = exist_mask & (gt_att_valid > 0.5)
        if att_mask.any():
            att_pred = output.attenuation[att_mask]
            att_gt = gt_att[att_mask].round().long().clamp(0, 3)
            if att_pred.shape[0] > 0 and att_pred.dim() == 2:
                att_loss = F.cross_entropy(att_pred, att_gt)
                total = total + w_att * att_loss
                comps["attenuation"] = att_loss.item()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ─────── Auxiliary losses (mode-independent) ───────
    if cfg.get("use_count_reg", False):
        cr_loss = count_reg_loss(
            output.exists, gt_exists,
            variant=cfg.get("count_reg_variant", "soft_l1"),
        )
        w = cfg.get("count_reg_weight", 0.5)
        total = total + w * cr_loss
        comps["count_reg"] = cr_loss.item()

    if cfg.get("use_joint_conf", False):
        jc_loss = joint_conf_loss(
            output.exists,
            {k: getattr(output, k) for k in ["cyst", "solid"]},
            gt_exists, gt_type, gt_valid,
        )
        w = cfg.get("joint_conf_weight", 0.3)
        total = total + w * jc_loss
        comps["joint_conf"] = jc_loss.item()

    # ─────── Hierarchical (noisy-OR) side-level loss ───────
    hier_w = cfg.get("hier_weight", 0.3)
    if hier_w > 0:
        hier_total, hier_comps = _noisy_or_hier_loss(output, labels, cfg.get("hier_mode", "l1"))
        total = total + hier_w * hier_total
        comps.update(hier_comps)

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
        labels[f"{side}_abnormal"] = torch.randint(0, 2, (B,)).float()
        labels[f"{side}_has_cyst"] = torch.randint(0, 2, (B,)).float()
        labels[f"{side}_has_solid"] = torch.randint(0, 2, (B,)).float()

    # Test all critical configurations
    test_cfgs = [
        # positional baseline
        ({"mode": "positional"}, "positional BCE only"),
        # positional + focal
        ({"mode": "positional", "use_focal_exists": True}, "positional + focal"),
        # positional + combined modifiers (focal + size_weight + hard_neg)
        ({"mode": "positional", "use_focal_exists": True,
          "use_size_weight": True, "use_hard_neg": True}, "positional + focal+size_weight+hard_neg"),
        # positional + all aux
        ({"mode": "positional", "use_focal_exists": True,
          "use_count_reg": True, "use_joint_conf": True}, "positional + focal+count+joint"),
        # hungarian baseline
        ({"mode": "hungarian"}, "hungarian BCE"),
        # hungarian + focal (THE KEY FIX — E9/E12)
        ({"mode": "hungarian", "use_focal_exists": True}, "hungarian + focal ★"),
        # hungarian kitchen sink (E11)
        ({"mode": "hungarian", "use_focal_exists": True,
          "use_count_reg": True, "use_joint_conf": True}, "hungarian + focal+count+joint ★"),
    ]

    for cfg, name in test_cfgs:
        total, comps = compute_detection_loss(out, labels, cfg)
        print(f"{name:55s}: total={total.item():.4f}, {len(comps)} comps")

    print("\n✓ combined.py smoke test passed (all method combinations)")
