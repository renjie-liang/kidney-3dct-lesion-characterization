"""
Label L3 classification heads — 4 approaches for per-lesion prediction.

All heads take encoder features (B, D) and output per-lesion predictions.
Max 3 lesions per side, 2 sides = 6 slots total.

Per-lesion features:
  exists: bool
  cyst, mass, tumor: bool (with valid mask)
  size_cm: float (with valid mask)
  enhancement: categorical 2-class (with valid mask)
  attenuation: categorical 4-class (with valid mask)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


MAX_SLOTS = 3  # per side
N_SIDES = 2
TOTAL_SLOTS = MAX_SLOTS * N_SIDES  # 6


class L3Output:
    """Container for L3 predictions."""

    def __init__(self, exists, cyst, mass, tumor, size, enhancement, attenuation):
        self.exists = exists              # (B, 6)
        self.cyst = cyst                  # (B, 6)
        self.mass = mass                  # (B, 6)
        self.tumor = tumor                # (B, 6)
        self.size = size                  # (B, 6)
        self.enhancement = enhancement    # (B, 6, 2)
        self.attenuation = attenuation    # (B, 6, 4)


# ─────────────────────────────────────────────
# Approach A: Per-side Summary
# ─────────────────────────────────────────────

class L3HeadA(nn.Module):
    """Per-side summary: count + aggregated features per side.
    Not truly per-lesion, but provides side-level summary."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        )
        # Per side: count(4) + cyst + mass + tumor + size + enhancement(2) + attenuation(4) = 13
        self.side_head = nn.Linear(hidden_dim, 13)

    def forward(self, features):
        h = self.shared(features)
        # Split output for left and right
        left = self.side_head(h)
        right = self.side_head(h)  # shared weights for both sides

        # Expand to slot format for unified evaluation
        return self._to_slot_format(left, right)

    def _to_slot_format(self, left, right):
        """Convert per-side summary to slot format (B, 6, ...)."""
        B = left.shape[0]
        device = left.device

        all_exists = []
        all_cyst, all_mass, all_tumor, all_size = [], [], [], []
        all_enh, all_att = [], []

        for side_out in [left, right]:
            count_logits = side_out[:, :4]   # (B, 4) for count 0-3
            cyst = side_out[:, 4:5]          # (B, 1)
            mass = side_out[:, 5:6]
            tumor = side_out[:, 6:7]
            size = side_out[:, 7:8]
            enh = side_out[:, 8:10]          # (B, 2)
            att = side_out[:, 10:13]         # pad to 4
            att = F.pad(att, (0, 1))         # (B, 4)

            # Replicate to 3 slots (same prediction for all slots in this side)
            for _ in range(MAX_SLOTS):
                all_exists.append(count_logits[:, 1:].sum(dim=1, keepdim=True))  # P(count >= 1)
                all_cyst.append(cyst)
                all_mass.append(mass)
                all_tumor.append(tumor)
                all_size.append(size)
                all_enh.append(enh)
                all_att.append(att)

        return L3Output(
            exists=torch.cat(all_exists, dim=1),
            cyst=torch.cat(all_cyst, dim=1),
            mass=torch.cat(all_mass, dim=1),
            tumor=torch.cat(all_tumor, dim=1),
            size=torch.cat(all_size, dim=1),
            enhancement=torch.stack(all_enh, dim=1),
            attenuation=torch.stack(all_att, dim=1),
        ), {"left_count_logits": left[:, :4], "right_count_logits": right[:, :4]}


# ─────────────────────────────────────────────
# Approach B: Fixed Slots (DETR-like)
# ─────────────────────────────────────────────

class L3HeadB(nn.Module):
    """DETR-like: learnable queries cross-attend to spatial encoder tokens.

    Forward signature accepts EITHER:
      features: (B, D)                    — pooled features (legacy path, not recommended)
      features: tuple(pooled, tokens)     — (B, D) pooled + (B, N, D) spatial tokens
                                            when encoder is called with return_spatial=True.

    When spatial tokens are provided, cross-attention gives each query a distinct
    spatial view, which is what DETR-style heads need to avoid constant-encoder
    collapse under Hungarian matching.
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(TOTAL_SLOTS, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.feat_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        )
        # Per slot: exists + cyst + mass + tumor + size + enhancement(2) + attenuation(4) = 11
        self.out_proj = nn.Linear(hidden_dim, 11)

    def forward(self, features):
        if isinstance(features, tuple):
            pooled, tokens = features  # (B, D), (B, N, D)
        else:
            pooled = features
            tokens = features.unsqueeze(1)  # (B, 1, D) — degenerate fallback
        B = pooled.shape[0]

        kv = self.norm_kv(self.feat_proj(tokens))  # (B, N, H)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, 6, H)
        queries = self.norm_q(self.query_proj(queries))

        refined, _ = self.cross_attn(queries, kv, kv)  # (B, 6, H)
        h = self.slot_head(refined)
        out = self.out_proj(h)  # (B, 6, 11)

        return L3Output(
            exists=out[:, :, 0],
            cyst=out[:, :, 1],
            mass=out[:, :, 2],
            tumor=out[:, :, 3],
            size=out[:, :, 4],
            enhancement=out[:, :, 5:7],
            attenuation=out[:, :, 7:11],
        ), {}


# ─────────────────────────────────────────────
# Approach C: Count → Attributes
# ─────────────────────────────────────────────

class L3HeadC(nn.Module):
    """Two-step: predict count, then per-slot attributes."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # Count predictor (per side)
        self.count_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, 8),  # 4 classes × 2 sides
        )

        # Count embedding
        self.count_embed = nn.Embedding(4, 32)  # 0-3

        # Slot index embedding — differentiates slots within a side
        self.slot_embed = nn.Embedding(MAX_SLOTS, 32)

        # Slot predictor (input = features + count_emb + slot_emb)
        self.slot_head = nn.Sequential(
            nn.Linear(input_dim + 32 + 32, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        )
        # Per slot: exists + cyst + mass + tumor + size + enhancement(2) + attenuation(4) = 11
        self.slot_out = nn.Linear(hidden_dim, 11)

    def forward(self, features):
        B = features.shape[0]

        # Step 1: predict counts
        count_logits = self.count_head(features)  # (B, 8)
        left_count_logits = count_logits[:, :4]
        right_count_logits = count_logits[:, 4:]

        # Use predicted count as embedding (soft: weighted sum)
        left_count_probs = F.softmax(left_count_logits, dim=-1)  # (B, 4)
        right_count_probs = F.softmax(right_count_logits, dim=-1)

        indices = torch.arange(4, device=features.device)
        left_count_emb = (left_count_probs.unsqueeze(-1) * self.count_embed(indices)).sum(dim=1)  # (B, 32)
        right_count_emb = (right_count_probs.unsqueeze(-1) * self.count_embed(indices)).sum(dim=1)

        # Step 2: per-slot prediction (each slot gets unique slot index embedding)
        slot_indices = torch.arange(MAX_SLOTS, device=features.device)
        slot_embs = self.slot_embed(slot_indices)  # (3, 32)

        slots = []
        for count_emb in [left_count_emb, right_count_emb]:
            for s in range(MAX_SLOTS):
                s_emb = slot_embs[s].unsqueeze(0).expand(B, -1)   # (B, 32)
                feat_in = torch.cat([features, count_emb, s_emb], dim=-1)  # (B, D+64)
                h = self.slot_head(feat_in)
                slots.append(self.slot_out(h))

        out = torch.stack(slots, dim=1)  # (B, 6, 11)

        return L3Output(
            exists=out[:, :, 0],
            cyst=out[:, :, 1],
            mass=out[:, :, 2],
            tumor=out[:, :, 3],
            size=out[:, :, 4],
            enhancement=out[:, :, 5:7],
            attenuation=out[:, :, 7:11],
        ), {"left_count_logits": left_count_logits, "right_count_logits": right_count_logits}


# ─────────────────────────────────────────────
# Approach D: Flatten (Ordered by size)
# ─────────────────────────────────────────────

class L3HeadD(nn.Module):
    """Direct regression: flatten all slots into a single vector."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # 6 slots × 10 features = 60
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, TOTAL_SLOTS * 11),
        )

    def forward(self, features):
        out = self.head(features).view(-1, TOTAL_SLOTS, 11)  # (B, 6, 11)

        return L3Output(
            exists=out[:, :, 0],
            cyst=out[:, :, 1],
            mass=out[:, :, 2],
            tumor=out[:, :, 3],
            size=out[:, :, 4],
            enhancement=out[:, :, 5:7],
            attenuation=out[:, :, 7:11],
        ), {}


# ─────────────────────────────────────────────
# L3 Loss with masked loss + Hungarian matching
# ─────────────────────────────────────────────

@torch.no_grad()
def _hungarian_match_per_side(output: L3Output, gt_exists: torch.Tensor, gt_size: torch.Tensor) -> torch.Tensor:
    """Compute per-side Hungarian matching permutation for approach B.

    Within each side (3 queries vs 3 GT slots), find optimal assignment
    minimizing combined existence + size cost.

    Args:
        output: L3Output with (B, 6) tensors
        gt_exists: (B, 6) binary existence labels
        gt_size: (B, 6) size labels in cm
    Returns:
        perm: (B, 6) long tensor — perm[b, i] = GT slot index that query i should match
              slots 0..2 permuted among 0..2, slots 3..5 permuted among 3..5
    """
    import numpy as np
    B = output.exists.shape[0]
    device = output.exists.device
    pred_exists = torch.sigmoid(output.exists).detach()   # (B, 6)
    pred_size = output.size.detach()                      # (B, 6)

    def side_cost(sl):
        p_e = pred_exists[:, sl]                          # (B, 3)
        p_s = pred_size[:, sl]
        g_e = gt_exists[:, sl]
        g_s = gt_size[:, sl]
        # cost[b, i, j] = |p_e_i - g_e_j| + 0.1 * g_e_j * |p_s_i - g_s_j|
        c_e = (p_e.unsqueeze(2) - g_e.unsqueeze(1)).abs()              # (B, 3, 3)
        c_s = (p_s.unsqueeze(2) - g_s.unsqueeze(1)).abs() * 0.1
        c_s = c_s * g_e.unsqueeze(1)                                    # only penalize size when GT exists
        return (c_e + c_s).cpu().numpy()

    cost_l = side_cost(slice(0, 3))
    cost_r = side_cost(slice(3, 6))

    perm = np.tile(np.arange(TOTAL_SLOTS), (B, 1))                      # (B, 6) identity default
    for b in range(B):
        _, col_l = linear_sum_assignment(cost_l[b])
        _, col_r = linear_sum_assignment(cost_r[b])
        perm[b, 0:3] = col_l                       # 0..2
        perm[b, 3:6] = col_r + 3                   # shift by 3 for right side indexing
    return torch.from_numpy(perm).long().to(device)


def _apply_perm(t: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute last meaningful dim of a (B, 6) or (B, 6, ...) tensor by perm (B, 6)."""
    if t.dim() == 2:
        return torch.gather(t, 1, perm)
    # For (B, 6, C) tensors — expand perm
    rest = t.shape[2:]
    perm_exp = perm.view(*perm.shape, *([1] * len(rest))).expand(-1, -1, *rest)
    return torch.gather(t, 1, perm_exp)


def compute_l3_loss(output: L3Output, extra: dict, labels: dict, approach: str,
                    hier_weight: float = 1.0, hier_mode: str = "l1l2"):
    """Compute L3 loss with masked loss for unknown features.

    For approach B: uses per-side Hungarian matching (queries 0-2 match left GT
                    slots, queries 3-5 match right GT slots; within each side the
                    3×3 bipartite assignment is solved to minimize cost).
    For approach A/C: uses count loss.
    For approach D: uses ordered (positional) matching with size-sorted slots.
    """
    device = output.exists.device
    B = output.exists.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    # Gather GT into (B, 6, ...) format: [left_0, left_1, left_2, right_0, right_1, right_2]
    gt_exists = torch.stack([labels["l3_left_exists"], labels["l3_right_exists"]], dim=1).view(B, TOTAL_SLOTS)
    gt_cyst = torch.stack([labels["l3_left_cyst"], labels["l3_right_cyst"]], dim=1).view(B, TOTAL_SLOTS)
    gt_cyst_v = torch.stack([labels["l3_left_cyst_valid"], labels["l3_right_cyst_valid"]], dim=1).view(B, TOTAL_SLOTS)
    gt_mass = torch.stack([labels["l3_left_mass"], labels["l3_right_mass"]], dim=1).view(B, TOTAL_SLOTS)
    gt_mass_v = torch.stack([labels["l3_left_mass_valid"], labels["l3_right_mass_valid"]], dim=1).view(B, TOTAL_SLOTS)
    gt_tumor = torch.stack([labels["l3_left_tumor"], labels["l3_right_tumor"]], dim=1).view(B, TOTAL_SLOTS)
    gt_tumor_v = torch.stack([labels["l3_left_tumor_valid"], labels["l3_right_tumor_valid"]], dim=1).view(B, TOTAL_SLOTS)
    gt_size = torch.stack([labels["l3_left_size"], labels["l3_right_size"]], dim=1).view(B, TOTAL_SLOTS)
    gt_size_v = torch.stack([labels["l3_left_size_valid"], labels["l3_right_size_valid"]], dim=1).view(B, TOTAL_SLOTS)
    gt_enh = torch.stack([labels["l3_left_enhancement"], labels["l3_right_enhancement"]], dim=1).view(B, TOTAL_SLOTS)
    gt_enh_v = torch.stack([labels["l3_left_enhancement_valid"], labels["l3_right_enhancement_valid"]], dim=1).view(B, TOTAL_SLOTS)
    gt_att = torch.stack([labels["l3_left_attenuation"], labels["l3_right_attenuation"]], dim=1).view(B, TOTAL_SLOTS)
    gt_att_v = torch.stack([labels["l3_left_attenuation_valid"], labels["l3_right_attenuation_valid"]], dim=1).view(B, TOTAL_SLOTS)

    # --- Approach B: per-side Hungarian reordering of GT ---
    if approach == "B":
        perm = _hungarian_match_per_side(output, gt_exists, gt_size)
        gt_exists = _apply_perm(gt_exists, perm)
        gt_cyst = _apply_perm(gt_cyst, perm)
        gt_cyst_v = _apply_perm(gt_cyst_v, perm)
        gt_mass = _apply_perm(gt_mass, perm)
        gt_mass_v = _apply_perm(gt_mass_v, perm)
        gt_tumor = _apply_perm(gt_tumor, perm)
        gt_tumor_v = _apply_perm(gt_tumor_v, perm)
        gt_size = _apply_perm(gt_size, perm)
        gt_size_v = _apply_perm(gt_size_v, perm)
        gt_enh = _apply_perm(gt_enh, perm)
        gt_enh_v = _apply_perm(gt_enh_v, perm)
        gt_att = _apply_perm(gt_att, perm)
        gt_att_v = _apply_perm(gt_att_v, perm)

    # Exists loss (always valid)
    exists_loss = F.binary_cross_entropy_with_logits(output.exists, gt_exists)
    total_loss = total_loss + exists_loss
    loss_dict["exists"] = exists_loss.item()

    # Feature losses (masked: only on existing + valid slots)
    exist_mask = gt_exists.bool()

    for name, pred, gt, valid in [
        ("cyst", output.cyst, gt_cyst, gt_cyst_v),
        ("mass", output.mass, gt_mass, gt_mass_v),
        ("tumor", output.tumor, gt_tumor, gt_tumor_v),
    ]:
        mask = exist_mask & valid.bool()
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(pred[mask], gt[mask])
            total_loss = total_loss + loss
            loss_dict[name] = loss.item()

    # Size (MSE, masked)
    size_mask = exist_mask & gt_size_v.bool()
    if size_mask.any():
        size_loss = F.mse_loss(output.size[size_mask], gt_size[size_mask])
        total_loss = total_loss + 0.1 * size_loss
        loss_dict["size"] = size_loss.item()

    # Enhancement (CE, masked) — output.enhancement is (B, 6, 2)
    enh_mask = exist_mask & gt_enh_v.bool()
    if enh_mask.sum() > 0:
        enh_pred = output.enhancement[enh_mask]  # (N, 2)
        enh_gt = gt_enh[enh_mask].round().long().clamp(0, 1)
        if enh_pred.shape[0] > 0 and enh_pred.dim() == 2:
            enh_loss = F.cross_entropy(enh_pred, enh_gt)
            total_loss = total_loss + enh_loss
            loss_dict["enhancement"] = enh_loss.item()

    # Attenuation (CE, masked) — output.attenuation is (B, 6, 4)
    att_mask = exist_mask & gt_att_v.bool()
    if att_mask.sum() > 0:
        att_pred = output.attenuation[att_mask]  # (N, 4)
        att_gt = gt_att[att_mask].round().long().clamp(0, 3)
        if att_pred.shape[0] > 0 and att_pred.dim() == 2:
            att_loss = F.cross_entropy(att_pred, att_gt)
            total_loss = total_loss + att_loss
            loss_dict["attenuation"] = att_loss.item()

    # Count loss (for approaches A, C)
    if "left_count_logits" in extra:
        gt_left_count = labels["l3_left_count"].long().clamp(0, 3)
        gt_right_count = labels["l3_right_count"].long().clamp(0, 3)
        count_loss = (F.cross_entropy(extra["left_count_logits"], gt_left_count)
                      + F.cross_entropy(extra["right_count_logits"], gt_right_count)) / 2
        total_loss = total_loss + count_loss
        loss_dict["count"] = count_loss.item()

    # ---- Hierarchical supervision via noisy-OR ----
    # Aggregate slot-level predictions to side-level using noisy-OR:
    #   P(side) = 1 - prod(1 - sigmoid(slot_i))
    # exists logits: (B, 6), slots [0:3]=left, [3:6]=right
    exists_probs = torch.sigmoid(output.exists)  # (B, 6)

    # L1 supervision: side-level abnormality
    # noisy-OR over exists probs per side
    left_abn_prob = 1 - (1 - exists_probs[:, :3]).prod(dim=1)   # (B,)
    right_abn_prob = 1 - (1 - exists_probs[:, 3:]).prod(dim=1)  # (B,)
    gt_left_abn = labels["left_abnormal"]    # (B,)
    gt_right_abn = labels["right_abnormal"]  # (B,)
    # Clamp to avoid log(0)
    eps = 1e-7
    left_abn_prob = left_abn_prob.clamp(eps, 1 - eps)
    right_abn_prob = right_abn_prob.clamp(eps, 1 - eps)
    hier_l1_loss = -(gt_left_abn * torch.log(left_abn_prob) + (1 - gt_left_abn) * torch.log(1 - left_abn_prob)).mean() \
                 + -(gt_right_abn * torch.log(right_abn_prob) + (1 - gt_right_abn) * torch.log(1 - right_abn_prob)).mean()
    hier_l1_loss = hier_l1_loss / 2  # average over sides
    total_loss = total_loss + hier_weight * hier_l1_loss
    loss_dict["hier_l1"] = hier_l1_loss.item()

    # L2 supervision: side-level cyst/solid via noisy-OR (skip if hier_mode="l1")
    if hier_mode != "l1":
        for feat_name, feat_logits, gt_label_key in [
            ("cyst", output.cyst, "has_cyst"),
            ("solid", None, "has_solid"),  # solid = mass OR tumor
        ]:
            if feat_name == "solid":
                mass_probs = torch.sigmoid(output.mass)    # (B, 6)
                tumor_probs = torch.sigmoid(output.tumor)  # (B, 6)
                slot_solid_probs = 1 - (1 - mass_probs) * (1 - tumor_probs)  # (B, 6)
            else:
                slot_solid_probs = torch.sigmoid(feat_logits)  # (B, 6)

            left_prob = 1 - (1 - slot_solid_probs[:, :3]).prod(dim=1)
            right_prob = 1 - (1 - slot_solid_probs[:, 3:]).prod(dim=1)
            gt_left = labels[f"left_{gt_label_key}"]
            gt_right = labels[f"right_{gt_label_key}"]
            left_prob = left_prob.clamp(eps, 1 - eps)
            right_prob = right_prob.clamp(eps, 1 - eps)
            feat_loss = -(gt_left * torch.log(left_prob) + (1 - gt_left) * torch.log(1 - left_prob)).mean() \
                       + -(gt_right * torch.log(right_prob) + (1 - gt_right) * torch.log(1 - right_prob)).mean()
            feat_loss = feat_loss / 2
            total_loss = total_loss + hier_weight * feat_loss
            loss_dict[f"hier_{feat_name}"] = feat_loss.item()

    return total_loss, loss_dict
