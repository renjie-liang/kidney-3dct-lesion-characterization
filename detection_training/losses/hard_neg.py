"""M7: Hard negative mining.

Motivation: most negative slots (empty slots) are trivial to classify. To focus
the model on the "hardest" negatives (empty slots that the model wrongly thinks
have lesions — i.e., false positives), we explicitly identify and upweight them.

Strategy:
  1. Compute per-slot BCE for all slots.
  2. Among negatives (targets = 0), find the top-K hardest (highest BCE = model
     most confident they are lesions).
  3. Apply extra weight to those K slots.

Typical K = 3 per sample (out of 6 slots), extra weight = 2.0.
"""
import torch
import torch.nn.functional as F


def hard_negative_mining_loss(
    logits: torch.Tensor,       # (B, 6)
    targets: torch.Tensor,       # (B, 6) binary
    top_k: int = 3,
    extra_weight: float = 1.0,
) -> torch.Tensor:
    """BCE + hard-negative mining surcharge.

    Args:
        logits, targets: (B, 6)
        top_k: how many hardest negatives per sample to upweight
        extra_weight: additional weight applied to hard negatives (on top of 1.0 base).
                      Total weight for hard negatives = 1 + extra_weight.
    Returns:
        Scalar loss.
    """
    assert logits.shape == targets.shape
    B, S = logits.shape
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Identify negatives (target=0)
    neg_mask = (targets < 0.5)

    # For each sample, find top_k hardest negatives (highest BCE)
    neg_bce = bce.clone()
    neg_bce[~neg_mask] = -1.0   # exclude positives from topk
    # Take top_k per row (if fewer than top_k negatives, just take all)
    k_effective = min(top_k, S)
    topk_vals, topk_idx = neg_bce.topk(k_effective, dim=1)

    # Build hard-negative mask
    hard_neg_mask = torch.zeros_like(bce)
    hard_neg_mask.scatter_(1, topk_idx, 1.0)
    # Only count "truly hard" (must be actual negatives and selected)
    hard_neg_mask = hard_neg_mask * neg_mask.float()

    # Final per-element weight
    weight = 1.0 + extra_weight * hard_neg_mask
    return (bce * weight).mean()


if __name__ == "__main__":
    B = 4
    logits = torch.randn(B, 6, requires_grad=True)
    targets = torch.tensor([[1, 0, 0, 0, 0, 0]] * B, dtype=torch.float32)

    plain = F.binary_cross_entropy_with_logits(logits, targets)
    hnm = hard_negative_mining_loss(logits, targets, top_k=3, extra_weight=1.0)
    print(f"Plain BCE:              {plain.item():.4f}")
    print(f"Hard-neg-mined (K=3):   {hnm.item():.4f}")

    hnm.backward()
    print("✓ hard_neg.py smoke test passed")
