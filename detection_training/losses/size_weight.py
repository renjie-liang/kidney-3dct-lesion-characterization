"""M6: Size-aware BCE weighting.

Motivation: larger lesions are more clinically important (more likely to be
tumors / malignant). Weight per-slot BCE by a function of GT size so the loss
focuses more on correctly classifying large lesions.

    weight_i = 1 + alpha * size_gt_i   (if exists_gt_i = 1)
    weight_i = 1                         (if exists_gt_i = 0)

Default alpha = 0.3 (so a 3cm lesion gets weight 1.9, a 6cm lesion gets 2.8).
"""
import torch
import torch.nn.functional as F


def size_weighted_bce(
    logits: torch.Tensor,           # (B, 6)
    targets: torch.Tensor,           # (B, 6) binary labels
    gt_size: torch.Tensor,           # (B, 6) size in cm
    alpha: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE with size-based per-sample weighting.

    Larger GT lesions receive higher loss weight.
    """
    assert logits.shape == targets.shape == gt_size.shape
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weight = 1.0 + alpha * gt_size * targets   # only lift positive examples
    weighted = bce * weight
    if reduction == "mean":
        return weighted.mean()
    if reduction == "sum":
        return weighted.sum()
    return weighted


if __name__ == "__main__":
    B = 4
    logits = torch.randn(B, 6, requires_grad=True)
    targets = torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32)
    gt_size = torch.tensor([[5.0, 1.0, 0, 0, 0, 0]] * B)    # first lesion is 5cm

    plain = F.binary_cross_entropy_with_logits(logits, targets)
    weighted = size_weighted_bce(logits, targets, gt_size, alpha=0.3)
    print(f"Plain BCE:           {plain.item():.4f}")
    print(f"Size-weighted BCE:   {weighted.item():.4f}  (should be higher, weights >= 1)")

    weighted.backward()
    print("✓ size_weight.py smoke test passed")
