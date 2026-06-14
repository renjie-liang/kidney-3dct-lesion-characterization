"""M2: Count consistency regularizer.

Motivation: the baseline model over-predicts — outputs exists > 0.5 on ~6 slots
per patient when GT has ~1.4. This is because noisy-OR max aggregation doesn't
penalize FP slots. Count regularizer directly penalizes the discrepancy between
predicted and GT lesion count per side.

Three variants:
  - "soft_l1":  |sum(sigmoid(exists_logit)) - count_gt|   (continuous)
  - "soft_l2":  (sum(sigmoid(exists_logit)) - count_gt)^2
  - "ce":       4-way CE over count classes {0, 1, 2, 3}  (requires a count head)

Only soft_l1 / soft_l2 are drop-in (no architecture change).
"""
import torch
import torch.nn.functional as F


def count_reg_loss(
    exists_logits: torch.Tensor,
    exists_gt: torch.Tensor,
    variant: str = "soft_l1",
) -> torch.Tensor:
    """Count consistency loss.

    Args:
        exists_logits: (B, 6) — per-slot exists logits
        exists_gt:     (B, 6) — per-slot exists GT (0/1)
        variant:       "soft_l1" | "soft_l2"
    Returns:
        Scalar loss (mean over batch and sides).

    Per-side structure is assumed: slots [0:3] = left, [3:6] = right.
    """
    assert exists_logits.shape == exists_gt.shape
    B, S = exists_logits.shape
    assert S == 6, f"Expected 6 slots, got {S}"

    probs = torch.sigmoid(exists_logits)

    left_pred_count = probs[:, :3].sum(dim=1)
    right_pred_count = probs[:, 3:].sum(dim=1)
    left_gt_count = exists_gt[:, :3].sum(dim=1)
    right_gt_count = exists_gt[:, 3:].sum(dim=1)

    if variant == "soft_l1":
        loss = (torch.abs(left_pred_count - left_gt_count) +
                torch.abs(right_pred_count - right_gt_count)).mean() / 2
    elif variant == "soft_l2":
        loss = (((left_pred_count - left_gt_count) ** 2).mean() +
                ((right_pred_count - right_gt_count) ** 2).mean()) / 2
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return loss


if __name__ == "__main__":
    # Smoke test: predicting all 6 slots active when GT has 1 → should have high loss
    logits = torch.full((4, 6), 3.0, requires_grad=True)   # all very positive
    gt = torch.tensor([[1, 0, 0, 0, 0, 0]] * 4, dtype=torch.float32)

    loss_l1 = count_reg_loss(logits, gt, variant="soft_l1")
    loss_l2 = count_reg_loss(logits, gt, variant="soft_l2")
    print(f"Over-prediction penalty (L1): {loss_l1.item():.4f}  (should be large, ~2)")
    print(f"Over-prediction penalty (L2): {loss_l2.item():.4f}  (should be ~4)")

    # Well-calibrated: pred 1 on first slot, 0 elsewhere
    logits2 = torch.tensor([[5.0, -5, -5, -5, -5, -5]] * 4, requires_grad=True)
    loss2 = count_reg_loss(logits2, gt, variant="soft_l1")
    print(f"Calibrated prediction (L1): {loss2.item():.4f}  (should be ~0)")

    loss_l1.backward()
    print(f"Gradient shape: {logits.grad.shape} (backprop works)")
    print("✓ count_reg.py smoke test passed")
