"""M1: Focal loss for binary classification.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).

Motivation: BCE weights easy and hard examples equally. In our setting, most
slots are "empty" (exists_gt = 0), producing easy negative gradients. Focal
loss down-weights easy examples so training focuses on hard ones (including
hard negatives — the FPs we want to suppress).

    FL(p, y) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t = p if y=1 else (1-p), and alpha_t = alpha if y=1 else (1-alpha).

Typical values:
  gamma = 2.0, alpha = 0.25 (DETR/RetinaNet default)
  gamma = 1.5 for milder downweighting
"""
import torch
import torch.nn.functional as F


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss on logits.

    Args:
        logits: (...,) raw logits (pre-sigmoid)
        targets: (...,) binary labels {0, 1} (same shape as logits)
        gamma: focusing parameter. 0 → BCE, higher → more focus on hard examples.
        alpha: class-balance weight for positives. 1.0 disables alpha weighting.
        reduction: "mean" | "sum" | "none"
    Returns:
        Scalar loss (if reduction != "none") or tensor of per-element losses.
    """
    assert logits.shape == targets.shape, f"{logits.shape} != {targets.shape}"
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)      # p if y=1 else (1-p)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


if __name__ == "__main__":
    # Smoke test
    logits = torch.randn(10, 6, requires_grad=True)
    targets = torch.randint(0, 2, (10, 6)).float()

    loss = focal_bce_loss(logits, targets, gamma=2.0, alpha=0.25)
    print(f"Focal loss: {loss.item():.4f}")

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Plain BCE:  {bce.item():.4f}")

    loss.backward()
    print(f"Gradient shape: {logits.grad.shape} (backprop works)")
    print("✓ focal.py smoke test passed")
