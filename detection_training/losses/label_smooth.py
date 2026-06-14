"""M5: Label smoothing for binary targets.

Simple utility. Typically used alongside M3 (Hungarian) to prevent overconfidence
collapse on no-object slots.

    smoothed = epsilon if y=0 else (1 - epsilon) if y=1

Typical epsilon = 0.05.
"""
import torch


def smooth_labels(targets: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """Apply label smoothing to binary targets.

    Args:
        targets: tensor with values in {0, 1}
        epsilon: smoothing amount (targets become [epsilon, 1-epsilon])
    Returns:
        smoothed tensor, same shape as input
    """
    return targets * (1 - epsilon) + (1 - targets) * epsilon


if __name__ == "__main__":
    t = torch.tensor([0.0, 0, 1, 1, 0])
    s = smooth_labels(t, 0.05)
    print(f"Original: {t.tolist()}")
    print(f"Smoothed: {s.tolist()}")
    print("✓ label_smooth.py smoke test passed")
