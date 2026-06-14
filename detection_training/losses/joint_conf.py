"""M4: Joint confidence supervision.

Motivation: per-class AP evaluation uses confidence = P(exists) * P(type=c)
as the class-c score. If exists and type are trained independently (as in the
baseline BCE), their product is poorly calibrated — the model has no incentive
to make the joint score meaningful.

This loss directly supervises the joint score:

    L_joint = sum_c BCE(sigmoid(exists) * sigmoid(type_c), exists_gt * type_c_gt * valid_c)

Only computed where type_c_valid = 1. Typically added as an auxiliary term with
a small weight (e.g. 0.1-0.3).
"""
import torch
import torch.nn.functional as F


def joint_conf_loss(
    exists_logits: torch.Tensor,              # (B, 6)
    type_logits: dict,                         # {'cyst','solid'} → (B, 6)
    exists_gt: torch.Tensor,                   # (B, 6)
    type_gt: dict,                             # {'cyst','solid'} → (B, 6)
    type_valid: dict,                          # {'cyst','solid'} → (B, 6)
    class_weights: dict = None,
) -> torch.Tensor:
    """Joint P(exists) * P(type) supervision.

    For each class c, compute BCE between:
      pred:   sigmoid(exists_logit) * sigmoid(type_c_logit)     ∈ [0, 1]
      target: exists_gt * type_c_gt                             ∈ {0, 1}

    Only sum over valid entries (type_c_valid > 0.5).
    """
    if class_weights is None:
        class_weights = {"cyst": 1.0, "solid": 1.0}

    exists_p = torch.sigmoid(exists_logits)
    total_loss = torch.tensor(0.0, device=exists_logits.device)
    n_terms = 0

    for feat, w in class_weights.items():
        type_p = torch.sigmoid(type_logits[feat])
        joint_p = exists_p * type_p
        joint_gt = exists_gt * type_gt[feat]
        mask = type_valid[feat] > 0.5
        if mask.any():
            # BCE on joint probabilities (not logits). Use F.binary_cross_entropy.
            joint_p_clamped = joint_p.clamp(1e-6, 1 - 1e-6)
            bce = -(joint_gt * torch.log(joint_p_clamped) +
                    (1 - joint_gt) * torch.log(1 - joint_p_clamped))
            total_loss = total_loss + w * bce[mask].mean()
            n_terms += 1

    return total_loss / max(n_terms, 1)


if __name__ == "__main__":
    B = 4
    exists_logits = torch.randn(B, 6, requires_grad=True)
    type_logits = {k: torch.randn(B, 6, requires_grad=True) for k in ["cyst", "solid"]}
    exists_gt = torch.randint(0, 2, (B, 6)).float()
    type_gt = {k: torch.randint(0, 2, (B, 6)).float() for k in ["cyst", "solid"]}
    type_valid = {k: torch.ones(B, 6) for k in ["cyst", "solid"]}

    loss = joint_conf_loss(exists_logits, type_logits, exists_gt, type_gt, type_valid)
    print(f"Joint conf loss: {loss.item():.4f}")
    loss.backward()
    print("✓ joint_conf.py smoke test passed")
