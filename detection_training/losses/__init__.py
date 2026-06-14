"""Detection-oriented loss modules for per-lesion training.

Each loss is a pure function with clear inputs/outputs. See PROPOSAL.md for
method-to-module mapping (M1-M7).
"""
from .focal import focal_bce_loss
from .count_reg import count_reg_loss
from .hungarian import hungarian_l3_loss, hungarian_match
from .joint_conf import joint_conf_loss
from .label_smooth import smooth_labels
from .size_weight import size_weighted_bce
from .hard_neg import hard_negative_mining_loss

__all__ = [
    "focal_bce_loss",
    "count_reg_loss",
    "hungarian_l3_loss",
    "hungarian_match",
    "joint_conf_loss",
    "smooth_labels",
    "size_weighted_bce",
    "hard_negative_mining_loss",
]
