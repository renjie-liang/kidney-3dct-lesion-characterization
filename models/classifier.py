"""
Classification heads for Label L1 and L2.
"""
import torch
import torch.nn as nn


class L1Head(nn.Module):
    """Label L1: left_abnormal, right_abnormal (2 binary outputs)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2) logits for [left_abnormal, right_abnormal]."""
        return self.head(features)


class L2Head(nn.Module):
    """Label L2: 4 binary + 2 regression outputs.

    Binary: left_has_cyst, right_has_cyst, left_has_solid, right_has_solid
    Regression: left_max_size, right_max_size
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim, 4)  # 4 binary
        self.reg_head = nn.Linear(hidden_dim, 2)  # 2 size regression

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_logits (B,4), size_pred (B,2))."""
        h = self.shared(features)
        cls_logits = self.cls_head(h)
        size_pred = self.reg_head(h)
        return cls_logits, size_pred


class KidneyClassifier(nn.Module):
    """Full model: encoder + classification head(s)."""

    def __init__(self, encoder: nn.Module, label_level: str = "L1",
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.label_level = label_level

        input_dim = encoder.output_dim

        if label_level == "L1":
            self.head = L1Head(input_dim, hidden_dim, dropout)
        elif label_level == "L2":
            self.head = L2Head(input_dim, hidden_dim, dropout)
        elif label_level.startswith("L3"):
            from models.l3_heads import L3HeadA, L3HeadB, L3HeadC, L3HeadD
            l3_approach = label_level.split("_")[-1]  # L3_A, L3_B, L3_C, L3_D
            heads = {"A": L3HeadA, "B": L3HeadB, "C": L3HeadC, "D": L3HeadD}
            self.head = heads[l3_approach](input_dim, hidden_dim=256)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass. For D2 mode, pass mask separately.

        For L3_B (DETR-like head) we request the encoder's spatial tokens so
        cross-attention has a non-trivial key/value set.
        """
        from models.mask_guided_pooling import MaskGuidedEncoder
        from models.encoder import SwinUNETREncoder
        from models.l3_heads import L3HeadB

        need_spatial = isinstance(self.head, L3HeadB) and isinstance(self.encoder, SwinUNETREncoder)
        if isinstance(self.encoder, MaskGuidedEncoder) and mask is not None:
            features = self.encoder(x, mask)
        elif need_spatial:
            features = self.encoder(x, return_spatial=True)
        else:
            features = self.encoder(x)
        return self.head(features)
