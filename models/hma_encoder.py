"""
Hierarchical Mask Attention (HMA) Encoder.

Injects segmentation mask as spatial attention at each SwinUNETR encoder stage,
rather than concatenating it as an input channel.

Two variants:
  - HMA-Binary: single gate from binary kidney mask (1 channel)
  - HMA-Decomposed: three independent gates from decomposed mask
    (kidney / tumor / cyst), allowing different trust levels per component

Key design: residual gating  h * (1 + sigmoid(gate(mask_down)))
  - When gate output = 0: features unchanged (mask ignored)
  - When gate output > 0: mask-attended regions amplified
  - Mask can only help, never hurt (residual connection)

CT input stays 1-channel, fully compatible with SuPreM pretrained weights.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import SwinUNETREncoder

logger = logging.getLogger(__name__)

STAGE_DIMS = [48, 96, 192, 384, 768]


class MaskGate(nn.Module):
    """Learns spatial attention from downsampled mask at one encoder stage.

    Input:  mask (B, C_mask, D', H', W')  — already interpolated to stage resolution
    Output: attention (B, C_feat, D', H', W')  — broadcast-multiplied with features
    """

    def __init__(self, mask_channels: int, feat_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(mask_channels, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(feat_channels),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(mask))


class HMAEncoder(nn.Module):
    """Hierarchical Mask Attention encoder wrapping SwinUNETR.

    Args:
        in_channels: CT input channels (always 1 for SuPreM compatibility)
        feature_size: SwinUNETR feature_size (48)
        pretrained: SuPreM / VoCo / from_scratch
        hma_mode: "binary" (1-channel mask) or "decomposed" (3-channel: kidney/tumor/cyst)
        gate_stages: which stages to apply mask gating (default: all 5)
    """

    def __init__(self, in_channels: int = 1, feature_size: int = 48,
                 pretrained: str = "suprem", hma_mode: str = "binary",
                 gate_stages: list[int] | None = None):
        super().__init__()

        self.base_encoder = SwinUNETREncoder(
            in_channels=in_channels, feature_size=feature_size, pretrained=pretrained,
        )

        self.hma_mode = hma_mode
        self.gate_stages = gate_stages if gate_stages is not None else list(range(5))

        mask_channels = 1 if hma_mode == "binary" else 3

        self.mask_gates = nn.ModuleDict()
        for stage_idx in self.gate_stages:
            self.mask_gates[str(stage_idx)] = MaskGate(
                mask_channels=mask_channels,
                feat_channels=STAGE_DIMS[stage_idx],
            )

        self.output_dim = self.base_encoder.output_dim  # 768
        self.pool = nn.AdaptiveAvgPool3d(1)

        logger.info(f"HMA-{hma_mode}: gates at stages {self.gate_stages}, "
                    f"mask_channels={mask_channels}")

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert integer mask (B, D, H, W) or (B, 1, D, H, W) to gate input.

        For binary mode: (B, 1, D, H, W) kidney mask
        For decomposed mode: (B, 3, D, H, W) — kidney, tumor, cyst channels
        """
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)  # (B, 1, D, H, W)

        if self.hma_mode == "binary":
            # Any non-zero label → kidney region
            return (mask > 0).float()

        # Decomposed: 3 channels
        kidney = ((mask == 1) | (mask == 2)).float()    # kidney parenchyma
        tumor = ((mask == 3) | (mask == 4)).float()      # tumor (L+R)
        cyst = ((mask == 5) | (mask == 6)).float()        # cyst (L+R)
        return torch.cat([kidney, tumor, cyst], dim=1)    # (B, 3, D, H, W)

    def forward(self, ct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct: (B, 1, D, H, W) — CT volume (1-channel)
            mask: (B, D, H, W) or (B, 1, D, H, W) — segmentation mask (integer labels 0-6)

        Returns:
            (B, 768) feature vector
        """
        mask_input = self._prepare_mask(mask)

        # Get hidden states from SwinUNETR backbone
        hidden_states = self.base_encoder.swin_unetr.swinViT(ct)

        # Apply mask gating at selected stages
        for stage_idx in self.gate_stages:
            hs = hidden_states[stage_idx]
            mask_down = F.interpolate(
                mask_input, size=hs.shape[2:], mode="nearest",
            )
            attention = self.mask_gates[str(stage_idx)](mask_down)
            hidden_states[stage_idx] = hs * (1.0 + attention)

        # Pool from last stage
        return self.pool(hidden_states[-1]).flatten(1)
