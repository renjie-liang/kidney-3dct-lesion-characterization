"""
D2: Mask-Guided Attention Pooling

Instead of adding mask as input channel (L3), uses mask at the OUTPUT layer
to guide spatial pooling. Encoder sees pure CT (1 channel), fully compatible
with pretrained weights.

Mask is downsampled to feature map resolution, then used to compute
region-specific embeddings (left kidney, right kidney, left lesion, right lesion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedPooling(nn.Module):
    """Pool encoder feature map using downsampled mask regions.

    Takes:
        feature_map: (B, C, D, H, W) from encoder's spatial output
        mask: (B, 1, D_orig, H_orig, W_orig) 7-class mask (uint8 values 0-6)

    Returns:
        pooled: (B, C * n_regions) concatenated region embeddings
    """

    # 7-class mask labels
    REGIONS = {
        "left_kidney": [1],       # label 1
        "right_kidney": [2],      # label 2
        "left_lesion": [3, 5],    # L-tumor + L-cyst
        "right_lesion": [4, 6],   # R-tumor + R-cyst
    }

    def __init__(self, feature_dim: int, use_global: bool = True):
        super().__init__()
        self.use_global = use_global
        n_regions = len(self.REGIONS) + (1 if use_global else 0)  # 4 regions + global
        self.output_dim = feature_dim * n_regions

    def forward(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C = feature_map.shape[:2]
        spatial = feature_map.shape[2:]  # (D, H, W)

        # Downsample mask to feature map resolution
        # mask is (B, 1, D, H, W) float with integer values 0-6
        mask_down = F.interpolate(
            mask.float(), size=spatial, mode="nearest"
        )  # (B, 1, d, h, w)
        mask_down = mask_down[:, 0]  # (B, d, h, w)

        region_embeddings = []

        for region_name, labels in self.REGIONS.items():
            # Create binary mask for this region
            region_mask = torch.zeros_like(mask_down, dtype=torch.bool)
            for lbl in labels:
                region_mask = region_mask | (mask_down == lbl)

            # Masked average pooling
            region_mask_expanded = region_mask.unsqueeze(1).float()  # (B, 1, d, h, w)
            masked_features = feature_map * region_mask_expanded  # (B, C, d, h, w)

            # Sum and normalize
            region_sum = masked_features.sum(dim=(2, 3, 4))  # (B, C)
            region_count = region_mask_expanded.sum(dim=(2, 3, 4)).clamp(min=1)  # (B, 1)
            region_emb = region_sum / region_count  # (B, C)

            region_embeddings.append(region_emb)

        if self.use_global:
            # Global average pooling as additional context
            global_emb = feature_map.mean(dim=(2, 3, 4))  # (B, C)
            region_embeddings.append(global_emb)

        return torch.cat(region_embeddings, dim=1)  # (B, C * n_regions)


class MaskGuidedEncoder(nn.Module):
    """SwinUNETR encoder with mask-guided pooling at output.

    Input: CT only (1 channel) — fully compatible with pretrained weights.
    Mask is used only for pooling, not as input.
    """

    def __init__(self, base_encoder, feature_dim: int = 768):
        super().__init__()
        self.base_encoder = base_encoder  # SwinUNETREncoder (returns spatial features)
        self.pooling = MaskGuidedPooling(feature_dim, use_global=True)
        self.output_dim = self.pooling.output_dim  # 768 * 5 = 3840

    def forward(self, ct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct: (B, 1, D, H, W) — pure CT input
            mask: (B, 1, D, H, W) — 7-class mask (NOT input to encoder)
        Returns:
            features: (B, output_dim) — region-pooled features
        """
        # Get spatial feature map (before GAP)
        hidden_states = self.base_encoder.swin_unetr.swinViT(ct)
        feature_map = hidden_states[-1]  # (B, 768, d, h, w)

        # Mask-guided pooling
        return self.pooling(feature_map, mask)
