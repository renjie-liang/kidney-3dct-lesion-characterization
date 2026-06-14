"""
CTViT encoder wrapper for kidney CT classification.

Uses the CTViT visual encoder from CT-CLIP, with pretrained weights
from CT-RATE (21K+ chest CT volumes, CLIP-style vision-language pretraining).

Input: (B, C, D, H, W) where H=W=480, D=241 (or other valid depth)
       C=1 for L2, C=2 for L3 (mask channel)
Output: (B, 512) feature vector

Weight source: set CT_CLIP_WEIGHT, or place weights at WEIGHTS_ROOT/CT-CLIP/CT-CLIP_v2.pt
"""
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Add CT-CLIP repo to path for importing CTViT and dependencies
CT_CLIP_REPO = Path(os.environ.get("CT_CLIP_REPO", "third_party/CT-CLIP"))
CTVIT_MODULE = CT_CLIP_REPO / "transformer_maskgit"

WEIGHT_PATH = Path(os.environ.get("CT_CLIP_WEIGHT", str(Path(os.environ.get("WEIGHTS_ROOT", "weights")) / "CT-CLIP" / "CT-CLIP_v2.pt")))

# CTViT pretrained config (inferred from checkpoint)
CTVIT_CONFIG = {
    "dim": 512,
    "codebook_size": 8192,
    "image_size": 480,
    "patch_size": 20,
    "temporal_patch_size": 10,
    "spatial_depth": 4,
    "temporal_depth": 4,
    "dim_head": 64,
    "heads": 8,
    "channels": 1,
    "use_vgg_and_gan": False,
}


class CTViTEncoder(nn.Module):
    """CTViT encoder for classification.

    Loads the visual encoder from CT-CLIP, strips decoder/VQ,
    adds global average pooling for feature extraction.

    Output: (B, 512)
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()

        # Import CTViT from CT-CLIP repo
        if str(CTVIT_MODULE) not in sys.path:
            sys.path.insert(0, str(CTVIT_MODULE))

        from transformer_maskgit.ctvit import CTViT

        # Build CTViT with pretrained config
        config = CTVIT_CONFIG.copy()
        self.ctvit = CTViT(**config)

        if pretrained:
            self._load_pretrained()

        # Adapt for 2-channel input (L3 mask)
        if in_channels == 2:
            self._adapt_input_channels(in_channels)

        self.output_dim = config["dim"]  # 512
        self.pool = nn.AdaptiveAvgPool3d(1)

    def _load_pretrained(self):
        """Load visual encoder weights from CT-CLIP checkpoint."""
        if not WEIGHT_PATH.exists():
            raise FileNotFoundError(f"CT-CLIP weights not found: {WEIGHT_PATH}")

        state_dict = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=False)

        # Extract visual_transformer.* keys → strip prefix
        ctvit_sd = {}
        for k, v in state_dict.items():
            if k.startswith("visual_transformer."):
                new_k = k.replace("visual_transformer.", "")
                ctvit_sd[new_k] = v

        # Filter by shape match (skip decoder/pixels layers)
        model_sd = self.ctvit.state_dict()
        filtered = {}
        for k, v in ctvit_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered[k] = v

        missing, unexpected = self.ctvit.load_state_dict(filtered, strict=False)
        encoder_keys = [k for k in filtered if "enc_" in k or "to_patch" in k or "spatial_rel" in k]
        logger.info(f"Loaded CTViT: {len(filtered)}/{len(ctvit_sd)} keys matched "
                    f"({len(encoder_keys)} encoder keys), "
                    f"{len(missing)} missing, {len(unexpected)} unexpected")

    def _adapt_input_channels(self, in_channels: int):
        """Adapt patch embedding for 2-channel input (CT + mask)."""
        # First frame patch embedding
        old_linear = self.ctvit.to_patch_emb_first_frame[2]  # Linear(400, 512)
        old_ln = self.ctvit.to_patch_emb_first_frame[1]  # LayerNorm(400)
        ph, pw = self.ctvit.patch_size
        new_input_dim = in_channels * ph * pw  # 2 * 20 * 20 = 800

        new_first = nn.Sequential(
            self.ctvit.to_patch_emb_first_frame[0].__class__(
                **{"pattern": f'b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)',
                   "p1": ph, "p2": pw}
            ) if hasattr(self.ctvit.to_patch_emb_first_frame[0], 'pattern') else
            self.ctvit.to_patch_emb_first_frame[0],
            nn.LayerNorm(new_input_dim),
            nn.Linear(new_input_dim, self.ctvit.to_patch_emb_first_frame[2].out_features),
            nn.LayerNorm(self.ctvit.to_patch_emb_first_frame[2].out_features),
        )

        # Copy weights for first channel, zero-init second
        with torch.no_grad():
            new_first[2].weight[:, :old_linear.in_features] = old_linear.weight
            new_first[2].weight[:, old_linear.in_features:] = 0
            new_first[2].bias.copy_(old_linear.bias)

        # Rest frames patch embedding
        old_linear_rest = self.ctvit.to_patch_emb[2]
        pt = self.ctvit.temporal_patch_size
        new_input_dim_rest = in_channels * pt * ph * pw  # 2 * 10 * 20 * 20 = 8000

        new_rest = nn.Sequential(
            self.ctvit.to_patch_emb[0],  # Rearrange (will handle 2 channels automatically)
            nn.LayerNorm(new_input_dim_rest),
            nn.Linear(new_input_dim_rest, old_linear_rest.out_features),
            nn.LayerNorm(old_linear_rest.out_features),
        )

        with torch.no_grad():
            new_rest[2].weight[:, :old_linear_rest.in_features] = old_linear_rest.weight
            new_rest[2].weight[:, old_linear_rest.in_features:] = 0
            new_rest[2].bias.copy_(old_linear_rest.bias)

        self.ctvit.to_patch_emb_first_frame = new_first
        self.ctvit.to_patch_emb = new_rest
        logger.info(f"Adapted CTViT input channels: 1 -> {in_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) — C=1 or 2, any spatial size (will be padded to 480×480×240)
        Returns:
            features: (B, 512)
        """
        # Pad to CTViT's original target size (480, 480, 240) to match pretrained pos bias
        target_D, target_H, target_W = 240, 480, 480
        B, C, D, H, W = x.shape

        # Center-pad to target size (crop if larger, pad if smaller)
        # D dimension
        if D > target_D:
            start = (D - target_D) // 2
            x = x[:, :, start:start + target_D, :, :]
        elif D < target_D:
            pad_before = (target_D - D) // 2
            pad_after = target_D - D - pad_before
            x = nn.functional.pad(x, (0, 0, 0, 0, pad_before, pad_after))

        # H dimension
        if H > target_H:
            start = (H - target_H) // 2
            x = x[:, :, :, start:start + target_H, :]
        elif H < target_H:
            pad_before = (target_H - H) // 2
            pad_after = target_H - H - pad_before
            x = nn.functional.pad(x, (0, 0, pad_before, pad_after, 0, 0))

        # W dimension
        if W > target_W:
            start = (W - target_W) // 2
            x = x[:, :, :, :, start:start + target_W]
        elif W < target_W:
            pad_before = (target_W - W) // 2
            pad_after = target_W - W - pad_before
            x = nn.functional.pad(x, (pad_before, pad_after, 0, 0, 0, 0))

        # Patch embedding on full volume
        tokens = self.ctvit.to_patch_emb(x)  # (B, T, h, w, dim)

        # Encode (spatial + temporal transformers)
        encoded = self.ctvit.encode(tokens)  # (B, T, h, w, dim)

        # Global average pooling → (B, dim)
        B, T, h, w, dim = encoded.shape
        encoded = encoded.permute(0, 4, 1, 2, 3)  # (B, dim, T, h, w)
        pooled = self.pool(encoded).flatten(1)  # (B, dim)

        return pooled
