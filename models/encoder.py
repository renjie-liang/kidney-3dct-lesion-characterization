"""
Encoder wrappers for kidney CT classification.

Supports:
  - SwinUNETR (SuPreM, VoCo, from_scratch)
  - 3D ViT (from_scratch, inflated_2d)
  - ResNet-18 3D (baseline)
"""
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)

_WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights"))
WEIGHT_PATHS = {
    "suprem": str(_WEIGHTS_ROOT / "SuPreM" / "supervised_suprem_swinunetr_2100.pth"),
    "voco": str(_WEIGHTS_ROOT / "VoCo" / "VoComni_B.pt"),
}


def _load_checkpoint(path: str) -> dict:
    """Load checkpoint and extract state_dict, handling different formats."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "net" in ckpt:
        sd = ckpt["net"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        k = k.replace("module.backbone.", "")
        k = k.replace("module.", "")
        cleaned[k] = v

    return cleaned


# ─────────────────────────────────────────────
# SwinUNETR Encoder
# ─────────────────────────────────────────────

class SwinUNETREncoder(nn.Module):
    """SwinUNETR encoder. Output: (batch, 768) for feature_size=48."""

    def __init__(self, in_channels: int = 1, feature_size: int = 48,
                 pretrained: str = "from_scratch", use_checkpoint: bool = False):
        super().__init__()

        self.swin_unetr = SwinUNETR(
            in_channels=1, out_channels=14, feature_size=feature_size,
            use_checkpoint=use_checkpoint, spatial_dims=3,
        )

        if pretrained != "from_scratch":
            self._load_pretrained(pretrained)

        if in_channels == 2:
            self._adapt_input_channels(in_channels)

        self.output_dim = feature_size * 16  # 768
        self.pool = nn.AdaptiveAvgPool3d(1)

    def _load_pretrained(self, name: str):
        weight_path = WEIGHT_PATHS[name]
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"Pretrained weights not found: {weight_path}")

        sd = _load_checkpoint(weight_path)
        model_sd = self.swin_unetr.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape}

        missing, unexpected = self.swin_unetr.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded {name}: {len(filtered)}/{len(sd)} keys matched, "
                    f"{len(missing)} missing, {len(unexpected)} unexpected")

    def _adapt_input_channels(self, in_channels: int):
        old_conv = self.swin_unetr.swinViT.patch_embed.proj
        new_conv = nn.Conv3d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :1] = old_conv.weight
            new_conv.weight[:, 1:] = 0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.swin_unetr.swinViT.patch_embed.proj = new_conv
        logger.info(f"Adapted input channels: 1 -> {in_channels}")

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        """Encode input CT volume.

        return_spatial=False: returns (B, 768) globally-pooled features.
        return_spatial=True:  returns (B, 768) pooled AND (B, N, 768) spatial tokens
                              where N = D' * H' * W' at the deepest Swin stage.
        """
        hidden_states = self.swin_unetr.swinViT(x)
        last = hidden_states[-1]  # (B, C, D', H', W')
        pooled = self.pool(last).flatten(1)  # (B, C)
        if return_spatial:
            B, C = last.shape[0], last.shape[1]
            tokens = last.view(B, C, -1).transpose(1, 2)  # (B, N, C)
            return pooled, tokens
        return pooled


# ─────────────────────────────────────────────
# 3D ViT Encoder
# ─────────────────────────────────────────────

class ViT3DEncoder(nn.Module):
    """3D Vision Transformer encoder.

    Modes:
      - from_scratch: random init
      - inflated_2d: inflate ImageNet ViT-B/16 weights to 3D

    Output: (batch, hidden_size).
    """

    def __init__(self, in_channels: int = 1, pretrained: str = "from_scratch",
                 img_size: tuple = (320, 192, 224),
                 patch_size: tuple = (16, 16, 16),
                 hidden_size: int = 768, num_layers: int = 12, num_heads: int = 12):
        super().__init__()

        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

        # Compute number of patches
        grid = tuple(s // p for s, p in zip(img_size, patch_size))
        num_patches = grid[0] * grid[1] * grid[2]
        self.grid_size = grid

        # CLS token + position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

        self.output_dim = hidden_size

        if pretrained == "inflated_2d":
            self._load_inflated_2d(in_channels)

    def _load_inflated_2d(self, in_channels: int):
        """Inflate ImageNet pretrained ViT-B/16 weights to 3D."""
        try:
            import timm
            model_2d = timm.create_model("vit_base_patch16_224", pretrained=True)
        except Exception as e:
            logger.warning(f"Could not load 2D ViT weights: {e}")
            return

        sd_2d = model_2d.state_dict()
        loaded = 0

        # 1. Inflate patch_embed: (768, 3, 16, 16) → (768, in_ch, 16, 16, 16)
        w_2d = sd_2d["patch_embed.proj.weight"]  # (768, 3, 16, 16)
        w_3d = w_2d.mean(dim=1, keepdim=True)  # (768, 1, 16, 16) — average RGB
        w_3d = w_3d.unsqueeze(2).repeat(1, 1, 16, 1, 1)  # (768, 1, 16, 16, 16)
        w_3d = w_3d / 16  # normalize for depth
        if in_channels == 2:
            w_3d = torch.cat([w_3d, torch.zeros_like(w_3d)], dim=1)
        self.patch_embed.weight.data.copy_(w_3d)

        if "patch_embed.proj.bias" in sd_2d:
            self.patch_embed.bias.data.copy_(sd_2d["patch_embed.proj.bias"])
        loaded += 1

        # 2. CLS token
        if "cls_token" in sd_2d:
            self.cls_token.data.copy_(sd_2d["cls_token"])
            loaded += 1

        # 3. Transformer blocks
        for i in range(min(12, len(self.transformer.layers))):
            prefix_2d = f"blocks.{i}"
            prefix_3d_layer = self.transformer.layers[i]

            mapping = {
                f"{prefix_2d}.norm1.weight": "norm1.weight",
                f"{prefix_2d}.norm1.bias": "norm1.bias",
                f"{prefix_2d}.attn.qkv.weight": "self_attn.in_proj_weight",
                f"{prefix_2d}.attn.qkv.bias": "self_attn.in_proj_bias",
                f"{prefix_2d}.attn.proj.weight": "self_attn.out_proj.weight",
                f"{prefix_2d}.attn.proj.bias": "self_attn.out_proj.bias",
                f"{prefix_2d}.norm2.weight": "norm2.weight",
                f"{prefix_2d}.norm2.bias": "norm2.bias",
                f"{prefix_2d}.mlp.fc1.weight": "linear1.weight",
                f"{prefix_2d}.mlp.fc1.bias": "linear1.bias",
                f"{prefix_2d}.mlp.fc2.weight": "linear2.weight",
                f"{prefix_2d}.mlp.fc2.bias": "linear2.bias",
            }

            layer_sd = prefix_3d_layer.state_dict()
            for key_2d, key_3d in mapping.items():
                if key_2d in sd_2d and key_3d in layer_sd:
                    if sd_2d[key_2d].shape == layer_sd[key_3d].shape:
                        layer_sd[key_3d] = sd_2d[key_2d]
                        loaded += 1

            prefix_3d_layer.load_state_dict(layer_sd, strict=False)

        # 4. Final norm
        if "norm.weight" in sd_2d:
            self.norm.weight.data.copy_(sd_2d["norm.weight"])
            self.norm.bias.data.copy_(sd_2d["norm.bias"])
            loaded += 1

        logger.info(f"Inflated 2D ViT-B/16 weights: {loaded} params transferred")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (B, C, D, H, W)
        Returns: (B, hidden_size)
        """
        # Patch embedding: (B, C, D, H, W) → (B, hidden, gD, gH, gW)
        x = self.patch_embed(x)
        # Flatten spatial: (B, hidden, N) → (B, N, hidden)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token output
        return x[:, 0]


# ─────────────────────────────────────────────
# ResNet-18 3D (baseline)
# ─────────────────────────────────────────────

class ResNet3DEncoder(nn.Module):
    """3D ResNet-18. Output: (batch, 512)."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        from monai.networks.nets import resnet18

        self.backbone = resnet18(
            spatial_dims=3, n_input_channels=in_channels, num_classes=1,
        )
        self.output_dim = 512
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
