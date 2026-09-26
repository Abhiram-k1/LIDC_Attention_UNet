"""
Cross-Feature Interaction Module for Multi-Scale CT Representations.

Harmonizes features from low-, mid-, and high-level encoder stages:
1. Spatially aligns feature maps to a common reference grid via adaptive pooling / bilinear sampling.
2. Projects diverse encoder channel depths into a unified latent dimension.
3. Performs cross-scale interactive gating to combine fine boundary details with deep contextual semantics.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFeatureInteraction(nn.Module):
    """
    Multi-level cross-feature interaction module.
    
    Accepts:
    - f_low: Low-level features (high resolution, e.g. H/2, W/2)
    - f_mid: Mid-level features (e.g. H/4, W/4)
    - f_high: High-level bottleneck features (e.g. H/8, W/8)
    
    Aligns spatial grids to reference resolution (H/8, W/8),
    projects all levels to unified latent dimension `fuse_dim`,
    computes inter-scale attention gating, and injects enriched context.
    """

    def __init__(
        self,
        in_channels_low: int,
        in_channels_mid: int,
        in_channels_high: int,
        fuse_dim: int = 64
    ):
        super().__init__()
        self.fuse_dim = fuse_dim

        # 1x1 Projections to unified channel dimension
        self.proj_low = nn.Sequential(
            nn.Conv2d(in_channels_low, fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_mid = nn.Sequential(
            nn.Conv2d(in_channels_mid, fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(in_channels_high, fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True)
        )

        # Fusion convolution over concatenated multi-level representations
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fuse_dim * 3, fuse_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True)
        )

        # Cross-scale attention gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fuse_dim, fuse_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_dim // 2, fuse_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Output projection back to high-level bottleneck dimension
        self.out_proj = nn.Sequential(
            nn.Conv2d(fuse_dim, in_channels_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels_high)
        )

    def forward(
        self,
        f_low: torch.Tensor,
        f_mid: torch.Tensor,
        f_high: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_low: (B, C_low, H_low, W_low)
            f_mid: (B, C_mid, H_mid, W_mid)
            f_high: (B, C_high, H_high, W_high)
        Returns:
            f_enhanced: Context-enriched high-level features (B, C_high, H_high, W_high)
            gate_weights: Cross-feature gating weights for visualization
        """
        target_size = f_high.shape[2:]  # (H_high, W_high)

        # Align spatial dimensions to reference bottleneck grid
        p_low = F.adaptive_avg_pool2d(self.proj_low(f_low), target_size)
        p_mid = F.adaptive_avg_pool2d(self.proj_mid(f_mid), target_size)
        p_high = self.proj_high(f_high)

        # Concatenate multi-level representations: (B, 3*fuse_dim, H_ref, W_ref)
        stacked = torch.cat([p_low, p_mid, p_high], dim=1)

        fused = self.fusion_conv(stacked)       # (B, fuse_dim, H_ref, W_ref)
        gate_w = self.gate(fused)               # Channel-attention gate
        gated_fused = fused * gate_w + fused    # Residual gated fusion

        # Project back to high-level channel dimension and add residual
        f_enhanced = self.out_proj(gated_fused) + f_high

        return f_enhanced, gate_w
