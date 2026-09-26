"""
Spatial Attention Module for Lung CT Nodule Segmentation.

Extracts spatial saliency weights across feature maps by pooling
inter-channel statistics and applying spatial convolutions to emphasize
nodular boundaries and suppress non-informative lung parenchyma.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Combines channel-wise average and max pooling followed by a spatial convolution.
    
    Formula:
        M_s(F) = Sigmoid(Conv7x7([AvgPool(F); MaxPool(F)]))
        Output = F * M_s(F) + F  (Residual connection)
    """

    def __init__(self, kernel_size: int = 7, use_residual: bool = True):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.use_residual = use_residual

        # 2 input channels: 1 from AvgPool, 1 from MaxPool
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature map of shape (B, C, H, W)
        Returns:
            out: Attention-modulated feature map (B, C, H, W)
            attn_map: Extracted spatial attention weights (B, 1, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B, 1, H, W)
        pool_cat = torch.cat([avg_out, max_out], dim=1)    # (B, 2, H, W)

        attn_map = self.conv(pool_cat)                     # (B, 1, H, W)

        if self.use_residual:
            out = x * attn_map + x
        else:
            out = x * attn_map

        return out, attn_map
