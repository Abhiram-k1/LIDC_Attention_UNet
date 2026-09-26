"""
Cross-Feature Spatial Sparse Linear Attention U-Net.

A research-grade PyTorch architecture integrating:
1. Multi-scale Cross-Feature Interaction (harmonizing low, mid, and high-level encoder cues)
2. Spatial Attention (localizing nodule contours and suppressing background air)
3. Sparse Top-k Attention Routing (filtering out redundant background parenchyma)
4. Linear Attention (factorizing attention matrix for O(N) complexity)
5. Symmetrical attention-enhanced decoder pathways
"""

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.unet import DoubleConv, DownBlock, OutConv
from src.spatial_attention import SpatialAttention
from src.linear_attention import LinearAttention
from src.sparse_attention import SparseSpatialAttention
from src.cross_feature import CrossFeatureInteraction


class SpatialSparseLinearAttentionBlock(nn.Module):
    """
    Combined Attention Module:
    Input -> Spatial Attention -> Sparse Top-k Selection -> Linear Attention Core -> Residual Output
    """

    def __init__(self, in_channels: int, k: int = 32, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # 1. Spatial Attention mechanism
        self.spatial_attn = SpatialAttention(kernel_size=7, use_residual=True)

        # 2. Sparse routing gate
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 3. Linear Q, K, V projections
        self.to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 4. Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.eps = 1e-6

    @staticmethod
    def feature_map(x: torch.Tensor) -> torch.Tensor:
        """Non-negative kernel feature map phi(x) = elu(x) + 1."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Enhanced feature map (B, C, H, W)
            attn_dict: Dictionary containing extracted spatial and sparse attention maps
        """
        b, c, h, w = x.shape
        n = h * w
        effective_k = min(self.k, n)

        # Step 1: Spatial Attention modulation
        x_spatial, spatial_map = self.spatial_attn(x)  # (B, C, H, W)

        # Step 2: Sparse Token Selection via routing scores
        router_scores = self.router(x_spatial)         # (B, 1, H, W)
        scores_flat = router_scores.view(b, n)          # (B, N)
        _, topk_indices = torch.topk(scores_flat, k=effective_k, dim=-1)  # (B, k)

        # Step 3: Project Q, K, V
        q = self.to_q(x_spatial).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, D)
        k = self.to_k(x_spatial).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, D)
        v = self.to_v(x_spatial).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, D)

        # Gather Top-k sparse Keys and Values
        topk_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        k_sparse = torch.gather(k, dim=2, index=topk_expanded)  # (B, heads, k, D)
        v_sparse = torch.gather(v, dim=2, index=topk_expanded)  # (B, heads, k, D)

        # Step 4: Linear kernel feature map factorization
        q_phi = self.feature_map(q)          # (B, heads, N, D)
        k_phi = self.feature_map(k_sparse)   # (B, heads, k, D)

        # Compute K_phi^T * V_sparse first: shape (B, heads, D, D)
        kv = torch.matmul(k_phi.transpose(-2, -1), v_sparse)  # (B, heads, D, D)

        # Normalization factor Z = Q_phi * sum(K_phi)
        k_sum = k_phi.sum(dim=-2, keepdim=True)               # (B, heads, 1, D)
        z = torch.matmul(q_phi, k_sum.transpose(-2, -1)) + self.eps  # (B, heads, N, 1)

        # Numerator: Q_phi * (K_phi^T * V_sparse)
        num = torch.matmul(q_phi, kv)                         # (B, heads, N, D)
        out_heads = num / z                                   # (B, heads, N, D)

        # Step 5: Reshape and residual projection
        out = out_heads.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.out_proj(out) + x  # Residual connection

        attn_dict = {
            "spatial_map": spatial_map,
            "sparse_router": router_scores
        }

        return out, attn_dict


class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with spatial attention applied to the incoming skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Spatial attention filter for encoder skip features
        skip_channels = in_channels // 2
        self.skip_attn = SpatialAttention(kernel_size=7, use_residual=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x1: from lower decoder stage
        # x2: skip connection from symmetrical encoder
        x1 = self.up(x1)

        # Pad if odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Filter skip connection
        x2_filtered, attn_map = self.skip_attn(x2)
        x = torch.cat([x2_filtered, x1], dim=1)
        out = self.conv(x)
        return out, attn_map


class CrossFeatureSpatialSparseLinearAttentionUNet(nn.Module):
    """
    Proposed Architecture:
    Cross-Feature Spatial Sparse Linear Attention U-Net
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        sparse_k: int = 32,
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        b = base_channels

        # Encoder stages
        self.inc = DoubleConv(in_channels, b)             # Level 1: (B, b, H, W)
        self.down1 = DownBlock(b, b * 2)                  # Level 2: (B, 2b, H/2, W/2)
        self.down2 = DownBlock(b * 2, b * 4)              # Level 3: (B, 4b, H/4, W/4)
        self.down3 = DownBlock(b * 4, b * 8)              # Level 4: (B, 8b, H/8, W/8)

        factor = 2 if bilinear else 1
        bottleneck_channels = (b * 16) // factor
        self.bottleneck_conv = DownBlock(b * 8, bottleneck_channels)  # (B, 16b//factor, H/16, W/16)

        # Multi-scale Cross-Feature Interaction
        self.cross_feature = CrossFeatureInteraction(
            in_channels_low=b * 2,
            in_channels_mid=b * 4,
            in_channels_high=bottleneck_channels,
            fuse_dim=b * 2
        )

        # Spatial Sparse Linear Attention Block at Bottleneck
        self.ssla_bottleneck = SpatialSparseLinearAttentionBlock(
            in_channels=bottleneck_channels,
            k=sparse_k,
            num_heads=4
        )

        # Attention-Enhanced Decoder stages
        self.up1 = AttentionDecoderBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = AttentionDecoderBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = AttentionDecoderBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = AttentionDecoderBlock(b * 2, b, bilinear)

        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning raw logits and all attention visualization maps.
        """
        # Encoder
        x1 = self.inc(x)        # (B, b, H, W)
        x2 = self.down1(x1)     # (B, 2b, H/2, W/2)
        x3 = self.down2(x2)     # (B, 4b, H/4, W/4)
        x4 = self.down3(x3)     # (B, 8b, H/8, W/8)
        x5 = self.bottleneck_conv(x4)  # (B, 16b//factor, H/16, W/16)

        # 1. Multi-scale Cross-Feature Interaction
        x5_cf, cf_gate = self.cross_feature(f_low=x2, f_mid=x3, f_high=x5)

        # 2. Spatial Sparse Linear Attention
        x5_ssla, ssla_maps = self.ssla_bottleneck(x5_cf)

        # 3. Decoder with attention-filtered skip connections
        d1, map1 = self.up1(x5_ssla, x4)
        d2, map2 = self.up2(d1, x3)
        d3, map3 = self.up3(d2, x2)
        d4, map4 = self.up4(d3, x1)

        logits = self.outc(d4)

        attn_maps = {
            "cf_gate": cf_gate,
            "spatial_bottleneck": ssla_maps["spatial_map"],
            "sparse_router": ssla_maps["sparse_router"],
            "decoder_attn1": map1,
            "decoder_attn2": map2,
            "decoder_attn3": map3,
            "decoder_attn4": map4
        }

        return logits, attn_maps
