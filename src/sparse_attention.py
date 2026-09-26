"""
Sparse Attention Module for Lung CT Feature Maps.

Implements Top-k Spatial Sparse Routing:
In lung CT images, nodules are compact focal lesions surrounded by homogeneous parenchyma.
Sparse attention selects the top-k most informative spatial tokens, restricting attention
computation to relevant anatomical structures.

Complexity:
Standard Attention: O(N^2 * d)
Top-k Sparse Attention: O(N * k * d) where k << N.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSpatialAttention(nn.Module):
    """
    Top-k Spatial Sparse Attention module.
    
    1. Projects input feature map to Q, K, V.
    2. Uses a lightweight spatial routing gate to identify the top-k most informative keys/values.
    3. Computes attention strictly between queries and the top-k sparse key tokens.
    """

    def __init__(
        self,
        in_channels: int,
        k: int = 32,
        num_heads: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Saliency routing scorer to rank top-k key positions
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature tensor of shape (B, C, H, W)
        Returns:
            out: Sparse-attention modulated feature map (B, C, H, W)
            saliency_map: Sparse routing weights for visualization (B, 1, H, W)
        """
        b, c, h, w = x.shape
        n = h * w
        effective_k = min(self.k, n)

        # Compute routing saliency scores
        saliency_map = self.router(x)  # (B, 1, H, W)
        scores_flat = saliency_map.view(b, n)  # (B, N)

        # Select top-k spatial locations across the entire feature map
        _, topk_indices = torch.topk(scores_flat, k=effective_k, dim=-1)  # (B, k)

        # Linear projections
        q = self.q_proj(x).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        k = self.k_proj(x).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        v = self.v_proj(x).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # Gather top-k keys and values: shape (B, heads, k, head_dim)
        topk_idx_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        k_sparse = torch.gather(k, dim=2, index=topk_idx_expanded)  # (B, heads, k, head_dim)
        v_sparse = torch.gather(v, dim=2, index=topk_idx_expanded)  # (B, heads, k, head_dim)

        # Compute Sparse Attention: Q (N x head_dim) * K_sparse^T (head_dim x k) -> (N x k)
        # O(B * heads * N * k * head_dim)
        attn_scores = torch.matmul(q, k_sparse.transpose(-2, -1)) * self.scale  # (B, heads, N, k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Context aggregation: Attn (N x k) * V_sparse (k x head_dim) -> (N x head_dim)
        out_heads = torch.matmul(attn_weights, v_sparse)  # (B, heads, N, head_dim)

        # Reshape back to (B, C, H, W)
        out = out_heads.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.out_proj(out) + x  # Residual connection

        return out, saliency_map
