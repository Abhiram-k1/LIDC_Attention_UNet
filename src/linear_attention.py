"""
Linear Attention Module for Lung CT Feature Maps.

Reduces standard attention computational complexity from quadratic O(N^2)
to linear O(N) using kernel feature map factorization:
(phi(Q) * phi(K)^T) * V = phi(Q) * (phi(K)^T * V)

Features:
- Non-negative feature map phi(x) = ELU(x) + 1
- Multi-head support
- O(N * d^2) compute complexity
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    Linear / Efficient Attention module.
    
    Complexity comparison:
    Standard dense attention: O(N^2 * d)
    Linear attention: O(N * d^2) where N = H*W and d << N.
    """

    def __init__(
        self,
        in_channels: int,
        key_dim: int = 32,
        num_heads: int = 4,
        eps: float = 1e-6
    ):
        super().__init__()
        self.in_channels = in_channels
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.eps = eps

        total_dim = key_dim * num_heads
        self.to_q = nn.Conv2d(in_channels, total_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, total_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    @staticmethod
    def feature_map(x: torch.Tensor) -> torch.Tensor:
        """Non-negative kernel feature map phi(x) = elu(x) + 1."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            out: Linear-attention output (B, C, H, W)
            attn_weights: Summary attention response for visualization (B, 1, H, W)
        """
        b, c, h, w = x.shape
        n = h * w

        q = self.to_q(x)  # (B, H*D, H, W)
        k = self.to_k(x)  # (B, H*D, H, W)
        v = self.to_v(x)  # (B, C, H, W)

        # Reshape to (B, num_heads, N, key_dim)
        q = q.view(b, self.num_heads, self.key_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, D)
        k = k.view(b, self.num_heads, self.key_dim, n).permute(0, 1, 3, 2)  # (B, heads, N, D)
        # Reshape v to (B, num_heads, N, c_per_head)
        c_per_head = c // self.num_heads
        v = v.view(b, self.num_heads, c_per_head, n).permute(0, 1, 3, 2)    # (B, heads, N, c_per_head)

        # Apply kernel feature map phi(.)
        q_phi = self.feature_map(q)  # (B, heads, N, D)
        k_phi = self.feature_map(k)  # (B, heads, N, D)

        # Linear factorization: Compute (K^T * V) first: shape (B, heads, D, c_per_head)
        # O(N * D * c_per_head)
        kv = torch.matmul(k_phi.transpose(-2, -1), v)

        # Normalization denominator: Z = Q_phi * sum(K_phi, dim=N)
        k_sum = k_phi.sum(dim=-2, keepdim=True)  # (B, heads, 1, D)
        z = torch.matmul(q_phi, k_sum.transpose(-2, -1)) + self.eps  # (B, heads, N, 1)

        # Numerator: Q_phi * (K_phi^T * V): shape (B, heads, N, c_per_head)
        num = torch.matmul(q_phi, kv)  # (B, heads, N, c_per_head)

        out_heads = num / z  # (B, heads, N, c_per_head)

        # Reshape back to (B, C, H, W)
        out = out_heads.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj_out(out) + x  # Residual connection

        # Saliency summary from Q_phi * K_sum for visualization
        summary_attn = z.mean(dim=1).view(b, 1, h, w)
        summary_attn = (summary_attn - summary_attn.min()) / (summary_attn.max() - summary_attn.min() + self.eps)

        return out, summary_attn
