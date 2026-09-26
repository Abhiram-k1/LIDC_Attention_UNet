"""
Ablation Study Module for LIDC-IDRI Nodule Segmentation.

Implements and evaluates the six ablation variants:
Model A: Standard U-Net
Model B: U-Net + Spatial Attention
Model C: U-Net + Linear Attention
Model D: U-Net + Sparse Attention
Model E: U-Net + Cross-Feature Interaction
Model F: Full Proposed Cross-Feature Spatial Sparse Linear Attention U-Net

Compiles a scientifically rigorous comparison table based strictly on measured empirical runs.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg
from src.unet import StandardUNet, DoubleConv, DownBlock, UpBlock, OutConv
from src.spatial_attention import SpatialAttention
from src.linear_attention import LinearAttention
from src.sparse_attention import SparseSpatialAttention
from src.cross_feature import CrossFeatureInteraction
from src.proposed_model import CrossFeatureSpatialSparseLinearAttentionUNet, AttentionDecoderBlock
from src.train import Trainer
from src.evaluate import evaluate_model_on_test_set


# --- Ablation Variant Definitions ---

class UNetSpatialAttention(nn.Module):
    """Model B: U-Net with Spatial Attention at the bottleneck and decoder skips."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, bilinear: bool = True):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        factor = 2 if bilinear else 1
        self.bottleneck = DownBlock(b * 8, (b * 16) // factor)

        self.spatial_attn = SpatialAttention(kernel_size=7, use_residual=True)

        self.up1 = AttentionDecoderBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = AttentionDecoderBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = AttentionDecoderBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = AttentionDecoderBlock(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x5_attn, s_map = self.spatial_attn(x5)

        d1, _ = self.up1(x5_attn, x4)
        d2, _ = self.up2(d1, x3)
        d3, _ = self.up3(d2, x2)
        d4, _ = self.up4(d3, x1)
        logits = self.outc(d4)
        return logits, {"spatial_map": s_map}


class UNetLinearAttention(nn.Module):
    """Model C: U-Net with Linear Attention at the bottleneck."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, bilinear: bool = True):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        factor = 2 if bilinear else 1
        bottleneck_dim = (b * 16) // factor
        self.bottleneck = DownBlock(b * 8, bottleneck_dim)

        self.linear_attn = LinearAttention(in_channels=bottleneck_dim, key_dim=32, num_heads=4)

        self.up1 = UpBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = UpBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = UpBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = UpBlock(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x5_attn, lin_map = self.linear_attn(x5)

        d1 = self.up1(x5_attn, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        logits = self.outc(d4)
        return logits, {"linear_map": lin_map}


class UNetSparseAttention(nn.Module):
    """Model D: U-Net with Sparse Top-k Spatial Attention at the bottleneck."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, k: int = 32, bilinear: bool = True):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        factor = 2 if bilinear else 1
        bottleneck_dim = (b * 16) // factor
        self.bottleneck = DownBlock(b * 8, bottleneck_dim)

        self.sparse_attn = SparseSpatialAttention(in_channels=bottleneck_dim, k=k, num_heads=4)

        self.up1 = UpBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = UpBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = UpBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = UpBlock(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x5_attn, sparse_map = self.sparse_attn(x5)

        d1 = self.up1(x5_attn, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        logits = self.outc(d4)
        return logits, {"sparse_map": sparse_map}


class UNetCrossFeature(nn.Module):
    """Model E: U-Net with Cross-Feature Interaction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, bilinear: bool = True):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        factor = 2 if bilinear else 1
        bottleneck_dim = (b * 16) // factor
        self.bottleneck = DownBlock(b * 8, bottleneck_dim)

        self.cross_feature = CrossFeatureInteraction(
            in_channels_low=b * 2,
            in_channels_mid=b * 4,
            in_channels_high=bottleneck_dim,
            fuse_dim=b * 2
        )

        self.up1 = UpBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = UpBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = UpBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = UpBlock(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x5_cf, cf_gate = self.cross_feature(f_low=x2, f_mid=x3, f_high=x5)

        d1 = self.up1(x5_cf, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        logits = self.outc(d4)
        return logits, {"cf_gate": cf_gate}


def run_ablation_study(
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 15
) -> pd.DataFrame:
    """
    Executes empirical training and evaluation of all 6 ablation models.
    """
    models = {
        "Model A (Standard U-Net)": StandardUNet(base_channels=cfg.model.base_channels),
        "Model B (U-Net + Spatial Attn)": UNetSpatialAttention(base_channels=cfg.model.base_channels),
        "Model C (U-Net + Linear Attn)": UNetLinearAttention(base_channels=cfg.model.base_channels),
        "Model D (U-Net + Sparse Attn)": UNetSparseAttention(base_channels=cfg.model.base_channels, k=cfg.model.sparse_top_k),
        "Model E (U-Net + Cross-Feature)": UNetCrossFeature(base_channels=cfg.model.base_channels),
        "Model F (Proposed Model)": CrossFeatureSpatialSparseLinearAttentionUNet(
            base_channels=cfg.model.base_channels,
            sparse_k=cfg.model.sparse_top_k
        )
    }

    records = []

    for name, model in models.items():
        print(f"\n==================================================")
        print(f"Executing Ablation: {name}")
        print(f"==================================================")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=name.replace(" ", "_").replace("(", "").replace(")", ""),
            epochs=epochs,
            early_stopping_patience=6
        )
        trainer.fit()

        test_res = evaluate_model_on_test_set(
            model=model,
            test_loader=test_loader,
            model_name=name
        )

        m = test_res["metrics"]
        records.append({
            "Model": name,
            "Dice": f"{m['dice']['mean']:.4f} ± {m['dice']['ci_95']:.4f}",
            "IoU": f"{m['iou']['mean']:.4f} ± {m['iou']['ci_95']:.4f}",
            "Precision": f"{m['precision']['mean']:.4f}",
            "Recall": f"{m['recall']['mean']:.4f}",
            "Specificity": f"{m['specificity']['mean']:.4f}",
            "F1": f"{m['f1']['mean']:.4f}",
            "Parameters": f"{test_res['total_parameters']:,}",
            "Latency (ms)": f"{test_res['avg_latency_ms']:.2f}"
        })

    df = pd.DataFrame(records)
    csv_path = cfg.paths.results_dir / "ablation_study_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Ablation] Saved complete ablation table to {csv_path}")
    return df
