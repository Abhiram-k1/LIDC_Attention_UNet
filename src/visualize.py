"""
Visualization module for qualitative results, error analysis, and attention maps.

Generates:
1. 5-Column comparative figures: [CT Slice | Ground Truth | Baseline U-Net | Proposed Model | Overlay]
2. Categorized error analysis cases: True Positives (successful), False Positives, False Negatives.
3. Attention map visualization: [CT Slice | Attention Heatmap | Overlay | Ground Truth | Prediction]
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg


class ResultVisualizer:
    """
    Renders research figures for qualitative comparison and attention analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else cfg.paths.results_dir
        self.plots_dir = self.output_dir / "plots"
        self.attn_dir = self.output_dir / "attention_maps"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.attn_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def plot_comparative_predictions(
        self,
        baseline_model: nn.Module,
        proposed_model: nn.Module,
        test_loader: DataLoader,
        num_examples: int = 5,
        device: Optional[torch.device] = None,
        save_name: str = "comparative_segmentation_results.png"
    ) -> Path:
        """
        Creates 5-column visualization:
        Col 1: CT Slice
        Col 2: Ground Truth Nodule Mask
        Col 3: Baseline U-Net Prediction
        Col 4: Proposed Model Prediction
        Col 5: Comparative Color-coded Overlay
        """
        device = device or cfg.device
        baseline_model = baseline_model.to(device).eval()
        proposed_model = proposed_model.to(device).eval()

        collected = []
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            metas = batch["meta"]

            out_b = baseline_model(images)
            out_p = proposed_model(images)

            logits_b = out_b[0] if isinstance(out_b, tuple) else out_b
            logits_p = out_p[0] if isinstance(out_p, tuple) else out_p

            probs_b = torch.sigmoid(logits_b).cpu().numpy()
            probs_p = torch.sigmoid(logits_p).cpu().numpy()
            imgs_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(images.size(0)):
                # Collect both positive and negative slices reproducibly
                collected.append({
                    "image": imgs_np[i, 0],
                    "gt": masks_np[i, 0],
                    "pred_base": (probs_b[i, 0] >= 0.5).astype(np.uint8),
                    "pred_prop": (probs_p[i, 0] >= 0.5).astype(np.uint8),
                    "pid": metas["patient_id"][i],
                    "slice_idx": metas["slice_index"][i],
                    "is_pos": metas["is_positive"][i]
                })
                if len(collected) >= num_examples * 2:
                    break
            if len(collected) >= num_examples * 2:
                break

        # Select samples: prioritize positive slices with nodules, plus negative slices
        positives = [c for c in collected if c["is_pos"] == 1]
        negatives = [c for c in collected if c["is_pos"] == 0]
        selected = positives[:num_examples] if len(positives) >= num_examples else (positives + negatives[:num_examples - len(positives)])
        selected = selected[:num_examples]

        n_rows = len(selected)
        fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        cols = [
            "Input CT Slice",
            "Ground Truth",
            "Baseline U-Net",
            "Proposed SSLA U-Net",
            "Comparative Overlay"
        ]

        for col_idx, col_name in enumerate(cols):
            axes[0, col_idx].set_title(col_name, fontsize=12, fontweight="bold", pad=8)

        for row_idx, item in enumerate(selected):
            img = item["image"]
            gt = item["gt"]
            pb = item["pred_base"]
            pp = item["pred_prop"]

            # Col 1: CT
            axes[row_idx, 0].imshow(img, cmap="gray")
            axes[row_idx, 0].set_ylabel(f"Pt: {item['pid']}\nSlice: {item['slice_idx']}", fontsize=10)

            # Col 2: GT Mask
            axes[row_idx, 1].imshow(img, cmap="gray")
            if np.any(gt > 0):
                axes[row_idx, 1].contour(gt, colors="lime", linewidths=1.5)

            # Col 3: Baseline Pred
            axes[row_idx, 2].imshow(img, cmap="gray")
            if np.any(pb > 0):
                axes[row_idx, 2].contour(pb, colors="cyan", linewidths=1.5)

            # Col 4: Proposed Pred
            axes[row_idx, 3].imshow(img, cmap="gray")
            if np.any(pp > 0):
                axes[row_idx, 3].contour(pp, colors="yellow", linewidths=1.5)

            # Col 5: Color-coded Overlay
            # Green = GT, Yellow = Proposed, Cyan = Baseline
            axes[row_idx, 5-1].imshow(img, cmap="gray")
            if np.any(gt > 0):
                axes[row_idx, 4].contour(gt, colors="lime", linewidths=1.5)
            if np.any(pp > 0):
                axes[row_idx, 4].contour(pp, colors="yellow", linewidths=1.5, linestyles="dashed")

            for col in range(5):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved comparative predictions figure to {save_path}")
        return save_path

    @torch.no_grad()
    def plot_attention_maps(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        save_name: str = "attention_saliency_maps.png"
    ) -> Path:
        """
        Visualizes learned attention mechanisms:
        [CT Slice | Spatial Attention Map | Sparse Router Weights | GT Mask | Final Prediction]
        """
        device = device or cfg.device
        model = model.to(device).eval()

        target_sample = None
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            metas = batch["meta"]

            out = model(images)
            if isinstance(out, tuple) and len(out) == 2:
                logits, attn_maps = out
                for i in range(images.size(0)):
                    if metas["is_positive"][i] == 1:
                        target_sample = {
                            "image": images[i, 0].cpu().numpy(),
                            "mask": masks[i, 0].cpu().numpy(),
                            "pred": (torch.sigmoid(logits[i, 0]).cpu().numpy() >= 0.5).astype(np.uint8),
                            "spatial_map": attn_maps.get("spatial_bottleneck", None),
                            "sparse_router": attn_maps.get("sparse_router", None),
                            "pid": metas["patient_id"][i],
                            "slice_idx": metas["slice_index"][i]
                        }
                        break
            if target_sample:
                break

        if not target_sample:
            print("[Visualizer] No suitable sample with attention maps found for attention plot.")
            return self.attn_dir

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        img = target_sample["image"]
        mask = target_sample["mask"]
        pred = target_sample["pred"]

        # 1. CT
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input CT Slice", fontsize=11, fontweight="bold")

        # 2. Spatial Attention Map
        s_map = target_sample["spatial_map"]
        if s_map is not None:
            s_np = s_map[0, 0].cpu().numpy()
            axes[1].imshow(s_np, cmap="hot")
            axes[1].set_title("Spatial Saliency Map", fontsize=11, fontweight="bold")
        else:
            axes[1].imshow(img, cmap="gray")

        # 3. Sparse Router Saliency
        sp_map = target_sample["sparse_router"]
        if sp_map is not None:
            sp_np = sp_map[0, 0].cpu().numpy()
            axes[2].imshow(sp_np, cmap="viridis")
            axes[2].set_title("Sparse Routing Weights", fontsize=11, fontweight="bold")
        else:
            axes[2].imshow(img, cmap="gray")

        # 4. Ground Truth
        axes[3].imshow(img, cmap="gray")
        if np.any(mask > 0):
            axes[3].contour(mask, colors="lime", linewidths=1.5)
        axes[3].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")

        # 5. Prediction
        axes[4].imshow(img, cmap="gray")
        if np.any(pred > 0):
            axes[4].contour(pred, colors="yellow", linewidths=1.5)
        axes[4].set_title("Predicted Mask", fontsize=11, fontweight="bold")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        save_path = self.attn_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Saved attention map visualization to {save_path}")
        return save_path
