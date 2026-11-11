"""
Evaluation metrics module for binary lung nodule segmentation.

Calculates:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
- Precision (Positive Predictive Value)
- Recall (Sensitivity)
- Specificity (True Negative Rate)
- F1 Score

Includes safe handling for empty masks (no nodule present) and
computes 95% confidence intervals across test cases.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> Dict[str, Any]:
    """
    Computes standard segmentation metrics for a single 2D or 3D binary pair.
    Explicitly distinguishes nodule-positive slices from empty control slices
    to prevent empty-slice false 1.0 Dice inflation.
    
    Args:
        pred: float array of predicted probabilities in [0, 1] or binary {0, 1}
        target: ground truth binary array in {0, 1}
        threshold: binarization threshold for pred
        eps: smoothing epsilon
    """
    pred_bin = (pred >= threshold).astype(np.uint8).flatten()
    target_bin = (target > 0).astype(np.uint8).flatten()

    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))
    tn = np.sum((pred_bin == 0) & (target_bin == 0))

    target_empty = bool(np.sum(target_bin) == 0)
    pred_empty = bool(np.sum(pred_bin) == 0)

    # Specificity is well-defined across all slices with background pixels
    specificity = 1.0 if fp == 0 else float(tn / (tn + fp + eps))

    # Case A: Empty Ground Truth Slice (no nodule present)
    if target_empty:
        # On empty slices, Dice/IoU are not defined for nodule segmentation.
        # Track false-positive rate and empty-slice specificity separately.
        return {
            "is_nodule": False,
            "target_empty": True,
            "pred_empty": pred_empty,
            "dice": None,
            "iou": None,
            "precision": 1.0 if pred_empty else 0.0,
            "recall": 1.0 if pred_empty else 0.0,
            "specificity": specificity,
            "f1": None,
            "empty_slice_fp": not pred_empty,
            "empty_slice_fp_pixels": int(fp),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    # Case B: Nodule-Positive Ground Truth Slice (target has >= 1 nodule pixel)
    if pred_empty:
        # Complete miss on a positive nodule slice
        return {
            "is_nodule": True,
            "target_empty": False,
            "pred_empty": True,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": specificity,
            "f1": 0.0,
            "empty_slice_fp": None,
            "empty_slice_fp_pixels": 0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    # Case C: Standard Positive Case with predictions
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
    iou = float(tp / (tp + fp + fn + eps))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float((2.0 * precision * recall) / (precision + recall + eps))

    return {
        "is_nodule": True,
        "target_empty": False,
        "pred_empty": False,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "empty_slice_fp": None,
        "empty_slice_fp_pixels": 0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


class MetricTracker:
    """
    Accumulates evaluation metrics across test batches, reporting:
    1. Nodule-positive slice segmentation metrics (Dice, IoU, Precision, Recall, Specificity, F1)
    2. Empty-slice false positive performance (FPR, Specificity, false positive pixel count)
    Prevents artificial score inflation by never assigning Dice 1.0 to negative slices.
    """

    def __init__(self):
        self.nodule_metrics: Dict[str, List[float]] = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "f1": []
        }
        self.empty_slice_fp: List[bool] = []
        self.empty_slice_fp_pixels: List[int] = []
        self.empty_slice_specificity: List[float] = []
        self.total_slices = 0

    def update(self, pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
        """Processes a single batch or sample."""
        if pred.ndim == 4:
            # Batch of shape (B, 1, H, W)
            for b in range(pred.shape[0]):
                p_slice = pred[b, 0]
                t_slice = target[b, 0]
                self._record_sample(compute_binary_metrics(p_slice, t_slice, threshold=threshold))
        else:
            self._record_sample(compute_binary_metrics(pred, target, threshold=threshold))

    def _record_sample(self, res: Dict[str, Any]):
        self.total_slices += 1
        if res["is_nodule"]:
            for k in self.nodule_metrics.keys():
                val = res.get(k)
                if val is not None:
                    self.nodule_metrics[k].append(float(val))
        else:
            self.empty_slice_fp.append(res["empty_slice_fp"])
            self.empty_slice_fp_pixels.append(res["empty_slice_fp_pixels"])
            self.empty_slice_specificity.append(res["specificity"])

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        n_nodule = len(self.nodule_metrics["dice"])
        n_empty = len(self.empty_slice_fp)

        # Nodule-positive slice metrics
        for k, vals in self.nodule_metrics.items():
            if not vals:
                summary[k] = {"mean": 0.0, "std": 0.0, "ci_95": 0.0}
                continue

            arr = np.array(vals, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if n_nodule > 1 else 0.0
            ci_95 = float(1.96 * (std_val / np.sqrt(n_nodule))) if n_nodule > 1 else 0.0

            summary[k] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": ci_95
            }

        # Empty-slice false positive analysis
        empty_fpr = float(np.mean(self.empty_slice_fp)) if n_empty > 0 else 0.0
        empty_spec = float(np.mean(self.empty_slice_specificity)) if n_empty > 0 else 1.0
        empty_fp_px = float(np.mean(self.empty_slice_fp_pixels)) if n_empty > 0 else 0.0

        summary["empty_slice_analysis"] = {
            "total_empty_slices": n_empty,
            "false_positive_slices": int(sum(self.empty_slice_fp)),
            "false_positive_rate": empty_fpr,
            "empty_slice_specificity": empty_spec,
            "avg_fp_pixels_per_empty_slice": empty_fp_px
        }

        summary["slice_counts"] = {
            "total_slices": self.total_slices,
            "nodule_positive_slices": n_nodule,
            "empty_control_slices": n_empty
        }

        return summary
