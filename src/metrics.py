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
) -> Dict[str, float]:
    """
    Computes all standard segmentation metrics for a single 2D or 3D binary pair.
    
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

    target_empty = (np.sum(target_bin) == 0)
    pred_empty = (np.sum(pred_bin) == 0)

    # Edge-case: Both ground truth and prediction are completely empty (clean negative case)
    if target_empty and pred_empty:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "specificity": 1.0,
            "f1": 1.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    # Edge-case: Target empty, but false positive predicted
    if target_empty and not pred_empty:
        spec = float(tn / (tn + fp + eps))
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 1.0,  # No positive cases to miss
            "specificity": spec,
            "f1": 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    # Edge-case: Target has nodule, but prediction completely missed it
    if not target_empty and pred_empty:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 1.0,
            "f1": 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }

    # Standard positive case
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
    iou = float(tp / (tp + fp + fn + eps))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    specificity = float(tn / (tn + fp + eps))
    f1 = float((2.0 * precision * recall) / (precision + recall + eps))

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


class MetricTracker:
    """
    Accumulates per-sample evaluation metrics across batches
    and computes mean, standard deviation, and 95% confidence intervals.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "specificity": [],
            "f1": []
        }

    def update(self, pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
        """Processes a single batch or sample."""
        if pred.ndim == 4:
            # Batch of shape (B, 1, H, W)
            for b in range(pred.shape[0]):
                p_slice = pred[b, 0]
                t_slice = target[b, 0]
                res = compute_binary_metrics(p_slice, t_slice, threshold=threshold)
                for k in self.metrics.keys():
                    self.metrics[k].append(res[k])
        else:
            res = compute_binary_metrics(pred, target, threshold=threshold)
            for k in self.metrics.keys():
                self.metrics[k].append(res[k])

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        n = len(self.metrics["dice"])
        for k, vals in self.metrics.items():
            if not vals:
                summary[k] = {"mean": 0.0, "std": 0.0, "ci_95": 0.0}
                continue

            arr = np.array(vals, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            # 95% Confidence Interval half-width: 1.96 * (std / sqrt(n))
            ci_95 = float(1.96 * (std_val / np.sqrt(n))) if n > 1 else 0.0

            summary[k] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": ci_95
            }
        return summary
