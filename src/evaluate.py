"""
Independent evaluation module on the isolated test set.

Calculates:
- Dice Similarity Coefficient
- Intersection over Union (IoU)
- Precision, Recall, Specificity, F1
- 95% Confidence Intervals
- Parameter counts (total & trainable)
- Average inference latency per slice (ms)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg, get_device
from src.metrics import MetricTracker


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate_model_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    device: Optional[torch.device] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a trained model checkpoint strictly on the test set.
    """
    device = device or cfg.device
    model = model.to(device)
    model.eval()

    tracker = MetricTracker()
    latencies = []

    print(f"\n[Evaluation] Evaluating '{model_name}' on test set ({len(test_loader.dataset)} samples)...")

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        start = time.perf_counter()
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000.0 / images.size(0)  # ms per slice
        latencies.append(latency)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = masks.cpu().numpy()
        tracker.update(probs, targets, threshold=0.5)

    summary = tracker.get_summary()
    total_params, trainable_params = count_parameters(model)
    avg_latency = float(np.mean(latencies))

    results = {
        "model_name": model_name,
        "test_samples": len(test_loader.dataset),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "avg_latency_ms": avg_latency,
        "metrics": {
            "dice": summary["dice"],
            "iou": summary["iou"],
            "precision": summary["precision"],
            "recall": summary["recall"],
            "specificity": summary["specificity"],
            "f1": summary["f1"]
        }
    }

    print(f"\n================ Test Results for {model_name} ================")
    print(f"  Dice Score:   {summary['dice']['mean']:.4f} ± {summary['dice']['ci_95']:.4f}")
    print(f"  IoU Score:    {summary['iou']['mean']:.4f} ± {summary['iou']['ci_95']:.4f}")
    print(f"  Precision:    {summary['precision']['mean']:.4f}")
    print(f"  Recall/Sens:  {summary['recall']['mean']:.4f}")
    print(f"  Specificity:  {summary['specificity']['mean']:.4f}")
    print(f"  F1 Score:     {summary['f1']['mean']:.4f}")
    print(f"  Parameters:   {total_params:,} (Trainable: {trainable_params:,})")
    print(f"  Latency:      {avg_latency:.2f} ms/slice")
    print("================================================================")

    if save_results:
        save_path = cfg.paths.metrics_dir / f"test_evaluation_{model_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluation] Saved test results to {save_path}")

    return results
