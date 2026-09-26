"""
Training engine for LIDC-IDRI Lung Nodule Segmentation models.

Features:
- AdamW optimizer with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on validation Dice score
- Checkpoint saving (best_model.pth and final_model.pth)
- Robust multi-metric evaluation per epoch
- Hardware-adaptive device selection (CUDA / MPS / CPU)
- Export of training logs and learning curves
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
from tqdm import tqdm

from src.config import cfg, get_device
from src.losses import get_loss_function
from src.metrics import compute_binary_metrics, MetricTracker


class Trainer:
    """
    Standardized trainer for segmentation models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str = "proposed_model",
        loss_name: str = "bce_dice",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        early_stopping_patience: int = 8,
        device: Optional[torch.device] = None,
        checkpoints_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        self.device = device or cfg.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.epochs = epochs
        self.patience = early_stopping_patience

        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else cfg.paths.checkpoints_dir
        self.results_dir = Path(results_dir) if results_dir else cfg.paths.results_dir
        self.metrics_dir = self.results_dir / "metrics"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = get_loss_function(
            name=loss_name,
            bce_weight=cfg.training.bce_weight,
            dice_weight=cfg.training.dice_weight
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": []
        }

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for batch in self.train_loader:
            images = batch["image"].to(self.device)  # (B, 1, H, W)
            masks = batch["mask"].to(self.device)    # (B, 1, H, W)

            self.optimizer.zero_grad()

            output = self.model(images)
            # Handle model returning (logits, attn_maps) tuple vs raw logits
            logits = output[0] if isinstance(output, tuple) else output

            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return float(running_loss / max(1, n_batches))

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        n_batches = len(self.val_loader)
        tracker = MetricTracker()

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            output = self.model(images)
            logits = output[0] if isinstance(output, tuple) else output

            loss = self.criterion(logits, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            tracker.update(probs, targets, threshold=0.5)

        avg_loss = float(running_loss / max(1, n_batches))
        summary = tracker.get_summary()

        metrics_flat = {
            "dice": summary["dice"]["mean"],
            "iou": summary["iou"]["mean"],
            "precision": summary["precision"]["mean"],
            "recall": summary["recall"]["mean"],
            "specificity": summary["specificity"]["mean"],
            "f1": summary["f1"]["mean"]
        }

        return avg_loss, metrics_flat

    def fit(self) -> Dict[str, Any]:
        print(f"\n==================================================")
        print(f"Starting Training: {self.model_name}")
        print(f"Device: {self.device} | Epochs: {self.epochs} | Criterion: {self.criterion.__class__.__name__}")
        print(f"==================================================")

        best_val_dice = -1.0
        patience_counter = 0
        best_checkpoint_path = self.checkpoints_dir / f"{self.model_name}_best.pth"
        final_checkpoint_path = self.checkpoints_dir / f"{self.model_name}_final.pth"

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["dice"])

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint best model
            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "val_metrics": val_metrics
                }, best_checkpoint_path)
                print(f"  --> Saved new best model checkpoint (Val Dice: {best_val_dice:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[Early Stopping] No improvement in validation Dice for {self.patience} epochs.")
                    break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds. Best Val Dice: {best_val_dice:.4f}")

        # Save final model
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "history": self.history
        }, final_checkpoint_path)

        # Save history log
        history_path = self.metrics_dir / f"training_history_{self.model_name}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "epochs_trained": epoch,
                "best_val_dice": best_val_dice,
                "total_time_seconds": total_time,
                "history": self.history
            }, f, indent=2)
        print(f"Saved training history to {history_path}")

        return self.history
