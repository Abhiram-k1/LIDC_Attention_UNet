"""
Loss functions module for lung nodule segmentation.

Implements:
1. Binary Cross Entropy (BCEWithLogits)
2. Soft Dice Loss
3. BCE + Dice Combined Loss
4. Focal Loss (for foreground class imbalance)
5. Tversky Loss (asymmetric penalty for false negatives)
6. Focal Tversky Loss
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Continuous Soft Dice Loss with Laplace smoothing.
    Expects raw logits as model predictions.
    """

    def __init__(self, smooth: float = 1e-6, p: int = 2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1, H, W)
        # targets: (B, 1, H, W) in {0, 1}
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        if self.p == 1:
            cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        else:
            cardinality = (probs_flat ** 2).sum(dim=1) + (targets_flat ** 2).sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_score
        return loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy and Soft Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.bce_weight * bce) + (self.dice_weight * dice)


class FocalLoss(nn.Module):
    """
    Focal Loss to counter extreme foreground/background class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss adds alpha and beta weights to trade off False Positives vs False Negatives.
    For small lung nodules, beta > alpha (e.g. alpha=0.3, beta=0.7) heavily penalizes missing a nodule.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + (self.alpha * fp) + (self.beta * fn) + self.smooth)
        return (1.0 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: (1 - Tversky)^gamma.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + (self.alpha * fp) + (self.beta * fn) + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()


def get_loss_function(name: str = "bce_dice", **kwargs) -> nn.Module:
    """Factory function for loss functions."""
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        return DiceLoss(**kwargs)
    elif name == "bce_dice":
        return BCEDiceLoss(**kwargs)
    elif name == "focal":
        return FocalLoss(**kwargs)
    elif name == "tversky":
        return TverskyLoss(**kwargs)
    elif name == "focal_tversky":
        return FocalTverskyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function name: {name}")
