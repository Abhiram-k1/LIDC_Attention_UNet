"""
Configuration module for LIDC_Attention_UNet.

Defines all paths, dataset parameters, preprocessing choices,
model architectures, training hyperparameters, and reproducibility seeds.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch


@dataclass
class PathConfig:
    # Base directory of the repository
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Data directories
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    splits_dir: Path = field(init=False)

    # Outputs
    checkpoints_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    attention_maps_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    configs_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"

        self.checkpoints_dir = self.project_root / "checkpoints"
        self.results_dir = self.project_root / "results"
        self.metrics_dir = self.results_dir / "metrics"
        self.plots_dir = self.results_dir / "plots"
        self.attention_maps_dir = self.results_dir / "attention_maps"
        self.predictions_dir = self.results_dir / "predictions"
        self.configs_dir = self.project_root / "configs"

        for p in [
            self.raw_dir, self.processed_dir, self.splits_dir,
            self.checkpoints_dir, self.metrics_dir, self.plots_dir,
            self.attention_maps_dir, self.predictions_dir, self.configs_dir
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class CTWindowConfig:
    # Standard lung CT windowing parameters (in Hounsfield Units)
    window_center: float = -600.0  # HU
    window_width: float = 1500.0   # HU
    # Calculated window min and max
    @property
    def window_min(self) -> float:
        return self.window_center - (self.window_width / 2.0)

    @property
    def window_max(self) -> float:
        return self.window_center + (self.window_width / 2.0)


@dataclass
class PreprocessingConfig:
    window: CTWindowConfig = field(default_factory=CTWindowConfig)
    target_size: Tuple[int, int] = (256, 256)
    # Output normalization: "zero_to_one" ([0, 1]) or "minus_one_to_one" ([-1, 1])
    normalization: str = "zero_to_one"
    # Slice selection:
    # 1. include all slices containing annotated nodule contours
    # 2. include neighbor_slices on either side of nodule slices
    neighbor_slices: int = 1
    # 3. negative slice sampling ratio: ratio of negative slices (no nodule) to positive slices
    negative_sample_ratio: float = 0.2
    # Multiple radiologist aggregation strategy: "consensus_50" (>=50% agreement), "union", "majority"
    annotation_aggregation: str = "consensus_50"


@dataclass
class AugmentationConfig:
    # Medically plausible transformations for 2D lung CT slices
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = False  # Anatomically inverted lungs are unrealistic
    rotation_degrees: float = 10.0      # Mild rotation within +/- 10 degrees
    translation_range: Tuple[float, float] = (0.05, 0.05)
    scaling_range: Tuple[float, float] = (0.95, 1.05)
    brightness_contrast_delta: float = 0.05


@dataclass
class DataSplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)  # [32, 64, 128, 256]
    bilinear_upsample: bool = True
    dropout_rate: float = 0.1

    # Proposed architecture hyperparameters:
    # 1. Spatial Attention reduction ratio
    spatial_reduction: int = 4
    # 2. Sparse Attention selection (top-k positions per query or local neighborhood)
    sparse_top_k: int = 32
    # 3. Linear Attention kernel feature map: "elu" (phi(x) = elu(x) + 1)
    linear_kernel: str = "elu"
    # 4. Cross-Feature fusion dimension
    cross_feature_dim: int = 64


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 0  # 0 is safe for Windows multiprocessing
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 8
    loss_function: str = "bce_dice"  # "bce", "dice", "bce_dice", "focal_tversky"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    seed: int = 42
    use_amp: bool = False  # Automatic Mixed Precision (CPU typically False, CUDA True)


def get_device() -> torch.device:
    """
    Detect compute device strictly adhering to platform support:
    NVIDIA CUDA -> Apple Silicon MPS -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@dataclass
class GlobalConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    splits: DataSplitConfig = field(default_factory=DataSplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: torch.device = field(default_factory=get_device)


# Singleton instance for simple imports
cfg = GlobalConfig()
