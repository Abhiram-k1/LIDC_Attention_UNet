"""
PyTorch Dataset and DataLoader module for LIDC-IDRI slices.

Enforces:
1. Strict patient-level train/val/test isolation (zero patient leakage).
2. Medically realistic augmentations on training data only.
3. Deterministic loading for validation and testing.
4. Rich metadata return for full provenance and traceability.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.config import cfg, AugmentationConfig


class Medical2DAugmentor:
    """
    Applies medically plausible augmentations to paired (CT image, binary mask).
    """

    def __init__(self, config: AugmentationConfig, seed: int = 42):
        self.cfg = config
        self.rng = random.Random(seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # image: (H, W) float32 in [0, 1]
        # mask: (H, W) uint8 in {0, 1}

        # 1. Random Horizontal Flip
        if self.cfg.random_horizontal_flip and self.rng.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 2. Mild Random Rotation (+/- degrees)
        if self.cfg.rotation_degrees > 0:
            deg = self.rng.uniform(-self.cfg.rotation_degrees, self.cfg.rotation_degrees)
            # Image: Bilinear
            pil_img = Image.fromarray(image)
            rot_img = pil_img.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
            image = np.array(rot_img, dtype=np.float32)

            # Mask: Nearest
            pil_mask = Image.fromarray(mask)
            rot_mask = pil_mask.rotate(deg, resample=Image.NEAREST, fillcolor=0)
            mask = np.array(rot_mask, dtype=np.uint8)

        # 3. Mild Scaling / Zoom
        if self.cfg.scaling_range:
            scale = self.rng.uniform(self.cfg.scaling_range[0], self.cfg.scaling_range[1])
            if abs(scale - 1.0) > 0.01:
                h, w = image.shape
                nh, nw = int(h * scale), int(w * scale)
                
                # Image
                pil_img = Image.fromarray(image).resize((nw, nh), resample=Image.BILINEAR)
                # Mask
                pil_mask = Image.fromarray(mask).resize((nw, nh), resample=Image.NEAREST)

                # Crop or pad back to (w, h)
                img_arr = np.array(pil_img, dtype=np.float32)
                mask_arr = np.array(pil_mask, dtype=np.uint8)

                if scale > 1.0:
                    # Center crop
                    dh = (nh - h) // 2
                    dw = (nw - w) // 2
                    image = img_arr[dh:dh+h, dw:dw+w]
                    mask = mask_arr[dh:dh+h, dw:dw+w]
                else:
                    # Pad with 0
                    pad_img = np.zeros((h, w), dtype=np.float32)
                    pad_mask = np.zeros((h, w), dtype=np.uint8)
                    dh = (h - nh) // 2
                    dw = (w - nw) // 2
                    pad_img[dh:dh+nh, dw:dw+nw] = img_arr
                    pad_mask[dh:dh+nh, dw:dw+nw] = mask_arr
                    image = pad_img
                    mask = pad_mask

        # 4. Mild Brightness / Contrast shift
        if self.cfg.brightness_contrast_delta > 0:
            delta = self.rng.uniform(-self.cfg.brightness_contrast_delta, self.cfg.brightness_contrast_delta)
            image = np.clip(image + delta, 0.0, 1.0)

        # Ensure mask remains strictly binary
        mask = (mask > 0).astype(np.uint8)

        return image, mask


class LIDCROIDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed LIDC-IDRI .npz slices.
    """

    def __init__(
        self,
        sample_paths: List[Path],
        is_train: bool = False,
        augmentor: Optional[Medical2DAugmentor] = None
    ):
        self.sample_paths = [Path(p) for p in sample_paths]
        self.is_train = is_train
        self.augmentor = augmentor if (is_train and augmentor is not None) else None

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.sample_paths[idx]
        data = np.load(str(path), allow_pickle=True)

        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)

        # Apply augmentation if training
        if self.is_train and self.augmentor:
            image, mask = self.augmentor(image, mask)

        # Convert to PyTorch tensors: Shape (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Metadata
        meta = {
            "patient_id": str(data["patient_id"]),
            "study_uid": str(data["study_uid"]),
            "series_uid": str(data["series_uid"]),
            "slice_index": int(data["slice_index"]),
            "sop_instance_uid": str(data["sop_instance_uid"]),
            "is_positive": int(data["is_positive"]),
            "file_path": str(path)
        }

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": meta
        }


class PatientSplitter:
    """
    Splits patient IDs into isolated Train / Val / Test sets.
    Ensures zero cross-split patient overlap to eliminate data leakage.
    """

    @staticmethod
    def create_split(
        patient_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        unique_patients = sorted(list(set(patient_ids)))
        rng = random.Random(seed)
        shuffled = unique_patients.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        
        # Ensure test set gets remaining
        train_pts = shuffled[:n_train]
        val_pts = shuffled[n_train:n_train + n_val]
        test_pts = shuffled[n_train + n_val:]

        # Handle edge cases with small numbers of patients
        if not test_pts and len(val_pts) > 1:
            test_pts = [val_pts.pop()]
        elif not test_pts and len(train_pts) > 1:
            test_pts = [train_pts.pop()]

        # Verification of mutual exclusivity
        s_train = set(train_pts)
        s_val = set(val_pts)
        s_test = set(test_pts)

        assert len(s_train.intersection(s_val)) == 0, "Train and Val patients overlap!"
        assert len(s_train.intersection(s_test)) == 0, "Train and Test patients overlap!"
        assert len(s_val.intersection(s_test)) == 0, "Val and Test patients overlap!"

        return {
            "train": train_pts,
            "val": val_pts,
            "test": test_pts
        }

    @staticmethod
    def save_split(split_dict: Dict[str, List[str]], output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_path = output_dir / "patient_splits.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)
        print(f"[PatientSplitter] Saved patient splits to {split_path}")
        print(f"  Train patients: {len(split_dict['train'])} -> {split_dict['train']}")
        print(f"  Val patients:   {len(split_dict['val'])} -> {split_dict['val']}")
        print(f"  Test patients:  {len(split_dict['test'])} -> {split_dict['test']}")

    @staticmethod
    def load_split(split_path: Path) -> Dict[str, List[str]]:
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)


def build_dataloaders(
    processed_dir: Path,
    splits_dict: Dict[str, List[str]],
    batch_size: int = 8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    all_files = list(Path(processed_dir).glob("*.npz"))
    
    # Filter files by patient split
    train_files = []
    val_files = []
    test_files = []

    for f in all_files:
        # File name format: {patient_id}_{series_uid}_slice_xxxx.npz
        pid = f.stem.split("_")[0]
        if pid in splits_dict["train"]:
            train_files.append(f)
        elif pid in splits_dict["val"]:
            val_files.append(f)
        elif pid in splits_dict["test"]:
            test_files.append(f)

    print(f"[DataLoader] Slices count: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    augmentor = Medical2DAugmentor(cfg.augmentation, seed=cfg.training.seed)

    train_ds = LIDCROIDataset(train_files, is_train=True, augmentor=augmentor)
    val_ds = LIDCROIDataset(val_files, is_train=False)
    test_ds = LIDCROIDataset(test_files, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
