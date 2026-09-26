"""
Preprocessing module for LIDC-IDRI CT images and segmentation masks.

Handles:
- Hounsfield Unit windowing (configurable center and width)
- Value normalization to [0, 1] or [-1, 1]
- High-fidelity resizing (bilinear for CT image, nearest-neighbor for mask)
- Configurable slice selection (nodule slices, neighboring margin, controlled negative sampling)
- Exporting processed samples to disk with full traceability
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from src.config import cfg, CTWindowConfig
from src.dicom_loader import DICOMSeriesLoader
from src.xml_parser import LIDCXMLParser
from src.mask_generator import MaskGenerator


class CTPreprocessor:
    """
    Applies medical windowing, normalization, resizing, and slice selection.
    """

    def __init__(
        self,
        window: Optional[CTWindowConfig] = None,
        target_size: Tuple[int, int] = (256, 256),
        normalization: str = "zero_to_one",
        neighbor_margin: int = 1,
        negative_ratio: float = 0.2
    ):
        self.window = window or cfg.preprocessing.window
        self.target_size = target_size
        self.normalization = normalization
        self.neighbor_margin = neighbor_margin
        self.negative_ratio = negative_ratio

    def apply_windowing_and_norm(self, hu_image: np.ndarray) -> np.ndarray:
        """
        Clips HU values to the lung window and normalizes.
        """
        w_min = self.window.window_min
        w_max = self.window.window_max

        # Clip to window
        clipped = np.clip(hu_image, w_min, w_max)

        # Normalize to [0, 1]
        norm = (clipped - w_min) / (w_max - w_min)

        if self.normalization == "minus_one_to_one":
            norm = (norm * 2.0) - 1.0

        return norm.astype(np.float32)

    def resize_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes image using bilinear interpolation and mask using nearest-neighbor.
        """
        target_w, target_h = self.target_size

        # Image: Bilinear
        pil_img = Image.fromarray(image)
        res_img = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
        out_img = np.array(res_img, dtype=np.float32)

        # Mask: Nearest-neighbor to preserve strictly binary {0, 1} values
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        res_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
        out_mask = np.array(res_mask, dtype=np.uint8)

        # Ensure mask is binary
        out_mask = (out_mask > 0).astype(np.uint8)

        return out_img, out_mask

    def select_slices(
        self,
        volume_mask: np.ndarray,
        random_seed: int = 42
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Selects slices to include based on nodule presence, neighbors, and negative sampling ratio.
        """
        num_slices = volume_mask.shape[0]
        nodule_slice_indices = [i for i in range(num_slices) if np.any(volume_mask[i] > 0)]

        # If series has no annotated nodules, sample negative slices
        if not nodule_slice_indices:
            num_samples = max(1, int(num_slices * self.negative_ratio))
            rng = np.random.RandomState(random_seed)
            selected = sorted(rng.choice(num_slices, size=min(num_samples, num_slices), replace=False).tolist())
            stats = {
                "total_slices": num_slices,
                "positive_slices": 0,
                "neighbor_slices": 0,
                "negative_slices": len(selected),
                "selected_indices": selected
            }
            return selected, stats

        # Find neighboring slices
        selected_set = set(nodule_slice_indices)
        for idx in nodule_slice_indices:
            for offset in range(-self.neighbor_margin, self.neighbor_margin + 1):
                neighbor = idx + offset
                if 0 <= neighbor < num_slices:
                    selected_set.add(neighbor)

        # Negative slice sampling from non-nodule, non-neighbor slices
        all_indices = set(range(num_slices))
        remaining_negatives = list(all_indices - selected_set)
        
        target_neg_count = int(len(nodule_slice_indices) * self.negative_ratio)
        rng = np.random.RandomState(random_seed)
        if remaining_negatives and target_neg_count > 0:
            sampled_negatives = rng.choice(
                remaining_negatives,
                size=min(target_neg_count, len(remaining_negatives)),
                replace=False
            ).tolist()
            selected_set.update(sampled_negatives)

        final_selected = sorted(list(selected_set))

        pos_count = sum(1 for i in final_selected if np.any(volume_mask[i] > 0))
        neg_count = len(final_selected) - pos_count

        stats = {
            "total_slices": num_slices,
            "positive_slices": pos_count,
            "negative_slices": neg_count,
            "positive_negative_ratio": float(pos_count / max(1, neg_count)),
            "selected_indices": final_selected
        }

        return final_selected, stats

    def process_series(
        self,
        series_dir: Path,
        xml_path: Optional[Path],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Loads CT series, computes masks from XML, applies preprocessing,
        and saves each selected slice as a .npz archive in output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = DICOMSeriesLoader(series_dir)
        volume, meta = loader.load_series()

        nodules = []
        if xml_path and xml_path.is_file():
            parser = LIDCXMLParser(xml_path)
            nodules = parser.parse()

        mask_gen = MaskGenerator(
            slice_shape=(volume.shape[1], volume.shape[2]),
            aggregation_strategy=cfg.preprocessing.annotation_aggregation
        )
        volume_mask, qc = mask_gen.generate_slice_masks(
            nodules=nodules,
            sop_instance_uids=meta["sop_instance_uids"],
            z_positions=meta["z_positions"]
        )

        selected_indices, slice_stats = self.select_slices(volume_mask)

        saved_records = []
        patient_id = meta["patient_id"]
        series_uid = meta["series_instance_uid"]

        for s_idx in selected_indices:
            hu_slice = volume[s_idx]
            raw_mask = volume_mask[s_idx]

            # 1. Window & Normalize
            norm_slice = self.apply_windowing_and_norm(hu_slice)

            # 2. Resize
            res_slice, res_mask = self.resize_pair(norm_slice, raw_mask)

            # 3. Save as .npz
            filename = f"{patient_id}_{series_uid[:8]}_slice_{s_idx:04d}.npz"
            file_path = output_dir / filename

            sop_uid = meta["sop_instance_uids"][s_idx] if s_idx < len(meta["sop_instance_uids"]) else ""
            z_pos = meta["z_positions"][s_idx] if s_idx < len(meta["z_positions"]) else 0.0

            np.savez_compressed(
                file_path,
                image=res_slice,
                mask=res_mask,
                patient_id=patient_id,
                study_uid=meta["study_instance_uid"],
                series_uid=series_uid,
                slice_index=s_idx,
                sop_instance_uid=sop_uid,
                z_position=z_pos,
                is_positive=int(np.any(res_mask > 0))
            )

            record = {
                "file_path": str(file_path),
                "filename": filename,
                "patient_id": patient_id,
                "slice_index": s_idx,
                "is_positive": int(np.any(res_mask > 0)),
                "foreground_pixels": int(np.sum(res_mask > 0)),
                "foreground_ratio": float(np.sum(res_mask > 0) / (res_mask.shape[0] * res_mask.shape[1]))
            }
            saved_records.append(record)

        return saved_records
