"""
Mask generator module for LIDC-IDRI annotations.

Converts vector polygon contours from XML radiologist markups into
pixel-aligned binary 2D and 3D segmentation masks.
Supports consensus, union, and majority reader agreement aggregation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw

from src.xml_parser import NoduleAnnotation, NoduleContourROI


class MaskGenerator:
    """
    Rasters polygon contours into binary masks and associates them with CT slices.
    """

    def __init__(
        self,
        slice_shape: Tuple[int, int] = (512, 512),
        aggregation_strategy: str = "consensus_50"
    ):
        self.height, self.width = slice_shape
        self.aggregation_strategy = aggregation_strategy

    def rasterize_contour(self, coords: List[Tuple[int, int]]) -> np.ndarray:
        """
        Rasterize list of (x, y) coordinates into a 2D binary numpy array.
        """
        if len(coords) < 3:
            # Cannot form a valid polygon with fewer than 3 vertices
            return np.zeros((self.height, self.width), dtype=np.uint8)

        # Create PIL Image mask
        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        # Convert coords to flat list or list of tuples
        draw.polygon(coords, outline=1, fill=1)
        return np.array(img, dtype=np.uint8)

    def generate_slice_masks(
        self,
        nodules: List[NoduleAnnotation],
        sop_instance_uids: List[str],
        z_positions: List[float],
        tolerance_z: float = 1.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates binary masks for all slices in a CT series.

        Returns:
            volume_mask: np.ndarray of shape (D, H, W), dtype uint8 (0 or 1)
            qc_report: Dict with mask statistics and quality control information
        """
        num_slices = len(sop_instance_uids)
        volume_mask = np.zeros((num_slices, self.height, self.width), dtype=np.uint8)

        # Map slice index by SOP UID and by Z position
        sop_to_idx = {sop: idx for idx, sop in enumerate(sop_instance_uids) if sop}
        
        # Per-slice, per-reader masks: slice_idx -> reader_id -> list of masks
        slice_reader_masks: Dict[int, Dict[int, List[np.ndarray]]] = {
            i: {} for i in range(num_slices)
        }

        total_rois_processed = 0
        rois_matched_by_sop = 0
        rois_matched_by_z = 0
        unmatched_rois = 0

        for nodule in nodules:
            for roi in nodule.rois:
                total_rois_processed += 1
                slice_idx = None

                # 1. Match by SOPInstanceUID (DICOM standard)
                if roi.sop_uid and roi.sop_uid in sop_to_idx:
                    slice_idx = sop_to_idx[roi.sop_uid]
                    rois_matched_by_sop += 1
                # 2. Match by closest Z position
                elif roi.z_position is not None and z_positions:
                    z_diffs = [abs(z - roi.z_position) for z in z_positions]
                    min_diff_idx = int(np.argmin(z_diffs))
                    if z_diffs[min_diff_idx] <= tolerance_z:
                        slice_idx = min_diff_idx
                        rois_matched_by_z += 1

                if slice_idx is not None:
                    raster_mask = self.rasterize_contour(roi.coords)
                    reader_id = nodule.reader_id
                    if reader_id not in slice_reader_masks[slice_idx]:
                        slice_reader_masks[slice_idx][reader_id] = []
                    slice_reader_masks[slice_idx][reader_id].append(raster_mask)
                else:
                    unmatched_rois += 1

        # Aggregate reader annotations per slice
        positive_slices = 0
        reader_counts_per_slice = []

        for s_idx in range(num_slices):
            readers = slice_reader_masks[s_idx]
            if not readers:
                continue

            num_readers = len(readers)
            reader_counts_per_slice.append(num_readers)

            # Combine multiple ROIs from the same reader on the same slice (union)
            reader_combined_masks = []
            for r_id, masks in readers.items():
                r_mask = np.bitwise_or.reduce(masks)
                reader_combined_masks.append(r_mask)

            # Aggregate across multiple readers
            stacked = np.stack(reader_combined_masks, axis=0)  # (R, H, W)
            agreement_map = np.sum(stacked, axis=0)           # values in [0, R]

            if self.aggregation_strategy == "union":
                slice_mask = (agreement_map >= 1).astype(np.uint8)
            elif self.aggregation_strategy == "majority":
                threshold = (num_readers / 2.0)
                slice_mask = (agreement_map > threshold).astype(np.uint8)
            elif self.aggregation_strategy == "consensus_50":
                # At least 50% reader agreement
                threshold = int(np.ceil(num_readers * 0.5))
                slice_mask = (agreement_map >= threshold).astype(np.uint8)
            else:
                slice_mask = (agreement_map >= 1).astype(np.uint8)

            volume_mask[s_idx] = slice_mask
            if np.any(slice_mask > 0):
                positive_slices += 1

        qc_report = {
            "total_rois_processed": total_rois_processed,
            "rois_matched_by_sop": rois_matched_by_sop,
            "rois_matched_by_z": rois_matched_by_z,
            "unmatched_rois": unmatched_rois,
            "num_slices": num_slices,
            "positive_slices": positive_slices,
            "negative_slices": num_slices - positive_slices,
            "positive_slice_indices": [i for i in range(num_slices) if np.any(volume_mask[i] > 0)],
            "reader_counts": reader_counts_per_slice,
            "aggregation_strategy": self.aggregation_strategy
        }

        return volume_mask, qc_report
