"""
DICOM loader module for LIDC-IDRI CT scans.

Handles reading DICOM files, extracting metadata, spatial sorting by
ImagePositionPatient, converting raw pixel data to Hounsfield Units (HU),
and constructing complete 3D CT volumes with accurate voxel spacing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pydicom
from pydicom.dataset import FileDataset


class DICOMSeriesLoader:
    """
    Loads and processes a CT DICOM series from a directory.
    Enforces strict physical spatial sorting and Hounsfield Unit conversion.
    """

    def __init__(self, series_dir: Path):
        self.series_dir = Path(series_dir)
        self.dicom_files: List[Path] = []
        self.datasets: List[FileDataset] = []
        self.volume: Optional[np.ndarray] = None
        self.voxel_spacing: Optional[Tuple[float, float, float]] = None  # (dz, dy, dx)
        self.metadata: Dict[str, Any] = {}
        self.slice_z_positions: List[float] = []
        self.sop_instance_uids: List[str] = []

    def scan_files(self) -> List[Path]:
        """Find all valid DICOM files in the directory."""
        candidates = [
            p for p in self.series_dir.rglob("*")
            if p.is_file() and not p.name.endswith(".xml") and not p.name.endswith(".json")
        ]
        valid_dicoms = []
        for p in candidates:
            try:
                # Read preamble and standard DICOM header
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
                # Verify modality is CT
                if getattr(ds, "Modality", None) == "CT":
                    valid_dicoms.append(p)
            except Exception:
                continue

        self.dicom_files = valid_dicoms
        return self.dicom_files

    def load_series(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads all slices, sorts them by spatial coordinates,
        converts pixel values to Hounsfield Units, and constructs the 3D volume.
        """
        if not self.dicom_files:
            self.scan_files()

        if not self.dicom_files:
            raise ValueError(f"No valid CT DICOM slices found in: {self.series_dir}")

        raw_slices = []
        for file_path in self.dicom_files:
            try:
                ds = pydicom.dcmread(str(file_path))
                raw_slices.append(ds)
            except Exception as e:
                print(f"[Warning] Failed to read {file_path.name}: {e}")

        if not raw_slices:
            raise RuntimeError(f"Could not read any DICOM datasets in {self.series_dir}")

        # Spatial sorting: Use ImagePositionPatient projected along normal vector
        # or z-coordinate
        sorted_slices = self._sort_slices_spatially(raw_slices)
        self.datasets = sorted_slices

        # Extract metadata from reference slice
        ref = sorted_slices[0]
        patient_id = getattr(ref, "PatientID", "UNKNOWN")
        study_uid = getattr(ref, "StudyInstanceUID", "UNKNOWN")
        series_uid = getattr(ref, "SeriesInstanceUID", "UNKNOWN")
        slice_thickness = float(getattr(ref, "SliceThickness", 2.5))
        pixel_spacing = getattr(ref, "PixelSpacing", [1.0, 1.0])
        dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])

        # Compute accurate z-spacing from sorted slices
        z_positions = []
        sop_uids = []
        for ds in sorted_slices:
            sop_uids.append(getattr(ds, "SOPInstanceUID", ""))
            if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) == 3:
                z_positions.append(float(ds.ImagePositionPatient[2]))
            else:
                z_positions.append(0.0)

        self.slice_z_positions = z_positions
        self.sop_instance_uids = sop_uids

        if len(z_positions) > 1:
            diffs = np.abs(np.diff(z_positions))
            valid_diffs = diffs[diffs > 1e-4]
            dz = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else slice_thickness
        else:
            dz = slice_thickness

        self.voxel_spacing = (dz, dy, dx)

        # Convert each slice to Hounsfield Units
        hu_slices = []
        for ds in sorted_slices:
            hu_slice = self._convert_to_hu(ds)
            hu_slices.append(hu_slice)

        self.volume = np.stack(hu_slices, axis=0).astype(np.float32)  # Shape: (Depth, Height, Width)

        self.metadata = {
            "patient_id": patient_id,
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "num_slices": len(sorted_slices),
            "slice_shape": (self.volume.shape[1], self.volume.shape[2]),
            "voxel_spacing": self.voxel_spacing,
            "z_positions": z_positions,
            "sop_instance_uids": sop_uids,
            "hu_min": float(self.volume.min()),
            "hu_max": float(self.volume.max()),
            "hu_mean": float(self.volume.mean()),
            "hu_std": float(self.volume.std())
        }

        return self.volume, self.metadata

    @staticmethod
    def _sort_slices_spatially(slices: List[FileDataset]) -> List[FileDataset]:
        """
        Sort slices using ImagePositionPatient.
        Projects position onto the slice normal computed from ImageOrientationPatient.
        """
        def get_slice_sort_key(ds: FileDataset) -> float:
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
                try:
                    iop = [float(x) for x in ds.ImageOrientationPatient]
                    ipp = [float(x) for x in ds.ImagePositionPatient]
                    # Row and Column vectors
                    r = np.array(iop[:3])
                    c = np.array(iop[3:])
                    # Normal vector is cross product
                    normal = np.cross(r, c)
                    # Project position onto normal
                    return float(np.dot(ipp, normal))
                except Exception:
                    pass

            if hasattr(ds, "ImagePositionPatient"):
                try:
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    pass

            if hasattr(ds, "SliceLocation"):
                try:
                    return float(ds.SliceLocation)
                except Exception:
                    pass

            if hasattr(ds, "InstanceNumber"):
                try:
                    return float(ds.InstanceNumber)
                except Exception:
                    pass

            return 0.0

        return sorted(slices, key=get_slice_sort_key)

    @staticmethod
    def _convert_to_hu(ds: FileDataset) -> np.ndarray:
        """
        Converts raw DICOM pixel array to Hounsfield Units (HU):
        HU = pixel_array * RescaleSlope + RescaleIntercept
        """
        pixel_array = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", -1024.0))

        if slope != 1.0:
            pixel_array = slope * pixel_array
        pixel_array = pixel_array + intercept

        # Handle vendor padding values outside field of view (often -2000 or -1000)
        # Standard air in CT is approximately -1000 HU
        pixel_array[pixel_array < -1024.0] = -1024.0

        return pixel_array
