"""
Annotation validation and visual verification script.

Generates visual proof of alignment between CT slices and XML contour masks:
- Panel 1: Original CT Slice
- Panel 2: Rasterized Radiologist Mask
- Panel 3: Green Outline Contour Overlay
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from src.config import cfg
from src.dicom_loader import DICOMSeriesLoader
from src.xml_parser import LIDCXMLParser
from src.mask_generator import MaskGenerator
from src.preprocessing import CTPreprocessor


def verify_patient_annotations(
    patient_id: str = "LIDC-IDRI-0001",
    output_dir: Optional[Path] = None
) -> Path:
    """
    Renders visual verification for the first slice with an annotated nodule.
    """
    output_dir = Path(output_dir) if output_dir else cfg.paths.plots_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dir = cfg.paths.raw_dir / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    # Find CT series directory
    series_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name != "tcia_annotations"]
    if not series_dirs:
        raise FileNotFoundError(f"No series directory in {patient_dir}")
    series_dir = series_dirs[0]
    series_uid = series_dir.name

    # Load series DICOM
    print(f"[Verification] Loading DICOM series for {patient_id}...")
    loader = DICOMSeriesLoader(series_dir)
    volume, meta = loader.load_series()

    # Find matching CT XML
    xml_matches = []
    for xml_p in cfg.paths.raw_dir.rglob("*.xml"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_p)
            for elem in tree.getroot().iter():
                if elem.tag.endswith("SeriesInstanceUid") and elem.text and elem.text.strip() == series_uid:
                    if any(e.tag.endswith("unblindedReadNodule") for e in tree.getroot().iter()):
                        xml_matches.append(xml_p)
                        break
        except Exception:
            pass

    if not xml_matches:
        print(f"[Warning] No matching CT XML found for {series_uid}")
        return output_dir

    xml_path = xml_matches[0]
    print(f"[Verification] Found matching XML: {xml_path.name}")
    parser = LIDCXMLParser(xml_path)
    nodules = parser.parse()

    # Generate masks
    mask_gen = MaskGenerator(
        slice_shape=(volume.shape[1], volume.shape[2]),
        aggregation_strategy=cfg.preprocessing.annotation_aggregation
    )
    volume_mask, qc = mask_gen.generate_slice_masks(
        nodules=nodules,
        sop_instance_uids=meta["sop_instance_uids"],
        z_positions=meta["z_positions"]
    )

    positive_slices = qc["positive_slice_indices"]
    if not positive_slices:
        print("[Verification] No positive slices detected with contours >= 3 points.")
        return output_dir

    # Pick the positive slice with the largest nodule area
    areas = [np.sum(volume_mask[idx]) for idx in positive_slices]
    best_slice_idx = positive_slices[int(np.argmax(areas))]
    print(f"[Verification] Best nodule slice: Index {best_slice_idx} with {np.sum(volume_mask[best_slice_idx])} pixels.")

    # Apply windowing for display
    prep = CTPreprocessor(window=cfg.preprocessing.window)
    display_img = prep.apply_windowing_and_norm(volume[best_slice_idx])
    mask = volume_mask[best_slice_idx]

    # Create 3-panel figure: [CT Slice | Binary Mask | Contour Overlay]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: CT Slice
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title(f"CT Slice (Z={meta['z_positions'][best_slice_idx]:.1f} mm)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Binary Mask
    axes[1].imshow(mask, cmap="Blues")
    axes[1].set_title(f"Radiologist Contour Mask ({qc['aggregation_strategy']})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Green Outline Overlay
    axes[2].imshow(display_img, cmap="gray")
    axes[2].contour(mask, colors="lime", linewidths=2.0)
    axes[2].set_title("Nodule Contour Overlay (Ground Truth)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"annotation_verification_{patient_id}_slice{best_slice_idx:04d}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Verification] Verification figure successfully saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    verify_patient_annotations("LIDC-IDRI-0001")
