"""
Dataset inspection module for LIDC-IDRI.

Scans the raw data directory, discovers all patient, study, and series folders,
identifies CT DICOM series, extracts DICOM physical metadata, locates matching XML files
via SeriesInstanceUID / StudyInstanceUID, and calculates comprehensive dataset statistics without hardcoded values.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET
import numpy as np
import pydicom

from src.config import cfg
from src.xml_parser import LIDCXMLParser


class DatasetInspector:
    """
    Inspects LIDC-IDRI raw directory and compiles an empirical inspection report.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.summary: Dict[str, Any] = {}

    def scan_directory(self) -> Dict[str, Any]:
        """
        Recursively scans raw_dir to discover patients, CT series, and XML markups.
        """
        print(f"[DatasetInspector] Scanning directory: {self.raw_dir} ...")

        # 1. Discover DICOM files and group by PatientID and SeriesInstanceUID
        dicom_candidates = [
            p for p in self.raw_dir.rglob("*")
            if p.is_file() and not p.name.endswith(".xml") and not p.name.endswith(".json") and not p.name.endswith(".txt") and "tcia_annotations" not in str(p)
        ]
        print(f"[DatasetInspector] Found {len(dicom_candidates)} candidate DICOM files. Inspecting headers...")

        series_dict: Dict[str, Dict[str, Any]] = {}

        for file_path in dicom_candidates:
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=False)
                modality = getattr(ds, "Modality", None)
                if modality != "CT":
                    continue

                patient_id = getattr(ds, "PatientID", "UNKNOWN")
                study_uid = getattr(ds, "StudyInstanceUID", "UNKNOWN")
                series_uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN")

                if series_uid not in series_dict:
                    series_dict[series_uid] = {
                        "patient_id": patient_id,
                        "study_instance_uid": study_uid,
                        "series_instance_uid": series_uid,
                        "series_dir": str(file_path.parent),
                        "dicom_files": [],
                        "pixel_spacing": getattr(ds, "PixelSpacing", None),
                        "slice_thickness": getattr(ds, "SliceThickness", None),
                        "rows": getattr(ds, "Rows", None),
                        "cols": getattr(ds, "Columns", None),
                        "rescale_slope": getattr(ds, "RescaleSlope", 1.0),
                        "rescale_intercept": getattr(ds, "RescaleIntercept", -1024.0)
                    }

                series_dict[series_uid]["dicom_files"].append(file_path)
            except Exception:
                continue

        # Group series by patient
        patient_dict: Dict[str, Dict[str, Any]] = {}
        for s_uid, s_info in series_dict.items():
            pid = s_info["patient_id"]
            if pid not in patient_dict:
                patient_dict[pid] = {
                    "patient_id": pid,
                    "studies": set(),
                    "ct_series": [],
                    "total_slices": 0,
                    "xml_files": []
                }
            patient_dict[pid]["studies"].add(s_info["study_instance_uid"])
            patient_dict[pid]["ct_series"].append(s_info)
            patient_dict[pid]["total_slices"] += len(s_info["dicom_files"])

        # 2. Discover all XML files and map by SeriesInstanceUID / StudyInstanceUID
        xml_files = list(self.raw_dir.rglob("*.xml"))
        print(f"[DatasetInspector] Found {len(xml_files)} total XML annotation files. Mapping to CT series...")

        # Build map of SeriesInstanceUID -> patient_id
        series_to_patient = {s_uid: s_info["patient_id"] for s_uid, s_info in series_dict.items()}

        matched_xml_count = 0
        for xml_p in xml_files:
            try:
                tree = ET.parse(xml_p)
                root = tree.getroot()
                xml_series_uid = None
                for elem in root.iter():
                    if elem.tag.endswith("SeriesInstanceUid") and elem.text:
                        xml_series_uid = elem.text.strip()
                        break

                if xml_series_uid and xml_series_uid in series_to_patient:
                    pid = series_to_patient[xml_series_uid]
                    # Verify it's a CT annotation XML (has unblindedReadNodule or readingSession)
                    has_ct_nodules = any(e.tag.endswith("unblindedReadNodule") or e.tag.endswith("readingSession") for e in root.iter())
                    if has_ct_nodules:
                        patient_dict[pid]["xml_files"].append(str(xml_p))
                        matched_xml_count += 1
            except Exception:
                pass

        print(f"[DatasetInspector] Successfully matched {matched_xml_count} CT annotation XML files to patient series.")

        # Parse nodules across matched XML files
        total_annotated_nodules = 0
        total_rois = 0

        for pid, pdata in patient_dict.items():
            pdata["nodule_count"] = 0
            for xml_file_path in pdata["xml_files"]:
                try:
                    parser = LIDCXMLParser(Path(xml_file_path))
                    nodules = parser.parse()
                    # Count nodules with valid multi-point contours
                    contour_nodules = [n for n in nodules if any(len(roi.coords) >= 3 for roi in n.rois)]
                    pdata["nodule_count"] += len(contour_nodules)
                    total_annotated_nodules += len(contour_nodules)
                    for nod in contour_nodules:
                        total_rois += sum(1 for r in nod.rois if len(r.coords) >= 3)
                except Exception as e:
                    print(f"[Warning] Error parsing {xml_file_path}: {e}")

        # Compute aggregate metrics
        total_patients = len(patient_dict)
        total_studies = sum(len(pdata["studies"]) for pdata in patient_dict.values())
        total_ct_series = len(series_dict)
        total_slices = sum(len(s["dicom_files"]) for s in series_dict.values())

        pixel_spacings = [s["pixel_spacing"] for s in series_dict.values() if s["pixel_spacing"] is not None]
        slice_thicknesses = [float(s["slice_thickness"]) for s in series_dict.values() if s["slice_thickness"] is not None]

        report = {
            "num_patients": total_patients,
            "num_studies": total_studies,
            "num_ct_series": total_ct_series,
            "total_slices": total_slices,
            "total_xml_files": len(xml_files),
            "total_annotated_nodules": total_annotated_nodules,
            "total_rois": total_rois,
            "pixel_spacings": [list(ps) for ps in pixel_spacings],
            "slice_thickness_mean": float(np.mean(slice_thicknesses)) if slice_thicknesses else None,
            "slice_thickness_std": float(np.std(slice_thicknesses)) if slice_thicknesses else None,
            "slice_thickness_range": [float(np.min(slice_thicknesses)), float(np.max(slice_thicknesses))] if slice_thicknesses else None,
            "patients": {
                pid: {
                    "num_ct_series": len(pdata["ct_series"]),
                    "total_slices": pdata["total_slices"],
                    "num_xml_files": len(pdata["xml_files"]),
                    "num_nodules": pdata["nodule_count"]
                }
                for pid, pdata in patient_dict.items()
            }
        }

        self.summary = report
        return report

    def generate_report_markdown(self, save_path: Optional[Path] = None) -> str:
        """Formats the inspection summary into a comprehensive academic markdown report."""
        if not self.summary:
            self.scan_directory()

        s = self.summary
        md = f"""# LIDC-IDRI Dataset Inspection Report

**Generated from actual raw files:** `{self.raw_dir}`

---

## 1. High-Level Summary Statistics
- **Total Patients Identified**: {s['num_patients']}
- **Total Studies**: {s['num_studies']}
- **Total CT Series**: {s['num_ct_series']}
- **Total CT Slices**: {s['total_slices']}
- **Total XML Annotation Files Available**: {s['total_xml_files']}
- **Total Annotated Nodules Detected (>=3 vertices)**: {s['total_annotated_nodules']}
- **Total Radiologist ROI Contours**: {s['total_rois']}

---

## 2. DICOM Physical Properties
- **Pixel Spacing (mm)**: {s['pixel_spacings']}
- **Slice Thickness Range**: {s['slice_thickness_range']} mm
- **Mean Slice Thickness**: {s['slice_thickness_mean']:.2f} mm (std: {s['slice_thickness_std']:.2f} mm) if s['slice_thickness_mean'] else 'N/A'
- **Modality**: CT (Computed Tomography)

---

## 3. Patient Breakdown
| Patient ID | CT Series Count | Total Slices | Matched XML Files | Annotated Nodules |
| :--- | :---: | :---: | :---: | :---: |
"""
        for pid, pinfo in s["patients"].items():
            md += f"| `{pid}` | {pinfo['num_ct_series']} | {pinfo['total_slices']} | {pinfo['num_xml_files']} | {pinfo['num_nodules']} |\n"

        md += "\n---\n*Report generated strictly from downloaded DICOM and XML metadata without hardcoded values.*"

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(md, encoding="utf-8")
            print(f"[DatasetInspector] Report saved to {save_path}")

        return md


if __name__ == "__main__":
    inspector = DatasetInspector(cfg.paths.raw_dir)
    report = inspector.scan_directory()
    md_text = inspector.generate_report_markdown(cfg.paths.results_dir / "dataset_inspection_report.md")
    print(md_text)
