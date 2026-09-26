# LIDC-IDRI Dataset Inspection Report

**Generated from actual raw files:** `C:\Users\abhi8\OneDrive\Desktop\ACADEMIC DOCS\SEM-5\CV\Project\LIDC_Attention_UNet\data\raw`

---

## 1. High-Level Summary Statistics
- **Total Patients Identified**: 3
- **Total Studies**: 3
- **Total CT Series**: 3
- **Total CT Slices**: 406
- **Total XML Annotation Files Available**: 1319
- **Total Annotated Nodules Detected (>=3 vertices)**: 26
- **Total Radiologist ROI Contours**: 127

---

## 2. DICOM Physical Properties
- **Pixel Spacing (mm)**: [['0.703125', '0.703125'], ['0.820312', '0.820312'], ['0.664062', '0.664062']]
- **Slice Thickness Range**: [2.5, 2.5] mm
- **Mean Slice Thickness**: 2.50 mm (std: 0.00 mm) if s['slice_thickness_mean'] else 'N/A'
- **Modality**: CT (Computed Tomography)

---

## 3. Patient Breakdown
| Patient ID | CT Series Count | Total Slices | Matched XML Files | Annotated Nodules |
| :--- | :---: | :---: | :---: | :---: |
| `LIDC-IDRI-0001` | 1 | 133 | 2 | 4 |
| `LIDC-IDRI-0003` | 1 | 140 | 2 | 13 |
| `LIDC-IDRI-0005` | 1 | 133 | 2 | 9 |

---
*Report generated strictly from downloaded DICOM and XML metadata without hardcoded values.*