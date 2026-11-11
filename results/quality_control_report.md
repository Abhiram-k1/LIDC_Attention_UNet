# LIDC-IDRI Quality Control (QC) & Data Integrity Report

**Date of Audit**: Automatic Execution  
**Raw Data Directory**: `C:\Users\abhi8\OneDrive\Desktop\ACADEMIC DOCS\SEM-5\CV\LIDC_Attention_UNet\data\raw`  
**Processed Data Directory**: `C:\Users\abhi8\OneDrive\Desktop\ACADEMIC DOCS\SEM-5\CV\LIDC_Attention_UNet\data\processed`  

---

## 1. Raw DICOM & XML Integrity Audit
| Verification Item | Status | Measured Metric |
| :--- | :---: | :--- |
| **Total XML Annotation Files** | PASS | 1320 parsed |
| **Malformed XML Count** | PASS | 0 malformed files detected |
| **DICOM Slices Scanned** | PASS | 65 slices |
| **Missing DICOM Metadata** | PASS | 0 slices flagged |
| **Out-of-range HU Intensities** | PASS | 0 slices flagged |
| **Discovered Patient IDs** | PASS | ['LIDC-IDRI-0957'] |

---

## 2. Preprocessed Image-Mask Correspondence Audit
| Verification Item | Status | Measured Metric |
| :--- | :---: | :--- |
| **Total Processed Pairs** | PASS | 207 `.npz` slices |
| **Positive Nodule Slices** | PASS | 153 slices |
| **Negative Control Slices** | PASS | 54 slices |
| **Positive / Total Ratio** | PASS | 73.91% |
| **Image-Mask Dimension Mismatches** | PASS | 0 mismatches |
| **Binary Mask Value Violations (non-{0, 1})** | PASS | 0 violations |
| **NaN / Inf Corruptions** | PASS | 0 corrupted slices |

---

## 3. Discarded Data Summary
- **Silently Discarded Cases**: **0**
- **Malformed XML Discarded**: **0**
- **Dimension Corrupted Discarded**: **0**

*Conclusion*: Processed CT slices and nodule masks evaluated with measured spatial alignment rate of **100.0%**.
