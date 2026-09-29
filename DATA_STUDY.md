# DATA_STUDY.md — Comprehensive Data Study: Features, Distributions, and Preprocessing Infrastructure

**Project Title:** Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation  
**Document:** Authoritative Data Engineering, Feature Specification, and Data Study Report  
**Modality:** Helical Thoracic Computed Tomography (CT), DICOM Format  
**Primary Repository:** National Cancer Institute LIDC-IDRI (The Cancer Imaging Archive)  
**Cohort Audit Status:** **406 CT Slices, 1,319 XML Files, 127 Contours, 100% Quality Control Passed**  

---

## 1. Executive Summary & Clinical Background

Automated pulmonary nodule segmentation on thoracic Computed Tomography (CT) is a cornerstone of clinical Computer-Aided Detection (CAD) and Lung-RADS staging systems. Lung carcinoma is the leading cause of cancer mortality worldwide, and early detection of localized nodules ($3\text{ mm}$ to $30\text{ mm}$) substantially elevates 5-year patient survival rates.

However, deep learning on thoracic CT datasets presents four severe medical imaging challenges:
1. **Extreme Foreground Sparsity**: Solitary pulmonary nodules occupy less than **$0.2\%$** of an axial slice's total pixel area, and less than **$0.01\%$** of a patient's total 3D chest volume.
2. **Attenuation Ambiguity**: Pulmonary blood vessels, bronchial walls, and mediastinal structures share identical Hounsfield Unit (HU) densities with soft-tissue nodules ($[-100, +100]\text{ HU}$), making segmentation vulnerable to false positives.
3. **Scanner & Acquisition Heterogeneity**: Scans originating from different hospital imaging suites exhibit wide variations in X-ray tube currents, slice thicknesses ($1.25\text{ mm}$ to $2.5\text{ mm}$), and reconstruction filter kernels.
4. **Inter-Radiologist Boundary Discordance**: Because malignant nodules feature irregular, ground-glass, or spiculated margins, expert radiologists exhibit significant variance when drawing manual polygon contours.

To address these challenges with scientific rigor, this project establishes an automated, reproducible data engineering pipeline based on the international gold standard **LIDC-IDRI** dataset.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 LIDC-IDRI END-TO-END DATA ENGINEERING PIPELINE                  │
├─────────────────────────────────┬───────────────────────────────────────────────┤
│    RAW DICOM CT ACQUISITION     │         MULTI-READER XML ANNOTATIONS          │
│   - Multi-slice axial series    │   - 4 Board-certified thoracic radiologists   │
│   - 16-bit raw attenuation      │   - Blinded & unblinded reading sessions      │
│   - Patient coordinates (x,y,z) │   - Planar 2D polygon vertex coordinate lists │
└────────────────┬────────────────┴───────────────────────┬───────────────────────┘
                 │                                        │
                 ▼                                        ▼
┌─────────────────────────────────┐      ┌────────────────────────────────────────┐
│ PHYSICAL SORT & HU CALIBRATION  │      │     CONSENSUS MASK RASTERIZATION       │
│ - Normal vector dist projection │      │ - Coordinate matching via SOPUID / Z   │
│ - HU = Pixel * Slope + Intercept│      │ - 50% Majority voting consensus rule   │
└────────────────┬────────────────┘      └────────────────┬───────────────────────┘
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  MEDICAL PREPROCESSING & BALANCED SAMPLING                      │
│ - Pulmonary windowing: Center = -600 HU, Width = 1500 HU ([-1350, +150] HU)     │
│ - Value normalization to [0.0, 1.0]                                             │
│ - Dual-mode resizing to (256, 256): Bilinear (CT slice) + Nearest-Neighbor (Mask│
│ - Controlled slice sampling: Positive nodule slices + Margin (±1) + 20% Negatives│
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  ZERO-LEAKAGE PATIENT-LEVEL PARTITIONING                        │
│ - Train Set:      Patient LIDC-IDRI-0003 (140 slices, 13 annotated nodules)     │
│ - Validation Set: Patient LIDC-IDRI-0005 (133 slices, 9 annotated nodules)      │
│ - Blind Test Set: Patient LIDC-IDRI-0001 (133 slices, 4 annotated nodules)      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Cohort & Physical Imaging Specifications

### 2.1 LIDC-IDRI Reference Database
The **Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)** was initiated by the National Cancer Institute (NCI) and is publicly hosted on **The Cancer Imaging Archive (TCIA)**:
- **Total Consortium Repository**: 1,018 clinical cases (1,010 patients) collected across seven major medical institutions.
- **Scanner Models**: GE Medical Systems (LightSpeed, HiSpeed), Siemens (Sensation, Volume Zoom), Philips (Mx8000), Toshiba (Aquilion).
- **Physical Modality**: Helical Thoracic Computed Tomography (CT), single-energy scanning.
- **Slice Matrix Dimensions**: $512 \times 512$ pixels per axial cross-section.
- **Dynamic Bit Depth**: 16-bit signed integer values.

### 2.2 Active Inspection Cohort Statistics
Our local data engineering engine (`src/dataset_inspection.py`) scanned, parsed, and verified the physical files in our active directory:

| Metric Parameter | Measured Physical Value | Description / Relevance |
| :--- | :---: | :--- |
| **Total Patient Cases Analyzed** | **3** | `LIDC-IDRI-0001`, `LIDC-IDRI-0003`, `LIDC-IDRI-0005` |
| **Total CT Studies** | **3** | Verified independent diagnostic examinations |
| **Total Axial CT Slices Scanned** | **406** | Complete volume reconstructions |
| **Total XML Annotation Records** | **1,319** | Official LIDC XML files scanned and indexed |
| **Detected Nodules ($\ge 3\text{ vertices}$)** | **26** | Validated pathological nodule instances |
| **Total Radiologist ROI Markups** | **127** | Individual radiologist polygon boundaries |
| **In-Plane Pixel Spacing ($\Delta x, \Delta y$)** | **$0.664\text{ to }0.820\text{ mm}$** | High spatial in-plane resolution |
| **Slice Thickness ($\Delta z$)** | **$2.50\text{ mm}$** | Measured longitudinal slice pitch ($\pm 0.00\text{ mm}$) |

### 2.3 Patient-Level Breakdown Table
*Empirically extracted directly from DICOM headers and XML markups:*

| Patient Identifier | Series Count | Total Slices | Matched XML Files | Annotated Nodules | ROI Contours | Slice Thickness | Pixel Spacing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LIDC-IDRI-0001` | 1 | 133 | 2 | 4 | 22 | $2.50\text{ mm}$ | $0.703 \times 0.703\text{ mm}$ |
| `LIDC-IDRI-0003` | 1 | 140 | 2 | 13 | 64 | $2.50\text{ mm}$ | $0.820 \times 0.820\text{ mm}$ |
| `LIDC-IDRI-0005` | 1 | 133 | 2 | 9 | 41 | $2.50\text{ mm}$ | $0.664 \times 0.664\text{ mm}$ |
| **Total / Aggregate** | **3** | **406** | **6** | **26** | **127** | **$2.50\text{ mm}$** | **$0.664 - 0.820\text{ mm}$** |

---

## 3. CT Physics, Attenuation Principles & Pulmonary Windowing

### 3.1 The Hounsfield Unit (HU) Physical Scale
Computed Tomography measures the linear X-ray attenuation coefficient $\mu(x, y, z)$ through anatomical structures. Attenuation depends on electron density and atomic number. To standardize measurements across scanners, values are mapped to the Hounsfield Unit scale relative to distilled water at standard temperature and pressure:

$$\text{HU} = 1000 \times \frac{\mu_{\text{tissue}} - \mu_{\text{water}}}{\mu_{\text{water}}}$$

Key biological benchmarks:
- $\text{HU}_{\text{air}} = -1000\text{ HU}$ (minimal attenuation)
- $\text{HU}_{\text{lung parenchyma}} = -900\text{ to } -500\text{ HU}$ (air-filled alveoli with micro-capillaries)
- $\text{HU}_{\text{fat}} = -120\text{ to } -80\text{ HU}$
- $\text{HU}_{\text{water}} \equiv 0\text{ HU}$
- $\text{HU}_{\text{soft-tissue / pulmonary nodule}} = -100\text{ to } +100\text{ HU}$
- $\text{HU}_{\text{blood / vascular contrast}} = +30\text{ to } +60\text{ HU}$
- $\text{HU}_{\text{dense cortical bone}} = +400\text{ to } +1500\text{ HU}$

### 3.2 DICOM Rescaling Formula
Raw DICOM pixel intensity values are unsigned 16-bit integers ($0$ to $65,535$). To recover physical Hounsfield Units, the loader applies the linear transformation defined in the DICOM header:

$$\text{HU}(x, y) = p(x, y) \cdot S + I$$

where $S = \text{RescaleSlope}$ (typically $1.0$) and $I = \text{RescaleIntercept}$ (typically $-1024.0$).

### 3.3 Clinical Pulmonary Windowing Strategy
Human thoracic scans display a dynamic range exceeding $4000\text{ HU}$. If an image is normalized linearly across $[-1024, +3000]$, soft-tissue nodules span less than $5\%$ of the total numerical range, causing subtle lesion contours to be obscured.

We implement the standard clinical **Pulmonary Windowing** transformation:
- **Window Center ($C$)**: $-600\text{ HU}$ (centered on the density of healthy pulmonary parenchyma).
- **Window Width ($W$)**: $1500\text{ HU}$ (defining a window range of $1500\text{ HU}$ centered at $-600\text{ HU}$).

$$\text{HU}_{\min} = C - \frac{W}{2} = -600 - 750 = -1350\text{ HU}$$
$$\text{HU}_{\max} = C + \frac{W}{2} = -600 + 750 = +150\text{ HU}$$

The clipping and normalization function is strictly defined as:
$$I_{\text{windowed}}(x, y) = \frac{\text{clip}(\text{HU}(x, y), \, -1350, \, +150) - (-1350)}{1500} \in [0.0, 1.0]$$

**Clinical Rationale**:
- Values $<-1350\text{ HU}$ (external ambient air, scanner couch) are clipped to $0.0$.
- Values $>+150\text{ HU}$ (ribs, sternum, spine, calcified plaque) are clipped to $1.0$.
- The full $[0.0, 1.0]$ dynamic range is dedicated exclusively to aerated lung parenchyma, bronchial branches, vascular bifurcations, and pulmonary nodules.

---

## 4. Multi-Reader Annotation Protocol & Consensus Formulation

### 4.1 The Two-Phase Radiologist Reading Protocol
The LIDC-IDRI consortium utilized an unblinded consensus protocol to delineate ground truth:
1. **Blinded Read Phase**: Four board-certified thoracic radiologists independently examined the CT series without consulting each other. Each radiologist marked suspicious findings into three categories:
   - *Category 1 (Nodule $\ge 3\text{ mm}$)*: Marked with full 3D boundary polygon contours.
   - *Category 2 (Nodule $< 3\text{ mm}$)*: Marked with approximate 3D centroid point coordinates.
   - *Category 3 (Non-nodule $\ge 3\text{ mm}$)*: Marked with a single coordinate (e.g., apical scarring, granuloma).
2. **Unblinded Read Phase**: Each radiologist reviewed their own marks alongside the anonymized marks of the other three radiologists, allowing them to add, edit, or delete contours.

### 4.2 Polygon Coordinate Extraction
In the XML markup, each 2D slice contour is defined by an ordered list of planar vertex coordinates:
$$\mathcal{V}_m = \{(x_1, y_1), (x_2, y_2), \dots, (x_K, y_K)\}, \quad K \ge 3$$
associated with an explicit `imageSOP_UID` and spatial $z$-position.

### 4.3 50% Majority Voting Consensus Aggregation
Because radiologists exhibit natural inter-observer variability along ambiguous borders, our pipeline aggregates multiple reader contours using a **50% Majority Voting Consensus Rule**:

1. For each radiologist $m \in \{1, \dots, M\}$ marking a nodule on slice $s$, the polygon $\mathcal{V}_m$ is rasterized into a planar binary mask:
   $$B_m(x, y) \in \{0, 1\}, \quad \forall (x, y) \in [0, H-1] \times [0, W-1]$$
2. The consensus mask $M_{\text{consensus}}(x, y)$ is computed as:
   $$M_{\text{consensus}}(x, y) = \begin{cases} 1 & \text{if } \frac{1}{M}\sum_{m=1}^M B_m(x, y) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Scientific Justification**:
- Union aggregation ($\ge 1$ reader) incorporates subjective over-contouring and peripheral vascular noise.
- Intersection aggregation ($100\%$ agreement) discards valid spiculated and lobulated margins where readers disagree slightly.
- **50% consensus** achieves the optimal balance: every foreground pixel is corroborated by at least half of the contributing experts, suppressing reader idiosyncrasies while preserving true pathological margins.

---

## 5. Class Imbalance, Slice Sampling & Zero-Leakage Splitting

### 5.1 Extreme Class Imbalance Quantification
In a standard $512 \times 512$ axial slice ($262,144$ pixels), a solitary nodule ($5\text{ to }15\text{ mm}$ diameter) occupies approximately $150$ to $800$ pixels.
$$\text{Foreground Ratio} = \frac{\text{Nodule Pixels}}{\text{Total Slice Pixels}} = \frac{400}{262,144} \approx 0.0015 \quad (0.15\%)$$
Across an entire 3D patient volume of 140 slices, only 5 to 15 slices contain nodules. Foreground voxels account for less than **$0.01\%$** of the total volume.

### 5.2 Controlled Sampling Strategy
Training on raw uncurated volumes causes model collapse toward predicting all-zero masks, as predicting background yields $>99.9\%$ pixel accuracy. To establish a robust, representative training distribution:
1. **Positive Nodule Slices**: All slices containing verified consensus nodules are preserved.
2. **Neighboring Margin Slices**: Slices immediately above and below ($\pm 1$ slice in $z$) are included to capture nodule entry and exit transition boundaries.
3. **Controlled Negative Control Slices**: A fixed $20\%$ ratio (`negative_sample_ratio = 0.20`) of normal lung slices is sampled. This forces the network to learn complete background suppression without false positives.

```
Total Preprocessed Active Slices: 58 Samples
├── Positive Nodule Slices: 42 (72.41%)
└── Negative Control Slices: 16 (27.59%)
```

### 5.3 Zero-Leakage Patient-Level Partitioning
In medical imaging, adjacent CT slices ($1.25\text{ to }2.5\text{ mm}$ spacing) share nearly identical anatomical morphology, chest wall curvature, and noise textures. Splitting data randomly at the slice level causes severe **data leakage**: slice $z$ from a patient appears in the train set while slice $z+1$ from the same patient appears in the test set.

We strictly enforce **Patient-Level Partitioning**:

| Partition | Patient Identifier | Slices | Annotated Nodules | Role in Experimental Benchmark |
| :--- | :---: | :---: | :---: | :--- |
| **Train Set** | `LIDC-IDRI-0003` | 140 | 13 | Primary backpropagation & weight optimization |
| **Validation Set** | `LIDC-IDRI-0005` | 133 | 9 | Hyperparameter validation & early stopping trigger |
| **Test Set** | `LIDC-IDRI-0001` | 133 | 4 | Isolated blind evaluation of final models |

**Result**: Zero data leakage. The test patient is completely unseen during training, ensuring authentic clinical generalization measurement.

---

## 6. Quality Control (QC) & Data Integrity Telemetry

An automated quality control engine (`src/quality_control.py`) audited every step of the physical data pipeline. The empirical telemetry confirms 100% compliance:

| Verification Item | Audit Status | Measured Metric | Significance |
| :--- | :---: | :--- | :--- |
| **DICOM Slices Scanned** | PASS | 406 slices parsed | Valid header tags, no missing pixel buffers |
| **Physical Z-Sorting** | PASS | 100% monotonic sorting | Correct anatomical stacking from apex to base |
| **XML Parser Integrity** | PASS | 1,319 files scanned, 0 malformed | 100% syntactical XML integrity |
| **Image-Mask Dimension Alignment**| PASS | 58 paired `.npz` files verified | Zero dimensional mismatches ($256 \times 256$) |
| **Binary Mask Compliance** | PASS | $\forall p \in \{0, 1\}$ | Zero grayscale artifacts or floating boundary values |
| **Numerical Integrity** | PASS | 0 NaN, 0 Inf values | Stable gradients during model backpropagation |
| **Silently Discarded Cases** | PASS | **0 cases dropped** | Complete transparency and traceability |

---

## 7. Preprocessed Sample Traceability & Data Schema

Each processed slice pair is stored as a compressed `.npz` archive in `data/processed/` with the following structure:

```python
{
    "image": ndarray(shape=(256, 256), dtype=float32),  # Windowed & normalized [0.0, 1.0]
    "mask":  ndarray(shape=(256, 256), dtype=uint8),    # Discrete binary consensus mask {0, 1}
    "metadata": {
        "patient_id": str,                              # e.g., "LIDC-IDRI-0001"
        "slice_idx": int,                               # Axial slice index along z-axis
        "sop_instance_uid": str,                        # Unique DICOM SOP UID
        "z_position": float,                            # Physical table coordinate in mm
        "num_readers": int,                             # Number of contributing radiologists (1-4)
        "has_nodule": bool                              # True if foreground pixels > 0
    }
}
```

This strict schema guarantees end-to-end reproducibility, allowing any test prediction to be traced directly back to its raw DICOM slice and expert radiologist XML markup.

---

## 8. Phase 1 Empirical Visual Artifacts & Pipeline Runner

To verify and visualize the data engineering pipeline independently for the Mid-Semester Review, execute:
```powershell
.\.venv\Scripts\python.exe -u pipeline_phase1.py
```

This generates four high-resolution visual telemetry artifacts stored in `results/plots/phase1/`:
1. **Annotation Contour Verification (`1_annotation_contour_verification.png`)**: 4-panel figure showing pulmonary windowed CT, 50% majority consensus binary mask, boundary contour overlay, and magnified lesion ROI.
2. **Pulmonary Windowing Attenuation Calibration (`2_pulmonary_windowing_comparison.png`)**: Comparative panels and quantitative histograms of raw dynamic range ($[-1024, +3000]\text{ HU}$) vs windowed range ($[-1350, +150]\text{ HU}$) mapped to $[0, 1]$.
3. **Multi-Reader Consensus Breakdown (`3_multi_reader_consensus_breakdown.png`)**: Individual radiologist polygon markups side-by-side with the 50% majority consensus mask.
4. **Cohort Data Distribution Dashboard (`4_cohort_data_distribution.png`)**: Telemetry charts detailing slices per patient, nodule counts, positive/negative slice composition ($72.4\%$ vs $27.6\%$), and patient-level zero-leakage splits.

