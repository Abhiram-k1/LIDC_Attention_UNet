# Cross-Feature Spatial Sparse Linear Attention U-Net for Lung Nodule Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: LIDC-IDRI](https://img.shields.io/badge/Dataset-LIDC--IDRI-blue.svg)](https://www.cancerimagingarchive.net/collection/lidc-idri/)

An end-to-end, research-grade Computer Vision and Deep Learning framework for lung nodule segmentation from chest Computed Tomography (CT) scans using the official **LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative) dataset.

> **Medical / Research Disclaimer**:  
> This project is strictly an experimental computer-aided research system for lung nodule segmentation and methodological investigation in Computer Vision. It does not provide clinical diagnosis or clinical validation and must not be used for medical decision-making.

---

## 1. Research Objectives & Motivation

Standard U-Net models for CT segmentation suffer from two major limitations:
1. **Quadratic Complexity of Dense Self-Attention**: Standard self-attention scales as $\mathcal{O}(N^2)$ where $N = H \times W$, making high-resolution feature interaction computationally prohibitive.
2. **Context Disconnect across Feature Scales**: Standard skip connections merely concatenate shallow spatial features with deep semantic features without interactive gating, causing false positive predictions on complex parenchyma and chest wall structures.

This project introduces and evaluates the **Cross-Feature Spatial Sparse Linear Attention U-Net (CF-SSLA U-Net)** to address these challenges:
- **Cross-Feature Interaction**: Proactively aligns low-, mid-, and high-level encoder feature grids to perform inter-scale semantic gating.
- **Spatial Attention**: Focuses feature representation on nodular lesion boundaries while attenuating homogeneous air parenchyma.
- **Sparse Top-$k$ Selection**: Restricts attention computation to the top-$k$ most salient spatial tokens ($\mathcal{O}(N \cdot k)$), filtering out irrelevant non-nodular voxels.
- **Linear Attention Factorization**: Utilizes non-negative kernel feature map decomposition $(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^TV)$ to achieve $\mathcal{O}(N)$ computational complexity.

---

## 2. Mathematical Formulations

### A. Linear Attention Factorization
Standard attention computes:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
We replace the softmax kernel with the non-negative feature map $\phi(x) = \text{ELU}(x) + 1$:
$$\text{Sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j)$$
Applying the associative property of matrix multiplication:
$$O_i = \frac{\phi(q_i)^T \sum_{j=1}^N \phi(k_j) v_j^T}{\phi(q_i)^T \sum_{j=1}^N \phi(k_j) + \epsilon}$$
By computing $\sum_{j=1}^N \phi(k_j) v_j^T \in \mathbb{R}^{d \times d}$ once, complexity drops from $\mathcal{O}(N^2 \cdot d)$ to $\mathcal{O}(N \cdot d^2)$.

### B. Top-$k$ Sparse Spatial Routing
Nodules represent compact localized anomalies. A lightweight gating network predicts routing scores:
$$S = \sigma(W_r X) \in \mathbb{R}^{N}$$
Indices $\mathcal{I}_k = \text{Top-}k(S)$ are gathered, reducing key/value sequence length from $N$ to $k \ll N$, yielding an effective complexity of $\mathcal{O}(N \cdot k \cdot d)$.

### C. Multi-Scale Cross-Feature Fusion
Encoder representations $F_{low}, F_{mid}, F_{high}$ are projected to a unified dimension $D$ and spatially aligned to the bottleneck grid via adaptive pooling:
$$F_{stacked} = [\mathcal{P}(F_{low}), \mathcal{P}(F_{mid}), \mathcal{P}(F_{high})] \in \mathbb{R}^{B \times 3D \times H_{ref} \times W_{ref}}$$
$$F_{fused} = \text{Conv}_{3\times 3}(F_{stacked}) \odot \sigma(\text{MLP}(\text{GAP}(F_{stacked})))$$

---

## 3. Dataset & Preprocessing Pipeline

- **Source**: Authentic LIDC-IDRI CT scans (DICOM) and official XML radiologist markups from The Cancer Imaging Archive (TCIA).
- **Physical DICOM Sorting**: Slices are sorted physically using `ImagePositionPatient` along the slice normal derived from `ImageOrientationPatient` (never by filename).
- **Hounsfield Unit (HU) Calibration**:
  $$\text{HU} = \text{PixelArray} \times \text{RescaleSlope} + \text{RescaleIntercept}$$
- **Lung Windowing**: Standard pulmonary window with Center $C = -600\text{ HU}$ and Width $W = 1500\text{ HU}$ (range: $[-1350\text{ HU}, 150\text{ HU}]$), normalized to $[0, 1]$.
- **Contour Rasterization**: XML polygon coordinates (`xCoord`, `yCoord`) are matched to DICOM slices by `imageSOP_UID` / `imageZposition` and rasterized.
- **Reader Consensus**: Supports 50% consensus ($\ge 50\%$ reader agreement), union, and majority vote.
- **Patient-Level Split**: Strict partitioning ensuring zero patient overlap across Train, Validation, and Test sets.

---

## 4. Project Directory Structure

```
LIDC_Attention_UNet/
│
├── data/
│   ├── raw/                 # Downloaded DICOM series & XML markups
│   ├── processed/           # Preprocessed .npz slice pairs (image + mask)
│   └── splits/              # patient_splits.json
│
├── src/
│   ├── config.py            # Central hyperparameter & path definitions
│   ├── tcia_downloader.py   # Automated TCIA REST API data fetcher
│   ├── dataset_inspection.py# Empirical DICOM/XML inspection & metadata report
│   ├── dicom_loader.py      # Spatial slice sorting & HU conversion
│   ├── xml_parser.py        # LIDC-IDRI XML contour extraction
│   ├── mask_generator.py    # Multi-radiologist polygon rasterization
│   ├── preprocessing.py     # Windowing, normalization, slice selection
│   ├── dataset.py           # PyTorch Dataset, augmentations, patient splitter
│   ├── losses.py            # BCE, Soft Dice, Focal, Tversky losses
│   ├── metrics.py           # Dice, IoU, Precision, Recall, Specificity, F1
│   ├── unet.py              # Standard Baseline U-Net
│   ├── spatial_attention.py # Channel-pooled spatial attention module
│   ├── sparse_attention.py  # Top-k spatial routing sparse attention
│   ├── linear_attention.py  # Factorized O(N) linear attention
│   ├── cross_feature.py     # Dimension-aligned multi-scale encoder fusion
│   ├── proposed_model.py    # Full CF-SSLA U-Net model
│   ├── train.py             # Training engine with early stopping
│   ├── evaluate.py          # Independent test evaluation & 95% CIs
│   ├── ablation.py          # 6-Model systematic ablation study
│   └── visualize.py         # 5-column comparison & attention heatmaps
│
├── notebooks/
│   └── LIDC_Attention_UNet_Colab_Kaggle.ipynb # Cloud GPU notebook
│
├── checkpoints/             # Best and final model weights (.pth)
│
├── results/
│   ├── metrics/             # Evaluation JSONs & ablation tables
│   ├── plots/               # 5-column qualitative comparison figures
│   └── attention_maps/      # Learned attention saliency visualizations
│
├── requirements.txt
├── run_pipeline.py          # End-to-end execution script
└── README.md
```

---

## 5. Quickstart & Execution

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/Abhiram-k1/LIDC_Attention_UNet.git
cd LIDC_Attention_UNet

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full 26-step research pipeline
python run_pipeline.py
```

### Running the 6-Model Ablation Study
```bash
python -c "
from src.dataset import build_dataloaders, PatientSplitter
from src.config import cfg
from src.ablation import run_ablation_study

splits = PatientSplitter.load_split(cfg.paths.splits_dir / 'patient_splits.json')
tr, val, te = build_dataloaders(cfg.paths.processed_dir, splits, batch_size=cfg.training.batch_size)
run_ablation_study(tr, val, te, epochs=15)
"
```

---

## 6. Research Questions & Experimental Verification

- **RQ1**: Can the proposed model segment lung nodules from LIDC-IDRI CT scans?  
  *Verified via empirical test set evaluation with Dice, IoU, and boundary contours.*
- **RQ2**: Does the proposed model outperform standard U-Net?  
  *Investigated via direct head-to-head comparison on the identical patient-isolated test set.*
- **RQ3**: Does cross-feature interaction improve segmentation?  
  *Evaluated in Model E ablation variant.*
- **RQ4**: Does spatial attention improve localization?  
  *Evaluated in Model B ablation variant and visualized via attention heatmaps.*
- **RQ5**: Does sparse attention reduce computational cost?  
  *Benchmarked in Model D measuring latency and FLOPs.*
- **RQ6**: Does linear attention provide an efficient alternative to dense attention?  
  *Benchmarked in Model C verifying linear scaling $\mathcal{O}(N)$ vs quadratic $\mathcal{O}(N^2)$.*
- **RQ7**: Which proposed component contributes most to performance?  
  *Synthesized in the final ablation study table.*
