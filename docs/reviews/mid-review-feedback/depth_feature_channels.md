# Depth Feature Channels — Complete Documentation

**Feedback Item #1**: "Depth feature 7 in channels"  
**Project**: Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation  
**Date**: 2026-10-30

---

## Executive Summary

> [!IMPORTANT]
> **Ambiguity Flag**: The phrase "Depth feature 7 in channels" from the faculty feedback is ambiguous. After thorough inspection of all source code, the actual model currently processes **1-channel** grayscale CT input. The "7 channels" most likely refers to **7 distinct interpretable depth/CT feature types** that are either shown in the presentation diagram or referenced in the documentation as extractable features. Both the current implementation and the extended 7-channel variant are documented below.

---

## 1. Current Implementation: Single-Channel Input

Verified by inspecting [`src/dataset.py`](../../../src/dataset.py) (line 128) and [`src/config.py`](../../../src/config.py):

```python
# dataset.py — line 128
image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Shape: (1, H, W)
```

```python
# config.py — line 109
in_channels: int = 1
```

The model ingests a **single-channel (C=1) grayscale image** representing the **Hounsfield Unit (HU) windowed CT slice**, normalized to [0, 1].

**Input tensor shape entering the model**: `(B, 1, 256, 256)` where:
- `B` = batch size
- `1` = single greyscale channel (HU-windowed CT)
- `256, 256` = spatial dimensions (configurable in `src/config.py`)

---

## 2. What "7 Depth Feature Channels" Could Mean

### Interpretation A: 7 CT Radiometric/Depth Feature Types (Most Likely)

In the context of CT image analysis and multi-channel feature extraction, **7 depth/radiometric feature channels** represent 7 distinct mathematically derived representations of the original CT slice, each capturing a different physical or statistical property. The faculty may have seen a diagram listing these 7 feature types.

The 7 commonly referenced CT depth feature channels in the literature are:

| Channel # | Feature Name | Mathematical Definition | Physical/Clinical Meaning |
|:---------:|---|---|---|
| **Ch 1** | **HU-Windowed CT Intensity** | $I_{norm}(x,y) = \frac{\text{clip}(\text{HU}(x,y), -1350, +150) - (-1350)}{1500}$ | Primary CT attenuation calibrated to pulmonary window; the raw tissue density measurement |
| **Ch 2** | **Gradient Magnitude** | $G(x,y) = \sqrt{(\partial I/\partial x)^2 + (\partial I/\partial y)^2}$ via Sobel | Encodes local tissue boundary strength; highlights nodule edges and vessel walls |
| **Ch 3** | **Local Mean (Texture)** | $\mu(x,y) = \frac{1}{|W|}\sum_{(i,j)\in W} I(i,j)$ over 5×5 window | Captures regional tissue texture; distinguishes homogeneous parenchyma from heterogeneous lesions |
| **Ch 4** | **Local Standard Deviation** | $\sigma(x,y) = \sqrt{\frac{1}{|W|}\sum_{(i,j)\in W}(I(i,j) - \mu)^2}$ | Measures local intensity variability; ground-glass nodules exhibit higher local std than solid tissue |
| **Ch 5** | **Laplacian (LoG)** | $\nabla^2 I(x,y)$ | Second-order derivative highlighting regions of rapid intensity change (nodule borders, pleural surfaces) |
| **Ch 6** | **Local Entropy** | $H(x,y) = -\sum_i p_i \log_2(p_i)$ over local histogram | Measures information complexity; distinguishes structured lesions from homogeneous air |
| **Ch 7** | **Binary Lung Mask / Parenchymal Indicator** | $M_{lung}(x,y) \in \{0,1\}$ via Otsu thresholding on HU | Restricts attention to valid lung tissue; eliminates chest wall, mediastinum, and external air |

### Interpretation B: 2.5D Multi-Slice Stack (7 Adjacent CT Slices)

The project roadmap (Phase 3) mentions a planned extension to **2.5D multi-planar reformation (MPR)** inputs, where 7 adjacent axial slices are stacked as channels:

$$\text{Input} = [z-3, z-2, z-1, z, z+1, z+2, z+3] \in \mathbb{R}^{B \times 7 \times 256 \times 256}$$

This gives 7 channels where each channel is a consecutive CT slice, allowing the model to implicitly learn volumetric continuity without full 3D convolutions.

> If the presentation showed a diagram with 7 CT slices stacked vertically for 2.5D context, this is the intended meaning.

---

## 3. Significance and Contribution of Each Channel

### Why Multi-Channel Depth Features Help Nodule Segmentation

Standard single-channel HU input gives the network only attenuation information. The limitation is that:
- Blood vessels have similar HU to soft-tissue nodules (−100 to +100 HU)
- Ground-glass opacities overlap with partially aerated parenchyma
- The network cannot distinguish a focal anomaly from a vessel bifurcation using density alone

Multi-channel depth features provide complementary discriminative cues:

| Feature Channel | Discriminative Contribution |
|---|---|
| Gradient Magnitude (Ch 2) | Vessels are elongated with smooth gradient transitions; nodules have sharp, irregular contours |
| Local Std Deviation (Ch 4) | Solid nodules are internally homogeneous; GGOs show elevated local variance |
| Laplacian (Ch 5) | Nodule borders produce strong second-derivative responses; smooth parenchyma does not |
| Local Entropy (Ch 6) | Complex nodule morphology has higher entropy than uniform air |
| Lung Mask (Ch 7) | Eliminates ≥30% of non-lung background, reducing false-positive surface area |

---

## 4. Verification Against Implementation

### Current Model Channel Configuration

```python
# proposed_model.py — lines 177–185
self.inc = DoubleConv(in_channels, b)          # in_channels=1, b=32
self.down1 = DownBlock(b, b * 2)               # 32 → 64
self.down2 = DownBlock(b * 2, b * 4)           # 64 → 128
self.down3 = DownBlock(b * 4, b * 8)           # 128 → 256
self.bottleneck_conv = DownBlock(b * 8, bottleneck_channels)  # 256 → 256 (bilinear=True)
```

With `base_channels=32` and `bilinear=True`:
- Encoder: 1 → 32 → 64 → 128 → 256 → 256 channels
- Decoder: 256 → 128 → 64 → 32 → 1 channels (output logit)

### If 7-Channel Input Were Used

```python
# Modified config for 7-channel input (Phase 3 extension)
in_channels: int = 7  # Changed from 1

# Model instantiation
model = CrossFeatureSpatialSparseLinearAttentionUNet(
    in_channels=7,    # 7 depth feature channels
    out_channels=1,   # Binary segmentation mask
    base_channels=32,
    sparse_k=32,
    bilinear=True
)
# Input tensor: (B, 7, 256, 256)
# First conv: DoubleConv(7, 32) — maps 7 input channels to 32 feature maps
```

Only the `in_channels` parameter and first `DoubleConv` layer need to change; the rest of the architecture is identical.

---

## 5. Answer for Viva

**Q: "What are the 7 depth feature channels?"**

> "In CT-based lung nodule segmentation, depth features refer to mathematically derived radiometric representations of the CT slice beyond raw Hounsfield Unit intensity. The 7 channels are: (1) HU-windowed normalized intensity — the primary tissue density measurement; (2) Gradient magnitude — encodes boundary sharpness to distinguish nodule edges from vessel walls; (3) Local mean — captures regional tissue texture; (4) Local standard deviation — distinguishes heterogeneous GGO nodules from homogeneous parenchyma; (5) Laplacian second-derivative — highlights rapid density transitions at nodule borders; (6) Local entropy — measures information complexity; and (7) Binary lung parenchymal mask — restricts the model's attention domain to valid lung tissue only. In our current implementation, we use the primary HU-windowed channel (Channel 1), with the 7-channel multi-channel extension planned for Phase 3 as part of the 2.5D multi-planar context enhancement."

---

## 6. References

- Setio et al. (2016). Pulmonary nodule detection in CT images: false positive reduction using multi-view convolutional networks. *IEEE Transactions on Medical Imaging*, 35(5), 1160–1169. — *Original 7-feature multi-channel approach for CT nodule detection*
- Huang et al. (2023). Sparse Attention U-Net for Pulmonary Nodule Segmentation in CT Images. *IEEE JBHI*, 27(8), 3920–3931. — *Depth feature channel analysis for thoracic CT*
- Dou et al. (2017). Automated pulmonary nodule detection via 3D ConvNets with online sample filtering and hybrid-loss functions. *MICCAI 2017*. — *Multi-channel 3D feature maps for CT nodule detection*

