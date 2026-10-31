# U-Net Decoder & Upsampling Methods — Deep Dive

**Feedback Items #3 & #5b**: "U-Net Decoder, (read about) / Upsampling → Methods / What is the method you are using here."  
**Project**: Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation  
**Date**: 2026-10-30

---

## 1. The U-Net Decoder — Complete Explanation

### 1.1 What is the U-Net Decoder?

The U-Net decoder is the **expanding path** of the U-shaped architecture, introduced by Ronneberger et al. (MICCAI 2015). Its purpose is to progressively **restore spatial resolution** from the coarse bottleneck representation back to the full input resolution, producing a pixel-wise segmentation map.

```
Bottleneck: (B, 256, 16, 16)  ← Low resolution, high semantic content
    ↓  Upsample × 4 decoder stages
Output:    (B,   1, 256, 256) ← Full resolution segmentation logit
```

### 1.2 Core Decoder Operations

Each decoder stage performs three operations:

1. **Upsampling** — doubles spatial resolution (H, W → 2H, 2W)
2. **Skip Connection Concatenation** — concatenates the upsampled feature map with the encoder's corresponding skip feature
3. **Double Convolution** — applies two 3×3 conv-BN-ReLU layers to fuse the concatenated features

$$X^l_{\text{out}} = \text{DoubleConv}\left([\text{Skip}^l \; ; \; \text{UpSample}(X^{l+1}_{\text{decoder}})]\right)$$

---

## 2. Upsampling Methods — Complete Comparison

### 2.1 Method 1: Bilinear Interpolation (Our Implementation)

**What it is**: A deterministic, parameter-free spatial interpolation that estimates missing pixel values using weighted averages of the four nearest neighbours.

**How it works**: For a target pixel at fractional source coordinate $(x_s, y_s)$:
$$I_{out}(x, y) = (1-\alpha)(1-\beta) \cdot I_{11} + \alpha(1-\beta) \cdot I_{21} + (1-\alpha)\beta \cdot I_{12} + \alpha\beta \cdot I_{22}$$
where $(I_{11}, I_{12}, I_{21}, I_{22})$ are the four surrounding pixels and $\alpha, \beta$ are fractional offsets.

**In our code** (verified in `unet.py` line 57 and `proposed_model.py` line 129):
```python
self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
```

**Followed by**: `DoubleConv(in_channels, out_channels, in_channels // 2)` — a mid-channel reduction is applied in the DoubleConv when using bilinear upsampling (halves channels before the second conv), reducing parameter count.

### 2.2 Method 2: Transposed Convolution (Available as Alternative)

**What it is**: A learnable upsampling layer that performs the reverse of strided convolution. Also called "fractionally strided convolution" or "deconvolution" (though technically not a true deconvolution).

**How it works**: The transposed convolution kernel slides over the output grid, placing learnable weights scaled by each input value. The output stride acts as the inverse of downsampling.

**In our code** (activated when `bilinear=False`):
```python
self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
```

### 2.3 Method 3: Pixel Shuffle (Sub-Pixel Convolution)

**What it is**: A technique that rearranges elements from a high-channel low-resolution tensor into a lower-channel high-resolution tensor. Used in ESPCN (Shi et al., CVPR 2016).  
**Not used in our project.**

### 2.4 Method 4: Nearest-Neighbour Interpolation

**What it is**: Copies the nearest source pixel to the target position. Simple but produces blocky artefacts.  
**Not used for upsampling in our model.** (Used only for mask resizing in preprocessing to preserve binary values.)

---

## 3. Why We Use Bilinear Interpolation (Justification)

| Criterion | Bilinear Interpolation | Transposed Convolution |
|---|---|---|
| **Parameters** | 0 (no learnable parameters) | $C_{in} \times C_{out} \times k^2$ parameters |
| **Checkerboard Artefacts** | None | Common if not carefully initialized |
| **Gradient Flow** | Smooth | Can be erratic early in training |
| **Performance** | Empirically competitive | Slightly higher capacity |
| **Memory** | Lower | Higher |
| **Standard in Medical Imaging** | Yes (Ronneberger's original U-Net) | Used in later variants |

**Justification**: Bilinear interpolation is the upsampling method recommended in the original U-Net paper (Ronneberger et al., MICCAI 2015) and its Attention U-Net extension (Oktay et al., MIDL 2018). For medical image segmentation with small training datasets (such as ours with 58 sample pairs), avoiding additional learnable parameters in the upsampling layer reduces overfitting risk. The subsequent `DoubleConv` layers provide sufficient learned capacity for spatial feature refinement after upsampling.

> [!TIP]
> **Answer for viva**: "We use bilinear interpolation (`nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)`) because it introduces zero additional parameters, produces smooth upsampled feature maps without checkerboard artefacts, and is consistent with the original U-Net design. The following DoubleConv layers with BatchNorm provide all the necessary learnable spatial refinement capacity."

---

## 4. Standard U-Net Decoder Stage-by-Stage

With `base_channels=32` and `bilinear=True`, the full decoder path:

| Decoder Stage | Input Shape | Upsample Output | Skip Shape | Concat Shape | DoubleConv Output |
|:---:|---|---|---|---|---|
| `up1` | (B, 256, 16, 16) | (B, 256, 32, 32) | (B, 256, 32, 32) | (B, 512, 32, 32) | (B, 128, 32, 32) |
| `up2` | (B, 128, 32, 32) | (B, 128, 64, 64) | (B, 128, 64, 64) | (B, 256, 64, 64) | (B, 64, 64, 64) |
| `up3` | (B, 64, 64, 64) | (B, 64, 128, 128) | (B, 64, 128, 128) | (B, 128, 128, 128) | (B, 32, 128, 128) |
| `up4` | (B, 32, 128, 128) | (B, 32, 256, 256) | (B, 32, 256, 256) | (B, 64, 256, 256) | (B, 32, 256, 256) |
| `outc` | (B, 32, 256, 256) | — | — | — | (B, 1, 256, 256) |

> [!NOTE]
> In bilinear mode, the `DoubleConv` uses `mid_channels = in_channels // 2` for the first conv, halving channels before the second conv. This is why the output channels are halved at each stage even though the concatenated input has the same number of channels from both skip and upsampled features.

---

## 5. Proposed Model Decoder (AttentionDecoderBlock)

The proposed model replaces standard `UpBlock` with `AttentionDecoderBlock`, which adds **spatial attention filtering of the skip connection** before concatenation:

```python
# proposed_model.py — AttentionDecoderBlock.forward()
def forward(self, x1, x2):        # x1: decoder, x2: skip
    x1 = self.up(x1)              # Bilinear upsample x2
    # Pad if needed (handles odd spatial dimensions)
    x2_filtered, attn_map = self.skip_attn(x2)   # ATTENTION on skip
    x = torch.cat([x2_filtered, x1], dim=1)      # Concatenate
    out = self.conv(x)                            # DoubleConv
    return out, attn_map
```

**What changes**: Before concatenating the skip connection features with the upsampled decoder features, `SpatialAttention` is applied to the skip. This suppresses non-informative parenchymal features in the skip connection, preventing them from polluting the decoder path.

**Why this matters**: In standard U-Net, all encoder features are blindly copied via skips — including noise from normal bronchial trees and vessels. The attention-gated skip ensures only salient nodule-relevant features pass through.

---

## 6. Viva Q&A

**Q: "What upsampling method do you use in the decoder?"**
> "We use bilinear interpolation via `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)`. It doubles spatial resolution without introducing learnable parameters, avoiding checkerboard artefacts that are common with transposed convolutions. After upsampling, a DoubleConv block with BatchNorm and ReLU handles learned feature refinement."

**Q: "What is the difference between bilinear interpolation and transposed convolution?"**
> "Bilinear interpolation is a fixed mathematical operation that estimates missing values from the four nearest neighbours — it has no trainable parameters and is deterministic. Transposed convolution is learnable — it has a trainable kernel matrix and can adapt to the data, but requires more parameters and can produce checkerboard artefacts if not carefully initialized. For our small medical dataset, bilinear upsampling is preferred to reduce overfitting risk."

**Q: "How does the decoder know to produce the correct segmentation?"**
> "The decoder receives two inputs at each stage: (1) the upsampled low-resolution but semantically rich features from the previous decoder stage, and (2) the high-resolution but lower-semantic skip features from the corresponding encoder stage. These are concatenated along the channel dimension and processed by a DoubleConv block, which fuses coarse context with fine spatial detail. In our proposed model, the skip features are additionally filtered by spatial attention to suppress non-nodular parenchymal noise before concatenation."

**Q: "What is `align_corners=True` in the Upsample call?"**
> "This parameter determines how grid coordinates are mapped during interpolation. When `align_corners=True`, the corner pixels of the input and output grids are aligned — this avoids fractional-pixel misalignment that can cause subtle spatial registration errors in pixel-precise segmentation tasks."

