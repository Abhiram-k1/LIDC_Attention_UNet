# Technical Terminology Glossary

**Feedback Item #2**: "U should know particular words when you explain like local features"  
**Project**: Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation  
**Date**: 2026-10-30

> [!NOTE]
> This glossary defines every domain-specific technical term used in this project with precise, academically correct definitions tied to the actual implementation. Use this as a viva preparation reference.

---

## A. Core CT Imaging Terms

### Hounsfield Unit (HU)
**Definition**: A dimensionless quantitative scale of radiodensity used in CT imaging, calibrated relative to water. Defined as:
$$\text{HU} = 1000 \times \frac{\mu_{\text{tissue}} - \mu_{\text{water}}}{\mu_{\text{water}}}$$
where $\mu$ is the linear X-ray attenuation coefficient.  
**In our project**: Raw DICOM pixel values are converted to HU using: $\text{HU}(x,y) = p(x,y) \cdot S + I$ where $S=1.0$ (RescaleSlope) and $I=-1024.0$ (RescaleIntercept). See [`src/dicom_loader.py`](../../../src/dicom_loader.py).

### Pulmonary Windowing (Window Leveling)
**Definition**: A contrast enhancement technique that clips the HU display range to a clinically relevant interval, suppressing irrelevant tissues and expanding focal pathology contrast.  
**In our project**: Window Center $C = -600$ HU, Window Width $W = 1500$ HU → Range: $[-1350, +150]$ HU → mapped to $[0, 1]$.

### Solitary Pulmonary Nodule (SPN)
**Definition**: A rounded lesion ≤30 mm in diameter, entirely surrounded by lung parenchyma, without associated atelectasis, hilar enlargement, or pleural effusion.

### Ground-Glass Opacity (GGO)
**Definition**: A hazy increase in lung attenuation that does not obscure underlying bronchial structures or pulmonary vessels. HU typically ranges −700 to −100. More difficult to segment than solid nodules.

### Juxta-vascular / Juxta-pleural Nodule
**Definition**: A nodule attached to a blood vessel or the pleural wall. Creates boundary ambiguity because the vessel/pleura and nodule share similar HU values.

---

## B. Deep Learning & Architecture Terms

### Local Features
**Definition**: Feature representations computed within a spatially restricted region (receptive field) of the input image. In CNNs, a 3×3 convolutional kernel computes local features by aggregating information only from its immediate 3×3 pixel neighborhood.  
**Technical precision**: "Local features" refers to low-level representations such as edges, textures, and local gradients extracted by early convolutional layers. They have **high spatial resolution** but **low semantic content**.  
**In our project**: The encoder's first two stages (`inc` → `down1`) produce local features: edges, boundary gradients, and pulmonary texture patterns. These are preserved via skip connections and passed to the decoder.

### Global Features (vs. Local Features)
**Definition**: Feature representations capturing long-range spatial dependencies across the entire feature map. Attention mechanisms compute global features by allowing each spatial location to attend to all other locations.  
**In our project**: The bottleneck's Linear Attention block provides global context by allowing every spatial token to aggregate information from the top-k most informative tokens across the entire H×W feature grid.

### Receptive Field
**Definition**: The region of the input image that influences a particular neuron's activation. A single 3×3 conv has a 3×3 receptive field; stacked convolutions and max-pooling increase the effective receptive field.  
**In our project**: After 4 downsampling stages (each halving spatial resolution), the bottleneck neurons have an effective receptive field covering the full 256×256 input slice.

### Skip Connection
**Definition**: A direct pathway that copies feature maps from an encoder stage to the symmetrically corresponding decoder stage, bypassing the bottleneck. Defined in the U-Net paper (Ronneberger et al., MICCAI 2015) as:
$$X^l_{\text{decoder}} = \text{DoubleConv}\left([X^l_{\text{encoder}} ; \text{UpSample}(X^{l+1}_{\text{decoder}})]\right)$$
**In our project**: 4 skip connections (x1, x2, x3, x4) pass encoder features directly to decoder blocks via concatenation. In the proposed model, skip features are first filtered by SpatialAttention before concatenation.

### Feature Map
**Definition**: A 3D tensor of shape $(C, H, W)$ representing learned feature representations, where $C$ is the number of feature channels (learned filters), and $H, W$ are the spatial dimensions.

### Encoder
**Definition**: The contracting path of a U-Net that progressively reduces spatial resolution while increasing the number of feature channels, extracting increasingly abstract semantic representations.

### Decoder
**Definition**: The expanding path of a U-Net that progressively restores spatial resolution using upsampling, combining coarse semantic information with fine-grained skip features to produce dense pixel-wise predictions.

### Bottleneck
**Definition**: The lowest-resolution, highest-abstraction stage at the junction between encoder and decoder. In our model: $(B, 256, 16, 16)$ for a 256×256 input. All attention mechanisms are applied here.

### Upsampling
**Definition**: An operation that increases the spatial resolution of a feature map.  
**In our project**: Bilinear interpolation is used (`nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)`). See [`unet_decoder_upsampling.md`](unet_decoder_upsampling.md) for full details.

### Transposed Convolution (Deconvolution)
**Definition**: A learnable upsampling operation that performs convolution in the reverse direction, using a learned kernel to map from a low-resolution feature map to a higher-resolution one. Also called "fractionally strided convolution."  
**In our project**: `nn.ConvTranspose2d` is available as an alternative (when `bilinear=False`), but bilinear interpolation is the default choice.

---

## C. Attention Mechanism Terms

### Attention Mechanism
**Definition**: A learnable weighting function that assigns higher importance scores to more relevant spatial positions or feature channels. Defined generally as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
where Q (query), K (key), V (value) are linear projections of the input.  
**In our project**: We implement three distinct attention mechanisms: (1) Spatial Attention (SAM), (2) Top-k Sparse Routing, and (3) Linear Kernel Attention. See [`attention_mechanism_deep_dive.md`](attention_mechanism_deep_dive.md).

### Spatial Attention (Channel-Pooling Attention)
**Definition**: A 2D attention map $M_s \in \mathbb{R}^{1 \times H \times W}$ computed from pooled inter-channel statistics. Suppresses spatially uninformative regions.  
**Formula**: $M_s(F) = \sigma(\text{Conv}_{7\times7}([\text{AvgPool}(F) ; \text{MaxPool}(F)]))$  
**In our project**: Implemented in [`src/spatial_attention.py`](../../../src/spatial_attention.py). Applied at bottleneck and on each skip connection in the decoder.

### Channel Attention
**Definition**: A vector $M_c \in \mathbb{R}^{C \times 1 \times 1}$ that reweights feature channels based on their global importance, using Global Average Pooling followed by FC layers.  
**In our project**: The cross-scale attention gate in `CrossFeatureInteraction` (`src/cross_feature.py`) uses `AdaptiveAvgPool2d(1)` followed by two 1×1 convolutions — this is channel attention (Squeeze-and-Excitation mechanism).

### Linear Attention (Kernel Attention)
**Definition**: An efficient alternative to standard softmax attention that replaces the softmax kernel with a feature map $\phi(x) = \text{ELU}(x) + 1$, enabling factorization of the attention computation from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N d^2)$.  
**Formula**: $O_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)}{\phi(q_i)^T \left(\sum_j \phi(k_j)\right)}$  
**In our project**: Implemented in [`src/linear_attention.py`](../../../src/linear_attention.py) and used within `SpatialSparseLinearAttentionBlock` in `proposed_model.py`.

### Sparse Attention (Top-k Routing)
**Definition**: An attention variant that restricts key-value computation to only the top-k highest-scoring spatial positions, reducing complexity to $\mathcal{O}(N \cdot k \cdot d)$ where $k \ll N$.  
**In our project**: A lightweight 1×1 conv router assigns saliency scores to each of the $N = H \times W$ spatial positions. Top $k=32$ positions are selected. Keys and values from these positions are gathered, and linear attention is computed between all queries and the sparse key-value set. Implemented in `SpatialSparseLinearAttentionBlock` in [`src/proposed_model.py`](../../../src/proposed_model.py).

### Multi-Head Attention
**Definition**: Splits the feature dimension into $h$ parallel heads, each computing attention in a subspace of dimension $d/h$. Different heads learn to attend to different semantic aspects.  
**In our project**: `num_heads=4`, `head_dim = in_channels // 4 = 64` (at the 256-channel bottleneck).

### Residual Connection
**Definition**: An additive bypass connection that adds the input to the output of a layer: $\text{out} = f(x) + x$. Prevents vanishing gradients and allows the network to learn identity mappings.  
**In our project**: Every attention block uses residual connections: `out = self.out_proj(attention_result) + x`.

---

## D. Cross-Feature Interaction Terms

### Cross-Feature Interaction (Multi-Scale Feature Fusion)
**Definition**: A mechanism that explicitly harmonizes feature representations from multiple encoder scales by spatially aligning, projecting, and gating their combinations.  
**In our project**: `CrossFeatureInteraction` (`src/cross_feature.py`) takes f_low (B,64,H/2,W/2), f_mid (B,128,H/4,W/4), and f_high (B,256,H/16,W/16), aligns them to H/16×W/16 via adaptive pooling, projects all to 64-dim, concatenates to 192-dim, fuses via 3×3 conv, and gates with SE attention.

### Squeeze-and-Excitation (SE) Gate
**Definition**: A channel recalibration mechanism that globally pools spatial information ("squeeze") and learns per-channel importance weights ("excitation") using two FC layers.  
**Formula**: $\hat{F} = F \odot \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(F)))$  
**In our project**: The `gate` module in `CrossFeatureInteraction` uses `AdaptiveAvgPool2d(1)` + two 1×1 convs + Sigmoid — this is the SE mechanism.

---

## E. Loss Function Terms

### Binary Cross-Entropy (BCE) Loss
**Definition**: $\mathcal{L}_{BCE} = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$. Penalizes incorrect probability predictions for binary (foreground/background) classification.

### Soft Dice Loss
**Definition**: A differentiable approximation of the Dice Similarity Coefficient optimized directly as a training loss:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum \hat{y} \cdot y + \epsilon}{\sum \hat{y} + \sum y + \epsilon}$$
Addresses class imbalance by measuring overlap rather than per-pixel correctness.

### Hybrid BCE-Dice Loss
**Definition**: $\mathcal{L} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$. Combines pixel-level accuracy (BCE) with region overlap optimization (Dice).  
**In our project**: Default loss function configured in `src/config.py` and implemented in `src/losses.py`.

---

## F. Evaluation Metric Terms

### Dice Similarity Coefficient (DSC)
$$\text{DSC} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
Measures volumetric overlap between predicted and ground-truth masks. Proposed model: **0.2500** vs baseline: 0.0132.

### Specificity (True Negative Rate)
$$\text{Specificity} = \frac{TN}{TN + FP}$$
Measures the model's ability to correctly identify non-nodule tissue. **Proposed model achieved perfect 1.0000 specificity**, eliminating all false-positive parenchymal hallucinations.

### Precision (Positive Predictive Value)
$$\text{Precision} = \frac{TP}{TP + FP}$$
Fraction of predicted foreground pixels that are actually nodule. Proposed model: **0.2500** vs baseline: 0.0067 (+37× improvement).

---

## G. Data Pipeline Terms

### DICOM (Digital Imaging and Communications in Medicine)
**Definition**: The international standard for medical imaging data, storing CT slices as 16-bit unsigned integer arrays along with acquisition metadata (patient ID, scanner parameters, image geometry).

### Patient-Level Splitting
**Definition**: A data partitioning strategy that ensures all slices from a single patient are placed entirely within one split (train, val, or test), preventing data leakage via inter-slice anatomical correlation.

### 50% Majority Consensus
**Definition**: An annotation aggregation rule that accepts a voxel as foreground if ≥50% of annotating radiologists marked it:
$$M_{\text{consensus}}(x,y) = \mathbb{I}\left(\frac{1}{M}\sum_{m=1}^M B_m(x,y) \geq 0.5\right)$$
**In our project**: Up to 4 thoracic radiologists annotate each nodule; the consensus mask is the intersection of majority agreement, filtering idiosyncratic outlier contours.

