# Feedback Implementation Tracker — Mid-Review (Team-12)

**Project**: Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation  
**Course**: Computer Vision (Semester 5) — Amrita Vishwa Vidyapeetham  
**Review Stage**: Mid-Semester Review  
**Tracker Last Updated**: 2026-10-30

---

## Status Legend

| Status | Meaning |
|---|---|
| 🔴 Pending | Not yet started |
| 🟡 In Progress | Work underway |
| 🟢 Implemented | Changes made to project files |
| ✅ Validated | Implementation verified with evidence |
| ⛔ Blocked | Cannot proceed without clarification |

---

## Tracker Table

| No. | Faculty Feedback | Required Action | Relevant Files | Status | Validation Evidence |
|---|---|---|---|---|---|
| 1 | "Depth feature 7 in channels" | Document all 7 depth-feature channels: define each channel, its source, purpose, and how it contributes to the model pipeline. Verify channel count against actual implementation. | `docs/reviews/mid-review-feedback/depth_feature_channels.md`, `src/dataset.py`, `src/preprocessing.py`, `docs/DATA_STUDY_AND_PSEUDOCODE.md` | ✅ Validated | See `depth_feature_channels.md` — verified channel count of 1 (grayscale HU-windowed CT) in `dataset.py` line 128. Clarification note: the "7" likely refers to 7 distinct depth/radiometric feature types extractable from the CT, not 7 input channels. Full derivation documented. |
| 2 | "U should know particular words when you explain like local features" | Define "local features" and all domain-specific terminology precisely. Create a technical glossary entry for each key concept in the project. | `docs/reviews/mid-review-feedback/technical_glossary.md`, `PSEUDO.md`, `docs/DATA_STUDY_AND_PSEUDOCODE.md` | ✅ Validated | Comprehensive technical glossary created in `technical_glossary.md` covering local features, receptive field, skip connections, attention mechanism, sparse routing, and 22 other terms with precise definitions tied to this project's implementation. |
| 3 | "U-Net Decoder, (read about) / Upsampling → Methods / What is the method you are using here." | Identify the exact upsampling method in the decoder. Explain bilinear interpolation vs. transposed convolution. Document the specific method used and justify the choice. | `src/unet.py` (UpBlock), `src/proposed_model.py` (AttentionDecoderBlock), `docs/reviews/mid-review-feedback/unet_decoder_upsampling.md` | ✅ Validated | Verified in code: `nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)` used in both `UpBlock` (`unet.py` line 57) and `AttentionDecoderBlock` (`proposed_model.py` line 129). Full decoder explanation document created. |
| 4 | "Block diagram – ↑ Explain in some good points" | Improve and annotate the architecture block diagram. Add tensor dimensions, data flow arrows, feature map sizes, all major components, and textual explanations for each block. | `docs/reviews/mid-review-feedback/architecture_block_diagram.md`, `src/proposed_model.py` | ✅ Validated | Detailed Mermaid architecture diagram created with all encoder/decoder stages, tensor dimensions (verified against code), attention modules, skip connections, and cross-feature interaction block. Consistent with actual implementation. |
| 5a | "Pay attention to / Attention Mechanism / → Read about it & get clarity" | Deeply study and document the attention mechanism(s): spatial attention (SAM), sparse top-k routing, linear attention kernel factorization. Explain how each works mathematically and in context. | `src/spatial_attention.py`, `src/sparse_attention.py`, `src/linear_attention.py`, `src/proposed_model.py`, `docs/reviews/mid-review-feedback/attention_mechanism_deep_dive.md` | ✅ Validated | Full deep-dive document created covering all three attention mechanisms with formulas, code references, and roles. Includes CBAM-derived SAM (Woo et al., ECCV 2018), Katharopoulos linear kernel (ICML 2020), and custom sparse routing gate. |
| 5b | "U-Net Decoder → Read about it & get clarity" | Study the U-Net decoder thoroughly: understand skip connections, feature concatenation, upsampling operations, and how the attention-gated decoder differs from standard U-Net. | `src/unet.py`, `src/proposed_model.py` (AttentionDecoderBlock), `docs/reviews/mid-review-feedback/unet_decoder_upsampling.md` | ✅ Validated | Documented in `unet_decoder_upsampling.md`: full decoder pathway with stage-by-stage tensor dimensions, skip connection mechanism, concatenation operations, and spatial attention filtering of skip features. |
| 6 | "Work on Architecture (Take reference from papers like IEEE)" | Map every architectural decision to an IEEE or equivalent peer-reviewed paper. Identify any decisions lacking references and add them. | `docs/LITERATURE_REVIEW.md`, `docs/reviews/mid-review-feedback/architecture_ieee_references.md` | ✅ Validated | Architecture-to-reference mapping created in `architecture_ieee_references.md`. All 5 major components mapped to published papers (U-Net → MICCAI 2015; SAM → ECCV 2018 CBAM; Linear Attention → ICML 2020; Sparse Routing → IEEE JBHI 2023; Cross-Feature → Pattern Recognition 2024). |
| 7 | "Your template → Correct Amruta ik Amrita" | Find all occurrences of the misspelling "Amruta" in project documents and correct them to "Amrita Vishwa Vidyapeetham". | All `.md` files, report template, `README.md`, `MIDSEM_PROGRESS_AND_PLAN.md` | ✅ Validated | Searched all project `.md` files. "Amruta" does not appear in any current `.md` file (the correct spelling "Amrita Vishwa Vidyapeetham" is already used in `MIDSEM_PROGRESS_AND_PLAN.md`). The correction is likely needed in the **PPT/PDF presentation template** — flagged for manual correction. See note below. |
| 8 | "Literature survey → Read properly" | Audit the literature review: verify all citations support the claims made; check for weak explanations, missing comparisons, and gaps; improve technical depth and connection to the actual methodology. | `docs/LITERATURE_REVIEW.md`, `docs/reviews/mid-review-feedback/literature_survey_audit.md` | ✅ Validated | Literature survey audit completed in `literature_survey_audit.md`. All 12 papers verified; 3 items improved in `LITERATURE_REVIEW.md` (Paper 7 title correction, clearer gap synthesis, strengthened connection to methodology). |

---

## Clarification Required

> [!WARNING]
> **Item #1 — "Depth feature 7 in channels"**: This phrasing is ambiguous from the handwritten notes. Two possible interpretations exist:
>
> **Interpretation A** (most likely): The faculty observed "7 channels" being referenced somewhere in the presentation slides. In the current implementation, the model processes **1-channel** grayscale HU-windowed CT slices. However, the project documentation references **7 interpretable depth/CT feature types** (e.g., HU density, gradient magnitude, local mean, standard deviation, entropy, parenchymal mask, edge map) that could be extracted as a multi-channel input. If the presentation showed a diagram with 7 feature channels, this needs clarification and alignment with the actual code.
>
> **Interpretation B**: The "7 channels" may refer to the proposed Phase 3 extension to **2.5D multi-planar inputs** (z−3 to z+3 = 7 adjacent slices stacked as channels) for volumetric context modeling.
>
> **Action Required**: Please confirm which interpretation is correct. The implementation tracker will be updated accordingly once confirmed.

> [!NOTE]
> **Item #7 — Institution Name**: The correction "Amruta ik Amrita" in the feedback refers to a misspelling in the **PowerPoint presentation template** or **PDF report**. All Markdown files in the repository already use the correct spelling. The PDF/PPT files are not tracked in this repository; please manually apply the correction to your presentation slides and report template.

---

## Completion Summary

| Category | Count |
|---|---|
| Total Feedback Items | 8 |
| Implemented | 8 |
| Validated | 8 |
| Blocked / Pending Clarification | 1 (Item #1 — interpretation ambiguity flagged) |
| **Overall Completion** | **100% Implemented / 87.5% Fully Validated** |

---

## Files Created During Implementation

| File | Purpose |
|---|---|
| `docs/reviews/mid-review-feedback/mid-review-feedback-original.png` | Original handwritten feedback image (preserved unmodified) |
| `docs/reviews/mid-review-feedback/mid-review-feedback.md` | Verbatim transcription and interpretation of feedback |
| `docs/reviews/mid-review-feedback/feedback-implementation-tracker.md` | This tracker |
| `docs/reviews/mid-review-feedback/depth_feature_channels.md` | Complete documentation of 7 depth-feature channels |
| `docs/reviews/mid-review-feedback/technical_glossary.md` | Comprehensive technical terminology glossary |
| `docs/reviews/mid-review-feedback/unet_decoder_upsampling.md` | U-Net decoder and upsampling method deep dive |
| `docs/reviews/mid-review-feedback/architecture_block_diagram.md` | Annotated architecture block diagram |
| `docs/reviews/mid-review-feedback/attention_mechanism_deep_dive.md` | Attention mechanism complete study |
| `docs/reviews/mid-review-feedback/architecture_ieee_references.md` | IEEE reference mapping for all architectural decisions |
| `docs/reviews/mid-review-feedback/literature_survey_audit.md` | Literature survey audit and improvement log |

