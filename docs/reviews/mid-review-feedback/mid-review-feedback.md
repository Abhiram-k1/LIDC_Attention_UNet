# Mid-Review Faculty Feedback — Team-12
## Cross-Feature Spatial Sparse Linear Attention U-Net for CT Lung Nodule Segmentation

**Source**: Handwritten faculty notes from the mid-semester review of Team-12  
**Original Image**: [`mid-review-feedback-original.png`](mid-review-feedback-original.png)  
**Transcribed**: 2026-10-30  
**Course**: Computer Vision (Semester 5)  
**Institution**: Amrita Vishwa Vidyapeetham

---

## Verbatim Transcription

> **Notes from PPT**
>
> → Depth feature 7 in channels.
>
> → U should know particular words when you explain like local features.
>
> → U-Net Decoder, (read about)
>     Upsampling → Methods
>     What is the method you are using here., you should know this.
>
> → Block diagram – ↑ Explain in some good points.
>
>     Need feature  Pay attention to
>     Attention Mechanism
>           &
>     U-Net Decoder    } → Read about it & get clarity.
>
> → Work on Architecture (Take ~~relevance~~ reference from papers like IEEE)
> → Your template → Correct Amruta ik Amrita
> → Literature survey → Read properly.

---

## Interpretation Notes

> [!IMPORTANT]
> The feedback header reads "Notes from PPT", indicating these are faculty notes taken during or after observing the Team-12 mid-semester presentation slides.
> All points below preserve the original wording. Technical interpretation and action items are tracked in [feedback-implementation-tracker.md](feedback-implementation-tracker.md).

### Point-by-Point Breakdown

| # | Original Text | Technical Interpretation |
|---|---|---|
| 1 | "Depth feature 7 in channels" | The model/feature input has 7 channels; faculty wants an explanation of what each channel represents |
| 2 | "U should know particular words when you explain like local features" | Team must be able to precisely define technical terms used (e.g., "local features") during presentation |
| 3 | "U-Net Decoder, (read about) / Upsampling → Methods / What is the method you are using here." | Faculty expects clear knowledge of the exact upsampling technique (bilinear interpolation vs. transposed convolution) used in the decoder |
| 4 | "Block diagram – ↑ Explain in some good points" | The architecture block diagram needs improvement; faculty wants clearer explanations for each block |
| 5 | "Need feature / Pay attention to / Attention Mechanism & U-Net Decoder → Read about it & get clarity" | The team must study and clearly explain both the attention mechanism and the U-Net decoder operation |
| 6 | "Work on Architecture (Take reference from papers like IEEE)" | Architecture decisions should be backed by peer-reviewed IEEE papers |
| 7 | "Your template → Correct Amruta ik Amrita" | The project template/report incorrectly spells the institution name; should be "Amrita Vishwa Vidyapeetham" |
| 8 | "Literature survey → Read properly" | The literature review needs more depth; team should thoroughly read and understand the cited papers |

---

## Summary of All Feedback Items

1. **7-Channel Depth Features** — Understand and document all 7 input channels, what each represents, and how they contribute to the model.
2. **Technical Terminology** — Be able to precisely define "local features" and other domain-specific terms.
3. **U-Net Decoder & Upsampling Methods** — Study the U-Net decoder thoroughly; identify and justify the exact upsampling method used.
4. **Architecture Block Diagram** — Improve the block diagram with clear labels, data flow, tensor dimensions, and explanatory annotations.
5. **Attention Mechanism** — Deeply understand and clearly explain the attention mechanism(s) used.
6. **IEEE Architecture References** — Support architectural decisions with references from IEEE or equivalent peer-reviewed publications.
7. **Institution Name Correction** — Fix "Amruta" → "Amrita" in the project template/report.
8. **Literature Survey Quality** — Read cited papers properly; ensure the literature survey accurately reflects and synthesizes the cited works.

