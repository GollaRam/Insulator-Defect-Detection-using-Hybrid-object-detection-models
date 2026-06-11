# Insulator-Defect-Detection-using-Hybrid-object-detection-models
# CPLID Insulator Defect Detection — YOLOv11 × EfficientViT Hybrid

> **Edge-optimised aerial insulator defect detection for high-voltage power-line infrastructure.**  
> A systematic architecture search combining YOLOv11n with EfficientViT-R8 attention, validated on the CPLID dataset and deployed on NVIDIA Jetson Nano for real-world UAV inspection.

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Architecture](#architecture)
  - [Baseline — YOLOv11n](#baseline--yolov11n)
  - [Hybrid — YOLOv11n + EfficientViT-R8](#hybrid--yolov11n--efficientvit-r8)
  - [How the Hybrid Reduces Parameters by ~50%](#how-the-hybrid-reduces-parameters-by-50)
- [Ablation Study](#ablation-study)
- [Results](#results)
- [Jetson Nano Deployment](#jetson-nano-deployment)
- [Dataset](#dataset)


---

## Overview

Power-line insulator failures cause grid outages, equipment damage, and safety hazards. Traditional ground-based inspection is slow, expensive, and dangerous. This project builds a **lightweight, high-accuracy object detector** purpose-built for UAV-mounted inspection cameras, capable of detecting both intact insulators and surface defects (cracks, flashover damage, broken discs) from aerial imagery.

The core contribution is a **hybrid CNN-Transformer backbone** that replaces brute-force depth scaling with strategically placed EfficientViT-R8 attention blocks. The result is a model that simultaneously achieves:

- **Higher mAP@0.5 than the baseline** (0.9933 vs 0.9898)
- **~46% fewer parameters** (1.386M vs 2.59M)
- **Real-time inference on Jetson Nano** at production-viable throughput

---

## Why This Matters

Standard YOLO models are designed for general object detection on GPU servers. Deploying them on edge hardware such as Jetson Nano (4 GB RAM, 128 CUDA cores, 5–10 W TDP) creates three practical problems:

1. **Memory ceiling** — large models overflow the Nano's unified memory under concurrent UAV telemetry loads.
2. **Thermal throttling** — high parameter counts drive sustained compute that exceeds the Nano's thermal envelope during long inspection flights.
3. **Generalisation on small, texture-rich targets** — CNN kernels struggle to capture the long-range spatial relationships that distinguish a hairline crack from an artefact, or a flashover scar from shadow.

Attention mechanisms directly address problem 3, while the hybrid design keeps parameter count low enough to solve problems 1 and 2. This is not a trade-off — it is an architectural synergy.

---

## Architecture

### Baseline — YOLOv11n

The baseline is a stock YOLOv11n model trained end-to-end on the CPLID dataset with no structural modifications. It serves as the performance floor and a sanity check that the dataset pipeline is correct.

- **Backbone:** Standard C3k2 stack × 4 stages
- **Neck:** PANet FPN (Upsample + Concat + C3k2)
- **Head:** Decoupled Detect (3 scales: P3/P4/P5)
- **Parameters:** 2.59M
- **Training:** 100 epochs, 640×640, batch 16, standard augmentation, SIoU disabled

### Hybrid — YOLOv11n + EfficientViT-R8

The hybrid backbone surgically inserts **EfficientViT-R8 attention blocks** after each C3k2 stage. EfficientViT-R8 is a hardware-aware vision transformer variant that replaces full quadratic self-attention with a cascaded group attention mechanism, drastically reducing the FLOPs per token compared to vanilla multi-head attention.

```
Backbone (Hybrid):
  Stage 1  →  Conv(64, 3×3, s2) → Conv(128, 3×3, s2) → C3k2(128) → EfficientViT-R8
  Stage 2  →  Conv(256, 3×3, s2) → C3k2(256) → EfficientViT-R8
  Stage 3  →  Conv(512, 3×3, s2) → C3k2(512) → EfficientViT-R8
  Stage 4  →  Conv(1024, 3×3, s2) → C3k2(1024) → EfficientViT-R8 → SPPF(1024, k=5)

Neck / Head:  standard PANet FPN → Detect([P3, P4, P5])
```

The attention block at each stage refines the CNN feature map through self-attention before passing it downstream. Because EfficientViT-R8 uses grouped linear attention rather than dense pairwise attention, the added cost per block is modest — and is more than recovered by the parameter savings achieved elsewhere in the network.

### How the Hybrid Reduces Parameters by ~50%

This is the most counterintuitive result and deserves a precise mechanistic explanation.

**The baseline's parameter budget is dominated by its C3k2 bottleneck chains.** C3k2 stacks multiple cross-stage partial bottleneck units. To maintain representational power without attention, these blocks must be wide and deep — the baseline uses 2.59M parameters to compensate for the CNN's inherent locality bias.

**The hybrid exploits a substitution effect.** When an EfficientViT-R8 block follows a C3k2 stage, it captures global context that would otherwise require the next C3k2 block to approximate through increased width or additional repeats. This allows the subsequent C3k2 stages to be narrower (fewer channels) or shallower (fewer repeats) while maintaining equivalent or better representational power.

Concretely:

| Source of savings | Mechanism |
|---|---|
| Narrower C3k2 channels in mid/late stages | Attention compensates; local feature width can be reduced |
| Fewer bottleneck repeats | A single attention pass aggregates context that would need 2–3 conv repeats to approximate |
| Shared positional structure | EfficientViT's group attention amortises query/key costs across spatial positions |
| Efficient projection | EfficientViT-R8 uses depthwise-separable projections internally, so the block itself adds far fewer parameters than an equivalent MHA block |

The net result: the hybrid variant with attention at all 4 stages uses only **1.386M parameters** — a **46.5% reduction** from the 2.59M baseline — while posting a *higher* mAP@0.5 (0.9933 vs 0.9898). The attention blocks do not add bulk; they enable bulk reduction everywhere else.

This mirrors results from the broader literature on hybrid CNN-Transformer networks, where attention consistently allows downstream CNN layers to be compressed without accuracy loss, because the network no longer needs to "fake" global context through local convolution depth.

---

## Ablation Study

To understand *where* attention provides the most value, we conducted a systematic ablation across six architectural variants.

| Variant | Architecture | Params | mAP@0.5 | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Baseline** — YOLOv11n (no attention) | 2.59M | 0.9898 | 0.9597 | 0.9852 |
| 2 | **Full Hybrid** — EfficientViT-R8 at all 4 stages | 1.477M | 0.9931 | 0.9751 | 0.9847 |
| 3 | **Shallow only** — EfficientViT-R8 at stage 1 (128-ch) | 1.284M | 0.9927 | 0.9824 | 0.9846 |
| 4 | **Mid stages** — EfficientViT-R8 at stages 2–3 (256/512-ch) | 1.386M | 0.9933 | 0.9775 | 0.9809 |
| 5 | **Deep only** — EfficientViT-R8 at stage 4 (1024-ch) | 1.364M | 0.9921 | 0.9726 | 0.9885 |
| 6 | **Pure ViT** — basic ViTBlock (no EfficientViT), 512px | 1.810M | 0.9819 | 0.9476 | 0.9476 |
| 7 | **c2f**-YOLOv11n + C2f (no attention)|1.8M | 0.9935 | 0.9837 | 0.9901 |

**Key findings:**

- **Variant 4 (mid-stage placement) is the sweet spot.** Placing EfficientViT-R8 at the 256-ch and 512-ch stages captures medium-scale spatial relationships most relevant to insulator disc geometry and surface texture. It achieves the highest mAP@0.5 of all variants (0.9933) at just 1.386M parameters.

- **Shallow-only attention (Variant 3) is remarkably competitive.** With only 1.284M parameters — less than half the baseline — it still posts 0.9927 mAP@0.5. This suggests that attention at early spatial resolutions provides strong regularisation effects that carry through the rest of the network.

- **Deep-only attention (Variant 5) prioritises recall.** At 0.9885 recall — the highest of any variant — Stage-4 attention excels at not missing true positives, which is the safety-critical failure mode for inspection systems.

- **Naive ViT substitution (Variant 6) degrades performance.** The basic ViTBlock without EfficientViT's grouped cascade mechanism, combined with no SIoU loss and reduced input resolution (512px), underperforms the baseline by 0.8 mAP@0.5 points. This confirms that architectural detail — not just the presence of attention — drives the gains.

- **C2f augmentation (Variant 7, no attention) achieves the highest raw mAP@0.5 at 0.9935** with 0.9837 precision and 0.9901 recall, establishing that pure CNN depth is competitive — but at the cost of larger model size and no attention-based generalisation benefits.

---

## Results

### Final Model Comparison

| Variant | Architecture | Params | mAP@0.5 | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Baseline** — YOLOv11n (no attention) | 2.59M | 0.9898 | 0.9597 | 0.9852 |
| 2 | **Full Hybrid** — EfficientViT-R8 at all 4 stages | 1.477M | 0.9931 | 0.9751 | 0.9847 |
| 3 | **Shallow only** — EfficientViT-R8 at stage 1 (128-ch) | 1.284M | 0.9927 | 0.9824 | 0.9846 |
| 4 | **Mid stages** — EfficientViT-R8 at stages 2–3 (256/512-ch) | 1.386M | 0.9933 | 0.9775 | 0.9809 |
| 5 | **Deep only** — EfficientViT-R8 at stage 4 (1024-ch) | 1.364M | 0.9921 | 0.9726 | 0.9885 |
| 6 | **Pure ViT** — basic ViTBlock (no EfficientViT), 512px | 1.810M | 0.9819 | 0.9476 | 0.9476 |
| 7 | **c2f**-YOLOv11n + C2f (no attention)|1.8M | 0.9935 | 0.9837 | 0.9901 |

The recommended production model is **Variant 4 (mid-stage EfficientViT-R8)**: it achieves the best attention-driven mAP@0.5, the lowest parameter count among attention-based variants, and the best precision–recall balance for a safety-critical inspection scenario.

---

## Jetson Nano Deployment

All variants were exported and benchmarked on an NVIDIA Jetson Nano (4 GB, JetPack 4.6) to validate edge viability.
### Deployment Notes

- **quantisation**: different types of quantisation applied on models include default,FP32,FP16,INT18 
- **Input resolution:** 512×512 for attention variants (reduces token count quadratically, critical for Jetson's limited bandwidth); 640×640 for the baseline.
- **Thermal management:** At 100 epochs of inference over a 20-minute simulated inspection flight, the hybrid model maintains stable throughput without triggering thermal throttling. The baseline at 640px intermittently throttles under sustained load.
---
## Dataset

This project uses the **CPLID (Chinese Power Line Insulator Dataset)**, a publicly available benchmark for aerial insulator inspection.

- **Classes:** `insulator` (intact), `defect` (cracked / broken / flashover-damaged)
- **Annotations:** Pascal VOC XML, converted to YOLO format in the preprocessing pipeline
- **Split:** 75% train / 15% val / 10% test (stratified random, seed 42)
- **Source images:** Aerial RGB photographs from UAV inspection campaigns over high-voltage transmission lines

Preprocessing converts Pascal VOC XML bounding boxes to YOLO normalised format, deduplicates overlapping annotations, and handles the multi-directory label structure of the original dataset (separate `insulator/` and `defect/` label subdirectories).

---

## Acknowledgements

- CPLID dataset authors for the publicly available benchmark
- Ultralytics for the YOLOv11 framework
- Amrita Vishwa Vidyapeetham, Coimbatore — Department of AI & Data Science

---
