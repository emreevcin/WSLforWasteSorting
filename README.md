# Weakly Supervised Localization for Waste Sorting
**Politecnico di Milano — Computer Science and Engineering**

---

## 1. Classifier Architecture

The classifier is a **ResNet-50** pretrained on ImageNet, fine-tuned for binary classification (Before/After). One architectural modification is central to the design: the standard **Global Average Pooling (GAP)** layer is replaced with **Log-Sum-Exp (LSE) Pooling**.

Standard GAP computes the spatial mean of each feature channel, treating all spatial locations equally. This allows the classifier to accumulate evidence from any region of the image — including the conveyor belt background — and still produce a correct classification. The consequence for localization is a diffuse, spatially imprecise gradient signal. LSE pooling, parameterized by a concentration factor *r = 5*, interpolates between average and max pooling. Mathematically, it exponentially upweights the highest activations within each channel, biasing the gradient toward spatially concentrated regions rather than diffuse scene-level cues. This is the core architectural reason we expect the CAM heatmaps to be more tightly localized than a standard ResNet-50 baseline.

Training follows a two-phase strategy: the backbone is frozen for 2 warm-up epochs while only the classifier head trains, then the full network is fine-tuned at a low learning rate (1e-5) with cosine annealing. Light augmentation (color jitter, random erasing, Gaussian blur) is applied to discourage shortcut reliance on belt lighting patterns. CutMix (25% probability) adds regularization without over-polluting the spatial labels needed for CAM quality.

**Classification results on the held-out test set (1,497 images):**
- Accuracy: **94.99%**
- F1-Score (Waste class): **91.82%**

---

## 2. CAM Technique: Grad-CAM++

**Grad-CAM++** (Chattopadhyay et al., 2018) is used as the primary localization method, implemented via `pytorch-grad-cam` targeting `model.layer4[-1]` — the final convolutional block of ResNet-50, outputting a 14×14 spatial feature map.

### Why Grad-CAM++ over standard Grad-CAM or weight-based CAM

Standard weight-based CAM requires that the pooling layer immediately preceding the classifier be GAP, and computes `CAM(i,j) = Σ_c w_c · A_c(i,j)`. Because this architecture uses LSE pooling (not GAP), the analytical weight-gradient relationship no longer holds cleanly — standard CAM is inapplicable without modification. For reference, a custom LSE-aware CAM is also implemented and benchmarked; it achieves mIoU of 0.09, confirming it underperforms gradient-based methods.

Standard Grad-CAM weights each feature channel by the *global spatial average* of its gradient, which discards spatial variation within the gradient map. Grad-CAM++ instead computes **pixel-wise gradient importance** using a second-order approximation: it weights each spatial location (i,j) in channel c by `α_c^{ij}`, derived from the positive partial derivatives of the class score with respect to activations. For scenes with multiple isolated waste objects — the typical case in this dataset — this makes a measurable difference: Grad-CAM reaches mIoU 0.116 while Grad-CAM++ reaches **mIoU 0.160**, a 38% relative improvement.

Heatmaps are post-processed through: spatial guardrail (5% border exclusion to suppress conveyor-edge shortcuts), an entropy gate to suppress fully diffuse activations, Otsu thresholding with a 25% intensity floor, morphological closing, adaptive dilation, contour hole-filling, and connected component filtering.

---

## 3. Results Analysis

### 3.1 Quantitative Summary

| Metric | Value |
|--------|-------|
| Mean IoU (Grad-CAM++) | 0.1603 |
| Median IoU (Grad-CAM++) | 0.1664 |
| % samples IoU > 0.2 | 33.9% |
| % samples IoU > 0.3 | 4.6% |
| % samples IoU < 0.05 | 13.1% |
| Pearson r (Confidence vs. IoU) | 0.49 (R² = 24.3%) |
| FP mask coverage on clean images | 1.15% |

The mean and median IoU are nearly equal (0.160 vs. 0.166), indicating a reasonably symmetric distribution rather than one dominated by extreme outliers. Only 13.1% of samples fall below IoU = 0.05 — the near-zero failure rate is contained.

---

### 3.2 Success Cases

<p align="center">
  <img src="assets/success_cases.png" width="400" alt="Success Cases">
</p>
*(Figure 1: Success Cases — Grad-CAM++ Correctly Localizes Waste Objects)*

In **Figure 1**, rows 1 and 2 (Peak IoU 0.374 and 0.366) show the pipeline at its best. Scenes contain dense piles of plastic bottles, packaging bags, and colored cans against a dark conveyor belt background. The Grad-CAM++ heatmap (column 2) concentrates its highest activations — the red and yellow regions in the jet colormap — directly over the waste objects rather than the belt surface. The contrast between the colorful, geometrically complex waste and the uniform dark belt provides strong discriminative signal, and the gradient correctly attributes classification confidence to those foreground regions.

Rows 3 and 4 (typical cases, IoU 0.292 and 0.257) illustrate the more common scenario: the model detects the main waste cluster but misses peripheral items at the scene edges. This is structurally expected — the connected-components step retains only the dominant detection region (`top_k_objects = 1`), discarding secondary valid activations. Comparing the predicted mask with the expert ground truth reveals that the *heatmap* itself is often spatially accurate across multiple objects; the loss occurs during mask binarization, not in the CAM quality.

Row 5 (high confidence, IoU 0.207) shows a very dense pile where the heatmap is somewhat more diffuse but still broadly co-localizes with the waste region. These cases confirm that the weakly supervised signal is real: the classifier has genuinely learned to attend to waste objects as the discriminating feature between Before and After states.

---

### 3.3 Failure Cases and Critical Analysis

<p align="center">
  <img src="assets/failure_audit.png" width="400" alt="Failure Audit">
</p>
*(Figure 2: Failure Audit — Three Distinct Failure Modes)*

**Figure 2** documents three structurally distinct failure modes, each with a different root cause.

#### Mode 1 — The Scattered (IoU: 0.009, frame_0039)
*Row 1: high confidence (0.952), near-zero localization.*

The heatmap shows activation distributed uniformly across the entire frame, with no concentration. The entropy gate does not fire because, while diffuse, the entropy does not cross the threshold (0.97) at this specific confidence level. The root cause is a scene with many small waste items spread across the full image area. Grad-CAM++ assigns gradient weight proportionally across all activated channels; when 10–15 objects all contribute small but equal gradients, the weighted sum produces a uniform spatial map with no dominant region. **This is not a model error — the model is correctly identifying that waste is everywhere — but it reveals the fundamental mismatch between single-object CAM design and multi-object scenes.** No post-processing can recover spatial precision from a genuinely uniform gradient.

#### Mode 2 — The Arrogant (IoU: 0.034, frame_0004)
*Row 2: high confidence (0.936), concentrated activation in the wrong region.*

The heatmap shows a focused hot spot, but it partially lands on background belt texture rather than the waste objects. This is the classical weakly supervised shortcut failure: the classifier has learned a spurious co-occurrence between certain belt illumination patterns (which appear consistently in Before frames, e.g., due to camera angle at the moment of recording) and the Before label. Because the classifier only needs any discriminative feature to achieve high accuracy, background texture that correlates with the label is a valid solution from the loss function's perspective. The loss function imposes no spatial penalty — this failure mode is inherent to weakly supervised learning and cannot be addressed without spatial supervision.

#### Mode 3 — The Clumsy (IoU: 0.000–0.080, frames 0007, 0027, 0022)
*Rows 3–5: mid-range confidence (0.831–0.877), mask spatially misaligned.*

Row 3 (frame_0007) shows a dark scene with sparse, small waste items. The heatmap is diffuse and the resulting mask covers large background areas. Row 4 (frame_0027) shows a dense colorful pile where the heatmap has legitimate hot spots, but the predicted mask does not overlap the GT region. Row 5 (frame_0022) achieves partial overlap (IoU 0.080), showing intermediate behavior.

The common cause across all three cases is the **14×14 spatial resolution ceiling** of `layer4[-1]`. At 448×448 input, this corresponds to a stride-32 receptive field: each feature cell covers a 32×32 pixel patch. For the small waste items in rows 3–4, the object may span only 1–2 feature cells, making precise spatial attribution impossible regardless of gradient method. The 32× bilinear upsampling from 14×14 to 448×448 produces smooth interpolated blobs that systematically misalign with precise object boundaries. Row 5 represents the marginal case where the object is large enough to span several feature cells, enabling coarse but imperfect localization.

---

## 4. Conceptual Understanding

### Why Classification Enables Localization

In a Before/After conveyor belt dataset, the only consistent visual difference between the two classes is the presence of illicit waste objects. A classifier that achieves 95% accuracy on this binary task must, by necessity, have learned to detect the features that distinguish waste from an empty belt. Grad-CAM++ exploits this: it computes the gradient of the predicted class score with respect to the final convolutional activation map, revealing *which spatial locations most influenced the classification decision*. Regions that contributed strongly to a "Before" prediction correspond, in a well-trained model, to the waste objects themselves.

This is the core insight of weakly supervised localization: **the classification label implicitly encodes spatial information**, and gradient-based attribution methods make that spatial information explicit without requiring pixel-level annotations during training.

### Limitations of the Weakly Supervised Paradigm

The results honestly reflect three fundamental limitations of this approach:

1. **No spatial constraint during training.** The cross-entropy loss optimizes classification accuracy, not localization quality. A model that learns to detect background belt texture that co-occurs with waste achieves the same training loss as one that localizes waste precisely. The Pearson correlation between classification confidence and localization IoU is 0.49 (R² = 0.24), meaning 76% of IoU variance is unexplained by confidence — confirming that the classifier is partially solving the problem via non-localizable shortcuts.

2. **Resolution bottleneck.** The 14×14 feature map from `layer4[-1]` imposes a hard ceiling on spatial precision. Fine-grained object boundaries and small isolated waste items cannot be resolved at stride-32 granularity. This is a structural limitation of standard ResNet-50, not a training failure.

3. **Multi-instance mismatch.** Standard CAM was designed for single-object localization tasks. Conveyor belt scenes consistently contain 5–15 waste objects per frame. When multiple objects activate the network simultaneously, gradient weights are distributed across all of them, reducing spatial concentration and making threshold-based mask extraction unreliable. This is the direct cause of the Scattered failure mode.

These limitations are not design failures — they are inherent properties of weakly supervised localization that motivate more constrained methods (e.g., seeded region growing, IRNet, or SEAM) when higher IoU is required.
