# Illicit Waste Detection: A Weakly Supervised XAI Approach

**Student:** Emre Evcin  
**Institution:** Politecnico di Milano – Computer Science and Engineering

---

## 1. Architecture: Why ResNet-50?

ResNet-50 pretrained on ImageNet was selected as the backbone for three 
reasons that go beyond raw performance:

**Feature Depth:** Its 50-layer residual architecture extracts high-level 
semantic features (textures, material patterns, edges) necessary to 
distinguish waste from a visually similar clean belt.

**Global Average Pooling (GAP):** ResNet-50 natively uses GAP before the 
classifier. This is mathematically essential for CAM-based localization — 
it establishes a direct linear relationship between spatial feature maps 
and class scores, making the FC weights interpretable as spatial importance.

**Two-Phase Fine-Tuning:** Phase 1 (frozen backbone, lr=1e-3) stabilizes 
the new classification head without destroying pretrained features. Phase 2 
(full network, lr=1e-5) specializes deep filters to waste-domain cues.

| Metric | Result |
|--------|--------|
| Classification Accuracy | 99.93% |
| F1 Score | 94.27% |
| Localization Mean IoU | 0.1753 |
| Misclassifications | 54 / 1497 |

---

## 2. The Weakly Supervised Paradigm

The core insight is this: **the only consistent visual difference between 
"Before" and "After" frames is the presence of illicit waste.** A classifier 
trained on this distinction must, by necessity, learn to look at those objects 
to make correct predictions. We exploit this by interrogating the model's 
learned spatial attention — no pixel-level labels required during training.

Three structural properties define this paradigm:

- **Not end-to-end optimized:** The network minimizes Cross-Entropy loss on 
  classification. It is never penalized for imprecise object boundaries.
- **Post-hoc interpretability:** CAM is applied after training as a 
  diagnostic tool, not as a training objective.
- **Minimal discriminative regions:** The classifier only needs one 
  discriminative signal (a shadow, a texture patch) to be confident. It has 
  no mathematical incentive to provide complete spatial coverage.

This is why strong classification accuracy (99.93%) does not guarantee strong 
localization — and why the Pearson correlation between confidence and IoU 
is nearly zero (-0.08).

---

## 3. XAI Strategy: Grad-CAM++ with Layer Fusion

Three CAM methods were implemented and compared:

- **Vanilla CAM** (from scratch): Uses FC weights × GAP feature maps. Fast 
  but limited to the coarse 7×7 feature resolution of layer4.
- **Grad-CAM**: Weights feature maps by their gradient with respect to the 
  target class score. More flexible than vanilla CAM.
- **Grad-CAM++** (selected for final evaluation): Uses second-order gradients, 
  assigning higher importance to pixels that individually activate the target 
  class. Superior for multiple or small object instances.

**Layer Fusion:** Both layer3[-1] (mid-level spatial detail) and layer4[-1] 
(high-level semantics) are targeted simultaneously. This reduces the 
"blobbing" effect common when only the final layer is used.

---

## 4. Quantitative Results

| Method | Mean IoU | Notes |
|--------|----------|-------|
| Vanilla CAM | 0.1536 | Baseline, coarsest resolution |
| Grad-CAM | 0.1750 | Gradient-weighted |
| Grad-CAM++ | 0.1610 | Final method, layer fusion |

Pearson Correlation (Confidence vs. IoU): **-0.0826**  
IoU > 0.2: **35.3%** of test images

---

## 5. Qualitative Analysis

### Success Cases

Five cases are shown covering different scenarios to demonstrate the model
generalizes across varied conditions:

![Success Cases](assets/success_cases.png)
*Columns: Original | Grad-CAM++ Heatmap | Predicted Mask (green) | Ground Truth*

The model succeeds when waste objects are:
- **Large and high-contrast** relative to the belt texture (peak IoU cases)
- **Centrally located** with a dominant activation peak
- **Visually isolated** — no competing background textures near the object

---

### Failure Cases

Three distinct failure modes were identified across 6 cases:

![Failure Audit](assets/failure_audit.png)
*Red: The Arrogant | Orange: The Clumsy | Purple: The Scattered*

**🔴 "The Arrogant" — High Confidence (>0.9), Near-Zero IoU**
The model classifies correctly with near-certain confidence but the heatmap
activates on belt edges, lighting gradients, or frame borders rather than
the object. Root cause: **shortcut learning**. The training set contains
systematic visual differences between Before/After video sequences
(illumination shifts, belt wear) that are easier to exploit than learning
the actual object appearance. The classifier is correct but for the wrong
reason — it found a spurious correlation that happens to correlate with
the label.

**🟠 "The Clumsy" — Mid Confidence, Boundary Mismatch**
The model is uncertain and the heatmap, while roughly centered on the right
region, produces a smooth circular blob that fails to match the sharp
irregular contours of expert annotations. Root cause: **resolution
bottleneck**. Grad-CAM heatmaps originate from a 7×7 spatial grid
(ResNet-50's layer4 output at 224×224 input). Upsampling this to 224×224
inherently smooths boundaries, making precise pixel-level alignment
impossible regardless of how well the model understands the scene.

**🟣 "The Scattered" — Diffuse Attention, Low IoU**
The heatmap activates across multiple disconnected regions rather than
focusing on the object. Root cause: **competing discriminative signals**.
When multiple background regions share visual features with waste objects
(e.g., similar textures, colors), the gradient signal is diluted across
all of them rather than concentrating on the true object.

---

### The Correlation Paradox

The Pearson correlation between model confidence and IoU is **-0.0826**
— effectively zero, and slightly negative. This is not a bug; it is the
expected consequence of the weakly supervised paradigm. The classifier
is optimized to answer *"is waste present?"* not *"where exactly is it?"*
Its certainty about the former tells us almost nothing about its precision
on the latter. This is the fundamental limitation of post-hoc localization
and the reason true segmentation models require pixel-level supervision.

---

## 6. Environment & Reproducibility

Dependencies managed via `requirements.txt`.  
Device selection is automatic: MPS (Apple Silicon) → CUDA → CPU.