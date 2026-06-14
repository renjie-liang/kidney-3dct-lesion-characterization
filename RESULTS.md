# Results

Headline numbers from *Multi-Granularity 3D Kidney Lesion Characterization from
CT Volumes*. All UF-Health numbers are on the held-out validation split
(156 patients / 452 volumes); KiTS23 is zero-shot external validation (489 cases).

These results require the **private UF Health dataset** and cannot be reproduced
from the public/synthetic data in this repository. They are reported here so the
quantitative outcomes are available even though the data is not.

## Headline

- **Side-level abnormality (bilateral L1 AUC):** `0.799 ± 0.009` on UF-Health,
  `0.817 ± 0.072` zero-shot on KiTS23 (LesionDETR, cropped CT, SuPreM encoder).
- **Per-lesion detection (cystic mAP):** `0.190 ± 0.083` with the
  count-conditioned head. Rare solid-lesion AP stays at the noise floor —
  the next bottleneck is targeted data collection, not architecture.
- **Two dominant design levers:** a segmentation mask as an input channel, and
  same-domain abdominal pretraining (SuPreM). Generic large-corpus pretraining
  (VoCo) is no better than random initialization.

## Table 2 — Side-level metrics across input × training granularity

L1/L2 columns are per-side AUC (↑); L2 *Size* is MAE in cm (↓); L3 columns are
per-class / mean Average Precision at size tolerance τ = 1 cm (↑). L3-trained
side-level numbers come from hierarchical aggregation, not direct supervision.

| CT input (training) | L1 Left | L1 Right | L2 Left cyst | L2 Right cyst | L2 Size | AP_cyst | AP_solid | mAP |
|---|---|---|---|---|---|---|---|---|
| **Trained at L1** | | | | | | | | |
| Whole CT          | 0.647 | 0.612 | — | — | — | — | — | — |
| Cropped CT        | 0.727 | 0.697 | — | — | — | — | — | — |
| + Kidney mask     | **0.873** | 0.802 | — | — | — | — | — | — |
| + Full mask       | 0.852 | **0.828** | — | — | — | — | — | — |
| **Trained at L2** | | | | | | | | |
| Whole CT          | 0.661 | 0.595 | 0.664 | 0.599 | 1.111 | — | — | — |
| Cropped CT        | 0.687 | 0.679 | 0.692 | 0.681 | 0.953 | — | — | — |
| + Kidney mask     | 0.843 | 0.795 | 0.843 | 0.797 | 0.790 | — | — | — |
| + Full mask       | **0.860** | **0.801** | **0.858** | **0.801** | **0.770** | — | — | — |
| **Trained at L3** | | | | | | | | |
| Whole CT          | 0.651 | 0.633 | 0.659 | 0.636 | 1.439 | 0.150 | **0.053** | **0.102** |
| Cropped CT        | **0.817** | 0.781 | **0.826** | 0.778 | 1.460 | 0.145 | 0.020 | 0.082 |
| + Kidney mask     | 0.777 | 0.726 | 0.788 | 0.723 | 1.392 | **0.165** | 0.015 | 0.090 |
| + Full mask       | 0.801 | **0.804** | 0.807 | **0.803** | **1.257** | 0.155 | 0.037 | 0.096 |

## Table 3 — Per-lesion head architecture comparison

Cropped-CT input, SuPreM encoder, layered LR (0.1× on encoder), mean ± std over
3 seeds. L1/L2 are bilateral AUC; AP at τ = 1 cm.

| Architecture | L1 abn ↑ | L2 cyst ↑ | L2 solid ↑ | AP_cyst ↑ | AP_solid ↑ |
|---|---|---|---|---|---|
| LesionDETR    | **0.799** ± 0.009 | **0.802** ± 0.007 | 0.698 ± 0.042 | 0.145 ± 0.013 | 0.020 ± 0.006 |
| Count-cond    | 0.690 ± 0.035 | 0.595 ± 0.031 | 0.498 ± 0.052 | **0.369** ± 0.074 | 0.012 ± 0.003 |
| Flatten-sort  | 0.638 ± 0.020 | 0.542 ± 0.010 | **0.728** ± 0.046 | 0.187 ± 0.035 | **0.109** ± 0.013 |
| Summary       | 0.639 ± 0.028 | 0.566 ± 0.062 | 0.676 ± 0.033 | 0.072 ± 0.072 | 0.076 ± 0.040 |

A three-way specialization: LesionDETR is the side-level specialist (lowest
variance), the count-conditioned head reaches the highest common-lesion mAP, and
the flatten-and-sort head is the only one that clears the rare solid-lesion noise
floor.

## Table 4 — Encoder comparison

L3-trained LesionDETR, kidney-mask input, L1+L2 hierarchical supervision. Bilateral
AUC (↑); *Size* is max-lesion MAE in cm (↓). Mean ± std over 3 seeds.

| Encoder | Abn AUC | Cyst AUC | Mean | Size (cm) ↓ |
|---|---|---|---|---|
| **SuPreM** (abdominal, supervised) | **0.768 ± 0.027** | **0.773 ± 0.027** | **0.771** | **1.27 ± 0.22** |
| VoCo (160K CT, self-supervised)    | 0.532 ± 0.011 | 0.528 ± 0.012 | 0.530 | 1.50 ± 0.13 |
| SwinUNETR (from scratch)           | 0.541 ± 0.055 | 0.560 ± 0.036 | 0.551 | 1.45 ± 0.02 |

SuPreM leads every encoder by ~0.11 AUC over from-scratch; VoCo is
indistinguishable from random initialization — task-aligned (abdominal)
pretraining matters more than corpus size.

---

See the paper for the full tables, hierarchical-supervision ablation, scaling-law
analysis, linear-probing study, and the synthetic validation of the per-lesion AP
metric.
