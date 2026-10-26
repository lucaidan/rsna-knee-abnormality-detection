# Experiments

Evidence behind the design choices in the pipeline notebook.

## Protocol

Unless stated otherwise, every number is a **pooled leave-one-out macro-AUC over the 58 gold-labelled studies**: each gold study is held out in turn, a model is trained on the other 57 plus the weakly labelled training pool, all 58 out-of-fold predictions are pooled, and the per-finding ROC-AUCs are averaged over the 12 findings. Leave-one-out was chosen over 5-fold because, at this sample size, which studies land in which fold is itself a large source of noise.

Caveats that apply to everything below:

- 58 studies with single-digit positives for some findings makes every estimate noisy. The same nominal configuration (last-block fine-tuning, seed 1) scored 0.644 in one run and 0.652 in another, so differences of roughly 0.01-0.02 should not be treated as real.
- Experiments were run in a separate research notebook that is not part of this repository; the pipeline notebook contains only the resulting decisions.

## 1. Laterality correction

Logistic-regression probe on frozen features, gold studies only. Side-dependent findings are the five defined relative to the body midline; the other seven act as controls.

| Finding | Side-dependent | Without | With | Change |
|---|---|---|---|---|
| Lateral Meniscus | yes | 0.640 | 0.651 | +0.011 |
| Lateral OA | yes | 0.638 | 0.677 | +0.039 |
| MCL | yes | 0.456 | 0.438 | -0.018 |
| Medial Meniscus | yes | 0.552 | 0.548 | -0.004 |
| Medial OA | yes | 0.783 | 0.783 | 0.000 |
| ACL | no | 0.513 | 0.512 | -0.001 |
| PF OA | no | 0.726 | 0.732 | +0.006 |
| Effusion | no | 0.727 | 0.752 | +0.025 |
| Synovitis | no | 0.523 | 0.527 | +0.004 |
| Baker's | no | 0.585 | 0.571 | -0.014 |
| Contusion | no | 0.713 | 0.722 | +0.009 |
| Fracture | no | 0.417 | 0.400 | -0.017 |

Side-dependent macro-AUC 0.614 -> 0.619; side-independent 0.601 -> 0.602. The expected signature (gain concentrated in side-dependent findings) is not clearly present. The correction is kept as physically motivated and harmless, not as a measured improvement.

## 2. Probes and the two-phase head

| Finding | Positives | Image probe (linear) | Text probe (TF-IDF) | Two-phase head |
|---|---|---|---|---|
| ACL | 24 | 0.523 | 0.608 | 0.612 |
| MCL | 9 | 0.435 | 0.361 | 0.467 |
| Medial Meniscus | 26 | 0.547 | 0.662 | 0.712 |
| Lateral Meniscus | 23 | 0.642 | 0.544 | 0.598 |
| Medial OA | 15 | 0.786 | 0.628 | 0.820 |
| Lateral OA | 11 | 0.681 | 0.569 | 0.665 |
| PF OA | 21 | 0.726 | 0.521 | 0.641 |
| Effusion | 35 | 0.745 | 0.545 | 0.799 |
| Synovitis | 27 | 0.532 | 0.522 | 0.532 |
| Baker's | 12 | 0.571 | 0.458 | 0.761 |
| Contusion | 19 | 0.717 | 0.613 | 0.698 |
| Fracture | 18 | 0.401 | 0.600 | 0.592 |
| **Macro** | | **0.609** | **0.553** | **0.658** |

The two-phase head (weak-plus-gold warm-up, then gold-only fine-tune) beats the linear probe on the same frozen features. A semantic sentence-embedding text probe scored 0.540 macro-AUC, below TF-IDF.

## 3. Weak-label source

Two-phase head trained with each label source, pooled leave-one-out:

| Finding | Keyword | TF-IDF | Stacked (TF-IDF + sentence embedding) |
|---|---|---|---|
| ACL | 0.511 | 0.630 | 0.603 |
| MCL | 0.510 | 0.501 | 0.424 |
| Medial Meniscus | 0.576 | 0.690 | 0.602 |
| Lateral Meniscus | 0.561 | 0.601 | 0.523 |
| Medial OA | 0.691 | 0.836 | 0.797 |
| Lateral OA | 0.660 | 0.675 | 0.516 |
| PF OA | 0.578 | 0.607 | 0.588 |
| Effusion | 0.545 | 0.789 | 0.648 |
| Synovitis | 0.431 | 0.535 | 0.524 |
| Baker's | 0.761 | 0.754 | 0.639 |
| Contusion | 0.580 | 0.700 | 0.667 |
| Fracture | 0.586 | 0.649 | 0.524 |
| **Macro** | **0.583** | **0.664** | **0.588** |

TF-IDF is adopted. It has full coverage (the keyword search matches 29.3% of pairs), needs no downloaded weights, and the gain is spread across most findings.

## 4. Fold stratification

Gold-positive studies per fold, before and after multi-label iterative stratification:

| Finding | Total | Before (f0..f4) | After (f0..f4) |
|---|---|---|---|
| ACL | 24 | 2 7 5 1 9 | 5 4 5 5 5 |
| MCL | 9 | 2 2 1 0 4 | 1 2 2 2 2 |
| Medial Meniscus | 26 | 1 6 4 3 12 | 4 5 5 5 7 |
| Lateral Meniscus | 23 | 1 5 2 3 12 | 5 5 5 4 4 |
| Medial OA | 15 | 2 3 1 2 7 | 3 3 3 3 3 |
| Lateral OA | 11 | 0 2 0 1 8 | 3 2 2 2 2 |
| PF OA | 21 | 3 3 0 7 8 | 5 4 4 4 4 |
| Effusion | 35 | 3 8 4 4 16 | 8 8 7 6 6 |
| Synovitis | 27 | 2 4 3 5 13 | 6 5 5 6 5 |
| Baker's | 12 | 0 3 1 1 7 | 3 2 3 2 2 |
| Contusion | 19 | 4 4 4 1 6 | 3 3 5 4 4 |
| Fracture | 18 | 5 4 2 1 6 | 3 3 4 4 4 |

## 5. Adapting the backbone

Pooled leave-one-out, seed 1. "Frozen" is the TF-IDF-label two-phase head; "last-block" unfreezes the final transformer block and norm (1.78M trainable parameters); "LoRA" is rank 8, alpha 16 on the attention projections of the last 4 blocks (73.7k parameters, 4.2% of the last-block count).

| Finding | Frozen | Last-block | LoRA |
|---|---|---|---|
| ACL | 0.630 | 0.650 | 0.638 |
| MCL | 0.501 | 0.447 | 0.422 |
| Medial Meniscus | 0.690 | 0.665 | 0.629 |
| Lateral Meniscus | 0.601 | 0.583 | 0.632 |
| Medial OA | 0.836 | 0.786 | 0.738 |
| Lateral OA | 0.675 | 0.652 | 0.683 |
| PF OA | 0.607 | 0.654 | 0.593 |
| Effusion | 0.789 | 0.802 | 0.778 |
| Synovitis | 0.535 | 0.527 | 0.456 |
| Baker's | 0.754 | 0.679 | 0.629 |
| Contusion | 0.700 | 0.709 | 0.659 |
| Fracture | 0.649 | 0.581 | 0.568 |
| **Macro** | **0.664** | **0.644** | **0.619** |

Neither adaptation beat the frozen head. An earlier comparison (numbers not retained) found that unfreezing two blocks overfit more than one, which is why one block is used.

### Augmentation

Last-block fine-tuning with and without light augmentation (rotation up to 5 degrees, translation 4%, intensity jitter 10%, noise sigma 0.02):

| Finding | No augmentation | Augmentation | Change |
|---|---|---|---|
| ACL | 0.661 | 0.674 | +0.013 |
| MCL | 0.463 | 0.465 | +0.002 |
| Medial Meniscus | 0.635 | 0.609 | -0.025 |
| Lateral Meniscus | 0.594 | 0.639 | +0.045 |
| Medial OA | 0.792 | 0.786 | -0.006 |
| Lateral OA | 0.648 | 0.646 | -0.002 |
| PF OA | 0.653 | 0.677 | +0.024 |
| Effusion | 0.750 | 0.770 | +0.020 |
| Synovitis | 0.521 | 0.583 | +0.062 |
| Baker's | 0.707 | 0.611 | -0.096 |
| Contusion | 0.784 | 0.764 | -0.020 |
| Fracture | 0.618 | 0.606 | -0.012 |
| **Macro** | **0.652** | **0.652** | **0.000** |

Six findings improved and six worsened, so the macro average hides real per-finding movement. Augmentation is included in the final ensemble but is not a demonstrated gain.

## 6. Leaderboard

| Version | Public score |
|---|---|
| Initial two-notebook pipeline (train notebook plus separate inference notebook) | ~0.60 |
| Single notebook, 5 frozen + 10 fine-tuned + 5 LoRA members, stratified folds, augmentation | 0.699 |
| Final notebook in this repo (15 frozen + 10 fine-tuned + 10 LoRA members) | not recorded |

Many things changed between the first two rows (stratified folds, a head-training fix, augmentation, LoRA members, ensemble weights, chunked test prediction), so the gain cannot be attributed to any one of them. Wall-clock time on a Kaggle GPU was about 1.5 h for the 0.699 version and 1.9 h for the final one.

## Not yet measured

- A frozen-only submission, to isolate what the fine-tuned and LoRA members contribute.
- Multi-seed repeats of the augmentation and LoRA comparisons and a direct test of the blended ensemble against frozen-only; the long run that would have produced them did not finish.
- Alternative backbones (a larger DINOv2, DINOv3), higher resolution, more slices per plane.
