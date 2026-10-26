# RSNA Knee Abnormality Detection: frozen DINOv2 features with a weak-label ensemble

A Kaggle code-competition pipeline that predicts 12 knee MRI findings (ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture) from DICOM studies, scored by macro-averaged ROC-AUC.

The central difficulty is label scarcity: only **58 of 4,407** training studies have direct labels. Everything else has a free-text, multilingual radiology report. The pipeline combines a frozen self-supervised vision backbone with weak labels derived from those reports, and ensembles several ways of training the head.

| | Public leaderboard |
|---|---|
| Initial two-notebook pipeline | ~0.602 |
| Single-notebook ensemble (this approach) | **~0.700** |

Details and caveats are in [docs/experiments.md](docs/experiments.md).

## Approach

```
DICOM studies
  -> one series per plane (fluid-sensitive preferred), slices ordered by physical position
  -> 3 slices/plane at fixed quantiles, cropped to 120 mm FOV, resized to 224 px
  -> axial/coronal mirrored to a canonical knee side
  -> frozen DINOv2 ViT-S/14 features (mean over slices, concat over planes) -> 1152-d vector
  -> weak labels from reports (TF-IDF classifier trained on the 58 gold studies)
  -> report-grouped, multi-label-stratified 5-fold split
  -> two-phase MLP head (weak+gold warm-up, then gold-only)
  -> ensemble: frozen heads + last-block fine-tuned backbones + LoRA backbones
  -> per-finding rank-normalised predictions -> submission.csv
```

Main design decisions, each backed by a leave-one-out experiment on the gold studies:

- **Frozen backbone + small head.** 58 labels cannot train a vision model; caching features also makes head experiments take seconds.
- **TF-IDF weak labels** over keyword rules (0.664 vs 0.583 macro-AUC). Keyword rules match only 29.3% of (study, finding) pairs.
- **Two-phase head training.** Training on weak and gold targets jointly lets the weak-label loss dominate. Warm-up then gold-only fine-tune reaches 0.658 vs 0.609 for a linear probe.
- **Leakage-free stratified folds.** Folds are grouped by report text (duplicates stay together), and gold studies are assigned with multi-label iterative stratification so every finding appears in every fold.
- **Laterality canonicalisation** for axial and coronal planes. Its measured effect is within noise; it is kept because the mechanism is physical and it is harmless.
- **Backbone adaptation as extra ensemble members, weighted below the frozen group.** In validation neither last-block fine-tuning (0.644) nor LoRA (0.619) beat the frozen head (0.664), so they contribute diversity rather than being trusted outright.

## Repository layout

```
notebooks/
  01-eda.ipynb          dataset exploration (DICOM structure, labels, reports)
  02-rsna-knee-dinov2-pipeline.ipynb       the full training + inference pipeline (with outputs)
docs/
  experiments.md                        experiment tables and caveats
requirements.txt
```

## Running it

The competition data is not included; obtain it from the Kaggle competition page and accept its terms. The pipeline notebook is designed to run on a Kaggle GPU notebook and takes about 1.5-2 hours end to end.

1. Attach the competition data to a Kaggle notebook and import `notebooks/rsna-knee-dinov2-pipeline.ipynb`.
2. **First run, internet on:** set `CONFIG["LOCAL_BACKBONE_CHECKPOINT"] = None` and `CONFIG["PRETRAINED"] = True`. The notebook downloads DINOv2 through `timm` and writes `/kaggle/working/dinov2_backbone.pth`.
3. Upload that file as a Kaggle dataset (for example `dinov2-weights`), attach it, and set `CONFIG["LOCAL_BACKBONE_CHECKPOINT"]` to its path. The default is `/kaggle/input/dinov2-weights/dinov2_backbone.pth`.
4. **Scored run, internet off:** the notebook now loads weights from disk, trains the whole ensemble and writes `/kaggle/working/submission.csv`. Nothing else needs to be attached.

Local runs with a different data path work by editing the paths in the configuration and `find_competition_root`.

## Configuration

Everything tunable is in one `CONFIG` block. The settings most worth knowing:

| Setting | Default | Effect |
|---|---|---|
| `SLICES_PER_PLANE`, `IMG_SIZE`, `CROP_MM` | 3, 224, 120 | Input to the backbone; changing them rebuilds the caches |
| `LATERALITY_CORRECTION` | `True` | Mirror axial/coronal images to one knee side |
| `N_TRAINABLE_BACKBONE_BLOCKS` | 1 | Blocks unfrozen in fine-tuned members |
| `FINETUNE_SEEDS`, `LORA_SEEDS`, `EXTRA_FROZEN_HEAD_SEEDS` | `[1,2]`, `[1,2]`, `[11,12]` | Seeds per fold in each ensemble group |
| `INCLUDE_BACKBONE_FINETUNE_IN_SUBMISSION`, `INCLUDE_LORA_MEMBERS` | `True` | Set both `False` for a fast frozen-only run |
| `AUGMENT_FINETUNING` | `True` | Light augmentation for fine-tuned and LoRA members |
| `ENSEMBLE_GROUP_WEIGHTS` | 0.5 / 0.25 / 0.25 | Frozen / fine-tuned / LoRA |
| `ENSEMBLE_TIME_BUDGET_SEC` | 6 h | Optional members are skipped past this, so a slow run cannot time out |

## Limitations

- Validation rests on 58 labelled studies; differences of about 0.01-0.02 macro-AUC are within noise.
- The leaderboard gain came from many simultaneous changes and has not been attributed to any one. A frozen-only run would show how much the fine-tuned and LoRA members add.
- The ensemble weights are a hedge, not tuned.
- The sagittal plane is not laterality-corrected (the required sign convention varies by scanner and could not be verified).
- The pipeline depends on Kaggle paths (`/kaggle/input`, `/kaggle/working`).

## References

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Sechidis, Tsoumakas and Vlahavas, *On the Stratification of Multi-Label Data*, 2011.
