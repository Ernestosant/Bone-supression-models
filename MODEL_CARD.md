# Model Card: Bone Suppression Models

## Overview

This repository contains research tooling for chest X-ray bone suppression, framed as
image-to-image translation from a frontal chest radiograph to a synthetic bone-suppressed image.

- Task: chest X-ray bone suppression / image-to-image translation.
- Input: frontal chest X-ray image.
- Output: synthetic image with rib and clavicle contrast reduced.
- Models: Pix2Pix GAN MSO2 (`gan_mso2`) and U-Net ResNet50 (`unet_resnet50`).
- Dataset: Kaggle Chest Xray Bone Shadow Suppression
  (`hmchuong/xray-bone-shadow-supression`), using paired JSRT and BSE_JSRT images.
- Metrics: MAE, RMSE, PSNR, and SSIM on a deterministic test split.

## Intended Use

These models are intended for reproducible research, engineering review, and educational
experimentation around bone-suppressed chest radiograph synthesis. The repository is structured to
make preprocessing, deterministic splitting, training, evaluation, manifests, checksums, and example
panels auditable.

The outputs should be interpreted as synthetic images for research workflows. They are not validated
as medical devices and should not be used for clinical diagnosis, triage, treatment planning, or
replacement of radiologist review.

## Dataset And Preprocessing

Training and evaluation use the public Kaggle Chest Xray Bone Shadow Suppression dataset. Dataset
files are not redistributed in this repository. The primary split pairs non-augmented `JSRT/JSRT`
inputs with `BSE_JSRT/BSE_JSRT` targets and uses a deterministic 70/15/15 split with seed `2026`.

The corrected retraining path reproduces the historical notebook MSO preprocessing:

- OpenCV default 8-bit image read.
- Intensity inversion with `255 - image`.
- Grayscale histogram equalization.
- RGB expansion.

## Evaluation

Reported metrics are measured on the deterministic test holdout. Step `0` is a no-model input
baseline and is excluded from model selection; the selected operating point for both current models
is autoregressive step `1`.

| Model | Step | MAE | RMSE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pix2Pix GAN MSO2 corrected | 1 | 0.020027 | 0.042244 | 27.802648 | 0.989218 |
| U-Net ResNet50 corrected | 1 | 0.076602 | 0.092698 | 20.675888 | 0.950540 |

Detailed step metrics, manifests, SHA256 values, run summaries, and visual panels are documented in
[`docs/results.md`](docs/results.md), [`docs/training.md`](docs/training.md), and
[`configs/model_registry.json`](configs/model_registry.json).

## Checkpoint Availability

Checkpoints are not redistributed due to size/storage constraints; metrics, hashes, manifests,
scripts, visual panels, and reproducibility instructions are provided.

Local checkpoint filenames and SHA256 values are recorded in the model registry:

| Model key | Expected file | SHA256 |
| --- | --- | --- |
| `gan_mso2` | `gan_mso2_retrained_v1.keras` | `09525519d3c51d6c7fd0377634bdff4f39ddf314180ea94b9d56fdcd49829dc1` |
| `unet_resnet50` | `unet_resnet50_retrained_v1.pkl` | `2c2c1d9c728c326608d6bc16123b01f9c23c819b9ab70b30f33a989fd3ca010b` |

If public artifact hosting is added later, the registry should be updated with verified direct
download links, `available` should be set to `true`, and the SHA256 values should be rechecked
against the published files.

## Limitations

- The dataset is small and focused on frontal chest radiographs, so performance may not generalize
  to other scanners, institutions, acquisition protocols, projections, or patient populations.
- Bone-suppressed outputs are synthetic and may alter, remove, hallucinate, or de-emphasize image
  features that could matter clinically.
- Metrics such as MAE, RMSE, PSNR, and SSIM measure similarity to paired BSE targets, not diagnostic
  safety or clinical utility.
- Autoregressive multi-step inference can amplify artifacts; the documented operating point is
  selected from measured holdout results.
- The U-Net ResNet50 run disables FastAI self-attention for Kaggle P100 compatibility, so it is not
  an exact reproduction of every historical notebook architecture flag.

## Responsible Use

Use these models only in research or educational settings with clear labeling that outputs are
synthetic. Do not present generated images as original radiographs, use them as sole evidence for a
medical conclusion, or deploy them in patient-facing workflows without appropriate validation,
regulatory review, privacy review, and clinical oversight.

When sharing results, include the dataset source, preprocessing path, split seed, selected inference
step, metric table, checkpoint SHA256, and the fact that checkpoint files are not redistributed in
this repository.
