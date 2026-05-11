# Training

This repository now includes a reproducible retraining workflow for the two documented model
families. The intended full run is Kaggle with GPU enabled, using the public Kaggle dataset
`hmchuong/xray-bone-shadow-supression`.

## Kaggle P100 Run

1. Open `notebooks/kaggle_retrain_bone_suppression.ipynb` in Kaggle.
2. Attach the `hmchuong/xray-bone-shadow-supression` dataset.
3. Enable GPU acceleration.
4. Run all cells.

For U-Net ResNet50 on Tesla P100, Kaggle's default PyTorch CUDA 12.8 wheel may fail with
`cudaErrorNoKernelImageForDevice` because it does not include `sm_60` kernels. The monitored
Kaggle runs therefore install PyTorch CUDA 12.6 before importing FastAI. This is recorded in the
run `environment.json` and `pip_freeze` artifacts.

The notebook writes all generated artifacts under:

```text
/kaggle/working/training_runs/retrained_v1/
```

Expected outputs:

- `splits.json`: deterministic train/validation/test split with seed `2026`.
- `gan_mso2/gan_mso2_retrained_v1.keras`.
- `unet_resnet50/unet_resnet50_retrained_v1.pkl`.
- `*_manifest.json`: checkpoint SHA256, seed, commit, environment, and hyperparameters.
- `evaluation/*_test_metrics.json`.
- `metrics.json`.
- `examples/*_comparison.png`.

For monitored long runs, U-Net is split into two Kaggle jobs:

- `BONE_SUPPRESSION_RUN_MODE=train_only` exports `unet_resnet50_retrained_v1.pkl` and its manifest.
- `BONE_SUPPRESSION_RUN_MODE=eval_only` evaluates the exported checkpoint for autoregressive steps
  `0,1,2,3,4,5` with `BONE_SUPPRESSION_EVAL_DEVICE=auto` and GPU enabled.

CPU inference is kept for user-facing portability, but the primary reported metrics are generated
on GPU to keep evaluation turnaround short.

## Local Commands

Create the split:

```bash
bone-suppression-repro prepare-splits \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --output training_runs/retrained_v1/splits.json \
  --seed 2026
```

Train the GAN:

```bash
bone-suppression-repro train \
  --model gan_mso2 \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --splits training_runs/retrained_v1/splits.json \
  --output-dir training_runs/retrained_v1/gan_mso2 \
  --epochs 50 \
  --batch-size 4 \
  --image-size 256
```

Train the U-Net:

```bash
bone-suppression-repro train \
  --model unet_resnet50 \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --splits training_runs/retrained_v1/splits.json \
  --output-dir training_runs/retrained_v1/unet_resnet50 \
  --epochs 50 \
  --batch-size 4 \
  --image-size 256
```

Use `--limit 4 --epochs 1` for a smoke test on a small local fixture or a tiny subset.

## Data Policy

The Kaggle `augmented/` directory is intentionally not used for primary metrics. The split is built
from non-augmented `JSRT/JSRT` inputs paired by filename with `BSE_JSRT/BSE_JSRT` targets. Any
augmentation used by the training code is applied only to the training split.
