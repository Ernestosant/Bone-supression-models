# Results

Retrained-v1 training and metrics were generated on Kaggle Tesla P100. Primary quantitative
results use GPU-enabled `device=auto` evaluation for speed. CPU inference remains supported for
end users, but CPU timing is not the primary results path.

## Current Status

The first retrained-v1 checkpoint pass is now marked **superseded**. Review of the historical
`Unet_MSO.ipynb` and `pix2pix.ipynb` notebooks showed that the original experiments trained on
OpenCV 8-bit images after `255 - image` and `cv2.equalizeHist`. The first pass used raw
Pillow/FastAI loading for parts of training/evaluation, which is wrong for the 16-bit BSE targets
and explains the poor visual outputs despite apparently strong numeric metrics.

Do not cite the tables below as final research results. They remain here only as engineering
provenance until both models are retrained and reevaluated with the restored notebook-compatible
preprocessing path.

## Kaggle Run Log

| Date | Kernel | Version | Status | Notes |
| --- | --- | ---: | --- | --- |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-retrained-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-retrained-v1) | 1 | error | Failed before training because Kaggle did not include the local `src/` package. Superseded by embedded pilot/full kernels below. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-pilot-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-pilot-v1) | 1 | error | Failed before training because Kaggle did not include the local `src/` package. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-pilot-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-pilot-v1) | 2 | complete | GAN 1-epoch pilot completed on Tesla P100. Epoch time: 66.59 s; train time: 70.67 s; run time including CPU evaluation: 97.70 s. Estimated 50-epoch GAN training: about 56-60 min. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-full-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-full-v1) | 1 | superseded | GAN 50-epoch run completed on Tesla P100, but later review found the preprocessing path did not match the historical notebooks. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-steps-eval-v2`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-steps-eval-v2) | 1 | superseded | GAN CPU autoregressive evaluation for steps 0-5 completed before the preprocessing mismatch was found. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-steps-gpu-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-steps-gpu-v1) | 1 | superseded | GAN GPU autoregressive evaluation for steps 0-5 completed before the preprocessing mismatch was found. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-u-net-pilot-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-u-net-pilot-v1) | 1 | error | Failed at first FastAI batch because Kaggle PyTorch `2.10.0+cu128` does not include kernels for Tesla P100 `sm_60`. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-u-net-pilot-fast-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-u-net-pilot-fast-v1) | 1 | complete | U-Net 1-epoch pilot completed after installing PyTorch CUDA 12.6. Train time: 79.04 s; run time after setup with limited CPU eval: 114.92 s. Estimated 50-epoch train time: about 66 min, excluding setup and full CPU step evaluation. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-unet-full-train-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-unet-full-train-v1) | 1 | superseded | U-Net 50-epoch train-only run completed, but later review found the preprocessing path did not match the historical notebooks. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-unet-steps-gpu-v1`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-unet-steps-gpu-v1) | 1 | superseded | U-Net GPU autoregressive evaluation for steps 0-5 completed, but later review found the preprocessing path did not match the historical notebooks. |

## Superseded Metrics

These values were computed before restoring notebook-compatible preprocessing. They are retained to
explain the failed visual QA pass, not as final model evidence.

| Model | Selected step | MAE | RMSE | PSNR | SSIM | GPU sec/image | SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pix2Pix GAN MSO2 Retrained v1 | 1 | 0.011330 | 0.046331 | 27.456801 | 0.749001 | 0.108497 | `e6d72033d643f378fcd7722d630abf935d590831da5911c8a0f0deaa31f375bd` |
| U-Net ResNet50 Retrained v1 | 1 | 0.003033 | 0.034316 | 36.057932 | 0.874740 | 0.635685 | `0a4be7cea81f7c8013b46ad276e8dd92b5bbd8e9689754faddd293621b044b3a` |

## Autoregressive Step Metrics

Step 0 is a no-model baseline: the input image resized to 256 x 256 and compared to the paired
bone-suppressed target resized to the same shape. Steps 1-5 feed each model output back as the next
input and compute metrics on the final output for that step count. The tables below are GPU
evaluations with `device=auto` on Kaggle P100.

### Pix2Pix GAN MSO2 Retrained v1

| Steps | MAE | RMSE | PSNR | SSIM | GPU sec/image |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.754264 | 0.778947 | 2.186525 | 0.039314 | 0.037517 |
| 1 | 0.011330 | 0.046331 | 27.456801 | 0.749001 | 0.108497 |
| 2 | 0.356173 | 0.422186 | 7.507776 | -0.051687 | 0.122114 |
| 3 | 0.219088 | 0.333024 | 10.160235 | -0.007659 | 0.176024 |
| 4 | 0.172268 | 0.267603 | 13.262484 | 0.031722 | 0.228141 |
| 5 | 0.289207 | 0.394774 | 9.119546 | -0.021601 | 0.285705 |

### U-Net ResNet50 Retrained v1

| Steps | MAE | RMSE | PSNR | SSIM | GPU sec/image |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.754264 | 0.778947 | 2.186524 | 0.039314 | 0.038345 |
| 1 | 0.003033 | 0.034316 | 36.057932 | 0.874740 | 0.635685 |
| 2 | 0.003488 | 0.039873 | 37.566404 | 0.826337 | 0.879797 |
| 3 | 0.004944 | 0.050877 | 35.970157 | 0.685148 | 1.140543 |
| 4 | 0.005655 | 0.055698 | 35.375760 | 0.603625 | 1.401692 |
| 5 | 0.006114 | 0.058591 | 35.048709 | 0.546010 | 1.674331 |

## Publication Checklist

Remaining publication tasks:

1. Upload checkpoints, manifests, metrics, and panels to Google Drive.
2. Make Drive files public/readable and copy direct download links into `configs/model_registry.json`
   and `src/bone_suppression/resources/model_registry.json`.
