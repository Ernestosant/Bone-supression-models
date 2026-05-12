# Results

Retrained-v1 training and metrics were generated on Kaggle Tesla P100. Primary quantitative
results use GPU-enabled `device=auto` evaluation for speed. CPU inference remains supported for
end users, but CPU timing is not the primary results path.

## Current Status

Corrected MSO retraining has now completed for both model families. The current citable metrics use
the historical notebook preprocessing path: OpenCV 8-bit read, `255 - image`,
`cv2.equalizeHist`, and RGB expansion. The earlier retrained-v1 pass remains documented below as
superseded engineering provenance because it did not match that preprocessing path.

Step `0` is a no-model baseline. It is useful for context, but model selection should use steps
`1..5`, where an actual checkpoint is run autoregressively.

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
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-gan-mso-corrected-v2`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-gan-mso-corrected-v2) | 1 | complete | GAN 50-epoch corrected MSO run completed. Train time: 2613.72 s; run time: 2729.91 s. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-unet-mso-corrected-v2`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-unet-mso-corrected-v2) | 1 | error | U-Net failed in epoch 1 on Kaggle P100 inside FastAI self-attention/spectral norm with `CUBLAS_STATUS_NOT_SUPPORTED`; rerun disables self-attention for P100 compatibility. |
| 2026-05-11 | [`ernestosantiesteban/bone-suppression-unet-mso-corrected-v3`](https://www.kaggle.com/code/ernestosantiesteban/bone-suppression-unet-mso-corrected-v3) | 1 | complete | U-Net 50-epoch corrected MSO run completed with self-attention disabled for P100 compatibility. Train time: 2270.66 s; run time: 2592.20 s. |

## Corrected MSO Metrics

These are the current retrained-v1 results. The selected rows are the best measured inference step
for each model among steps `1..5`; step `0` is excluded from selection because it does not run a
model.

| Model | Selected step | MAE | RMSE | PSNR | SSIM | GPU sec/image | SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pix2Pix GAN MSO2 corrected | 1 | 0.020027 | 0.042244 | 27.802648 | 0.989218 | 0.097549 | `09525519d3c51d6c7fd0377634bdff4f39ddf314180ea94b9d56fdcd49829dc1` |
| U-Net ResNet50 corrected | 1 | 0.076602 | 0.092698 | 20.675888 | 0.950540 | 0.737866 | `2c2c1d9c728c326608d6bc16123b01f9c23c819b9ab70b30f33a989fd3ca010b` |

### Pix2Pix GAN MSO2 Corrected

| Steps | MAE | RMSE | PSNR | SSIM | GPU sec/image |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.018665 | 0.036400 | 29.562923 | 0.991440 | 0.054963 |
| 1 | 0.020027 | 0.042244 | 27.802648 | 0.989218 | 0.097549 |
| 2 | 0.024498 | 0.047357 | 26.740057 | 0.986480 | 0.161409 |
| 3 | 0.029965 | 0.054304 | 25.496492 | 0.982401 | 0.233899 |
| 4 | 0.035239 | 0.061502 | 24.378221 | 0.977592 | 0.317723 |
| 5 | 0.040321 | 0.068837 | 23.372787 | 0.972074 | 0.394302 |

### U-Net ResNet50 Corrected

| Steps | MAE | RMSE | PSNR | SSIM | GPU sec/image |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.018665 | 0.036400 | 29.562923 | 0.991440 | 0.056046 |
| 1 | 0.076602 | 0.092698 | 20.675888 | 0.950540 | 0.737866 |
| 2 | 0.082528 | 0.098987 | 20.096765 | 0.942528 | 1.014697 |
| 3 | 0.084564 | 0.100788 | 19.939899 | 0.940305 | 1.286333 |
| 4 | 0.086183 | 0.102270 | 19.813140 | 0.938321 | 1.583577 |
| 5 | 0.087519 | 0.103564 | 19.703887 | 0.936539 | 1.835310 |

## Superseded Metrics

These values were computed before restoring notebook-compatible preprocessing. They are retained to
explain the failed visual QA pass, not as final model evidence.

| Model | Selected step | MAE | RMSE | PSNR | SSIM | GPU sec/image | SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pix2Pix GAN MSO2 Retrained v1 | 1 | 0.011330 | 0.046331 | 27.456801 | 0.749001 | 0.108497 | `e6d72033d643f378fcd7722d630abf935d590831da5911c8a0f0deaa31f375bd` |
| U-Net ResNet50 Retrained v1 | 1 | 0.003033 | 0.034316 | 36.057932 | 0.874740 | 0.635685 | `0a4be7cea81f7c8013b46ad276e8dd92b5bbd8e9689754faddd293621b044b3a` |

## Superseded Autoregressive Step Metrics

The tables below are from the superseded preprocessing-mismatch pass and are kept only so the
failed visual QA path is traceable.

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
