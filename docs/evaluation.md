# Evaluation

Evaluation is performed on the deterministic holdout split created by:

```bash
bone-suppression-repro prepare-splits --seed 2026
```

The default split is 70% train, 15% validation, and 15% test over paired non-augmented JSRT/BSE_JSRT
images. Test metrics must not use the Kaggle `augmented/` directory.

## Autoregressive Steps

Both retrained models are evaluated with autoregressive step counts `0, 1, 2, 3, 4, 5`.

- Step 0 is the no-model baseline: the input image resized to 256 x 256.
- Step 1 runs the model once.
- Steps 2-5 feed each output back as the next input and calculate metrics on the final output.

This is especially important for the U-Net workflow because the intended inference path is
autoregressive.

## Metrics

The evaluation CLI reports:

- `mae`: mean absolute error on normalized `[0, 1]` pixels.
- `rmse`: root mean squared error on normalized `[0, 1]` pixels.
- `psnr`: peak signal-to-noise ratio in dB.
- `ssim`: global luminance SSIM.
- `inference_seconds_per_image`: mean end-to-end inference time per test image when evaluation runs
  with `device=auto` on Kaggle GPU.
- `cpu_seconds_per_image`: optional CPU inference timing for user-facing compatibility checks.

Generated predictions are compared against targets resized to the prediction resolution. This keeps
the metric contract stable for 256 x 256 retrained models while preserving the original dataset files.

## Commands

Evaluate autoregressive steps for the GAN checkpoint on the fastest available device:

```bash
bone-suppression-repro evaluate-steps \
  --model gan_mso2 \
  --checkpoint training_runs/retrained_v1/gan_mso2/gan_mso2_retrained_v1.keras \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --splits training_runs/retrained_v1/splits.json \
  --output-dir training_runs/retrained_v1/evaluation \
  --device auto \
  --steps 0,1,2,3,4,5
```

Evaluate autoregressive steps for the U-Net checkpoint on the fastest available device:

```bash
bone-suppression-repro evaluate-steps \
  --model unet_resnet50 \
  --checkpoint training_runs/retrained_v1/unet_resnet50/unet_resnet50_retrained_v1.pkl \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --splits training_runs/retrained_v1/splits.json \
  --output-dir training_runs/retrained_v1/evaluation \
  --device auto \
  --steps 0,1,2,3,4,5
```

Merge model-level metrics and generate fixed visual panels:

```bash
bone-suppression-repro merge-metrics \
  --output training_runs/retrained_v1/metrics.json \
  training_runs/retrained_v1/evaluation/gan_mso2_test_metrics.json \
  training_runs/retrained_v1/evaluation/unet_resnet50_test_metrics.json

bone-suppression-repro comparison-examples \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --splits training_runs/retrained_v1/splits.json \
  --output-dir training_runs/retrained_v1/examples \
  --gan-predictions training_runs/retrained_v1/evaluation/predictions/gan_mso2 \
  --unet-predictions training_runs/retrained_v1/evaluation/predictions/unet_resnet50 \
  --count 3
```
