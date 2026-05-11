# Models

Model metadata is centralized in `configs/model_registry.json`. The registry is the source of
truth for framework type, expected checkpoint filename, default inference steps, and availability.

## `gan_mso2`

- Framework: TensorFlow/Keras.
- Architecture: Pix2Pix-style conditional GAN.
- Expected checkpoint: `gan_mso2.h5`.
- Status: the Google Drive link opens, but the file should be downloaded and validated before
  publishing generated examples or metrics.
- Default inference steps: `2`.

The inference routine applies histogram equalization, resizes large inputs to 256 x 256, normalizes
to `[-1, 1]`, calls the generator with `training=True` for compatibility with the original code, and
converts the output back to an 8-bit RGB image.

## `unet_resnet50`

- Framework: FastAI.
- Architecture: U-Net with a pretrained ResNet50 encoder and single-channel attention component.
- Expected checkpoint: `unet_resnet50.pkl`.
- Status: pending. The historical Google Drive URL currently returns `404`.
- Default inference steps: `2`.

The model is retained in the registry so the project can support it again once a valid checkpoint is
available. Until then, README examples and reproducibility claims should not rely on this model.

## Checkpoint Policy

Do not commit model weights to Git. Store downloaded checkpoints in `models/checkpoints/` or another
local path and pass the path to the CLI or Gradio app. The repository `.gitignore` excludes common
checkpoint formats such as `.h5`, `.pkl`, `.pt`, `.pth`, and `.ckpt`.

## Known Gaps

- No quantitative evaluation metrics are currently included.
- Training notebooks are referenced historically but are not present in the repository.
- The model registry records availability, not checksum verification. Add hashes before using this
  project for strict reproducibility.
