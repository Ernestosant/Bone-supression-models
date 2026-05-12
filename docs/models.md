# Models

Model metadata is centralized in `configs/model_registry.json` and packaged in
`src/bone_suppression/resources/model_registry.json`.

## Registry Fields

Each model entry records framework, architecture, availability, checkpoint filename, checkpoint
policy, SHA256, device support, example panels, metrics, preprocessing, artifact paths, and training
artifact URL.

`available` is `true` for the corrected MSO release because public checkpoint assets are available
and their SHA256 values are recorded in the registry. Large weights remain outside Git history.

## `gan_mso2`

- Framework: TensorFlow/Keras.
- Architecture: Pix2Pix-style conditional GAN.
- Retrained-v1 checkpoint: `gan_mso2_retrained_v1.keras`.
- Device support: CPU and GPU.
- Current status: corrected MSO retrain complete; public checkpoint available.

The inference routine applies the historical MSO preprocessing (`255 - image` and histogram
equalization), resizes large inputs to 256 x 256, normalizes to `[-1, 1]`, calls the generator with
`training=True` for compatibility, and converts the output back to an 8-bit RGB image.

## `unet_resnet50`

- Framework: FastAI.
- Architecture: U-Net with a pretrained ResNet50 encoder.
- Retrained-v1 checkpoint: `unet_resnet50_retrained_v1.pkl`.
- Device support: CPU and GPU.
- Current status: corrected MSO retrain complete; public checkpoint available.

The historical U-Net checkpoint URL returned 404 during review. The retrained-v1 checkpoint should
be presented as a new reproducible artifact, not as a recovered historical weight.
The Kaggle P100 run disables FastAI self-attention because spectral-normalized self-attention failed
on that GPU with `CUBLAS_STATUS_NOT_SUPPORTED`.

## Checkpoint Policy

Do not commit model weights to Git. Store downloaded checkpoints in `models/checkpoints/` or another
local path and pass the path to the CLI or Gradio app. The repository `.gitignore` excludes common
checkpoint formats such as `.h5`, `.keras`, `.pkl`, `.pt`, `.pth`, and `.ckpt`.

Current retrained-v1 checkpoints are published in the
[`corrected-mso-v1`](https://github.com/Ernestosant/Bone-supression-models/releases/tag/corrected-mso-v1)
GitHub Release. The GAN is a single `.keras` asset. The corrected U-Net `.pkl` is larger, so it is
published as `part01`, `part02`, and `part03`; concatenate the parts in order and verify the
reconstructed file against the registry SHA256.

Every published retrained-v1 checkpoint has:

- A public GitHub Release URL.
- A SHA256 checksum in the registry.
- Manifests, metrics, checksums, and example panels in the release artifact bundle.
- GPU holdout metrics in `docs/results.md`; optional CPU checks are for user-facing inference
  support.
- Example panels in `docs/assets/examples/retrained_v1/`.
