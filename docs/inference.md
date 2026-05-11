# Inference

The project provides two inference entrypoints: a Gradio demo for interactive use and a CLI for
single-image batchable workflows. Both support `auto` and CPU-only loading.

## Installation

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Install model-specific framework dependencies only when you need that model:

```bash
python -m pip install -r requirements/gan-mso2.txt
python -m pip install -r requirements/unet-resnet50.txt
```

## Gradio Demo

```bash
python app.py
```

Inputs:

- Input chest X-ray image.
- Model key selected from the registry.
- Local checkpoint path.
- Number of iterative inference steps.
- Device: `auto` or `cpu`.

Output:

- A bone-suppressed image as an RGB array rendered by Gradio.

## CLI

```bash
bone-suppression \
  --model gan_mso2 \
  --checkpoint models/checkpoints/gan_mso2_retrained_v1.keras \
  --input path/to/chest-xray.png \
  --output outputs/bone-suppressed.png \
  --steps 2 \
  --device cpu
```

The CLI reads the input image as RGB, runs the selected model, and writes the generated image to the
requested output path. Use `--device cpu` for reproducible CPU-only validation and timing.

## Custom Registry Path

By default, the package reads `configs/model_registry.json` in an editable checkout and falls back
to the packaged registry resource after normal installation. To test a custom registry, set:

```bash
BONE_SUPPRESSION_MODEL_REGISTRY=path/to/model_registry.json
```

## Troubleshooting

- `Checkpoint not found`: verify the local path and keep weights outside Git.
- `Install the TensorFlow requirements`: install `requirements/gan-mso2.txt`.
- `Install the U-Net/FastAI requirements`: install `requirements/unet-resnet50.txt`.
- `Unknown model`: check valid keys in `configs/model_registry.json`.
- Empty model dropdown: registry entries remain unavailable until public checkpoint URLs and
  SHA256 values are filled after retraining.
