# Inference

The project provides two inference entrypoints: a Gradio demo for interactive use and a CLI for
single-image batchable workflows.

## Installation

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Install the model-specific framework dependencies only when you need that model:

```bash
python -m pip install -r requirements/gan-mso2.txt
python -m pip install -r requirements/unet-resnet50.txt
```

The `requirements/legacy-all.txt` file preserves the original pinned dependency set. It may require
an older Python environment than the current development setup.

## Gradio Demo

```bash
python app.py
```

Inputs:

- Input chest X-ray image.
- Model key selected from the registry.
- Local checkpoint path.
- Number of iterative inference steps.

Output:

- A bone-suppressed image as an RGB array rendered by Gradio.

## CLI

```bash
bone-suppression \
  --model gan_mso2 \
  --checkpoint models/checkpoints/gan_mso2.h5 \
  --input path/to/chest-xray.png \
  --output outputs/bone-suppressed.png \
  --steps 2
```

The CLI reads the input image as RGB, runs the selected model, and writes the generated image to the
requested output path.

## Troubleshooting

- `Checkpoint not found`: verify the local path and keep weights outside Git.
- `Install the TensorFlow requirements`: install `requirements/gan-mso2.txt`.
- `Install the U-Net/FastAI requirements`: install `requirements/unet-resnet50.txt`.
- `Unknown model`: check valid keys in `configs/model_registry.json`.
- U-Net checkpoint unavailable: restore a valid checkpoint link before claiming reproducible U-Net
  inference.
