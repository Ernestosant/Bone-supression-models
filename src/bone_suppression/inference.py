"""Reusable inference routines for the supported bone suppression models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from bone_suppression.model_io import load_model
from bone_suppression.preprocessing import (
    ensure_rgb,
    histogram_equalize_rgb,
    legacy_mso_preprocess,
    normalize_to_minus_one_one,
    output_to_uint8_image,
    resize_if_larger,
    to_uint8,
    validate_steps,
)
from bone_suppression.registry import ModelSpec, get_model_spec

try:  # pragma: no cover - fallback only matters when OpenCV is unavailable.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def run_inference(
    model_key: str,
    checkpoint_path: str | Path,
    image: np.ndarray,
    steps: int | None = None,
    device: str = "auto",
) -> np.ndarray:
    """Load a checkpoint and run inference on a single image."""
    spec = get_model_spec(model_key)
    model = load_model(spec, checkpoint_path, device=device)
    return predict_model(model, spec, image, steps=steps)


def predict_model(
    model,
    spec: str | ModelSpec,
    image: np.ndarray,
    steps: int | None = None,
) -> np.ndarray:
    """Run inference for a loaded model according to its registry spec."""
    model_spec = get_model_spec(spec) if isinstance(spec, str) else spec
    step_count = validate_steps(steps, model_spec.default_steps)

    if model_spec.framework == "fastai":
        return predict_unet_resnet50(model, image, steps=step_count)
    if model_spec.framework == "tensorflow":
        return predict_pix2pix_gan(model, image, steps=step_count)

    raise ValueError(f"Unsupported framework for inference: {model_spec.framework}")


def predict_pix2pix_gan(model, image: np.ndarray, steps: int = 2) -> np.ndarray:
    """Run Pix2Pix-style iterative inference."""
    current = legacy_mso_preprocess(image)
    prediction = current

    for _ in range(validate_steps(steps, 2)):
        model_input = resize_if_larger(current, target_size=(256, 256))
        model_input = normalize_to_minus_one_one(model_input)
        raw_output = model(model_input[np.newaxis, ...], training=True)
        prediction = _as_numpy(raw_output)[0] * 0.5 + 0.5
        prediction = to_uint8(np.clip(prediction, 0.0, 1.0))
        current = histogram_equalize_rgb(prediction)

    return ensure_rgb(prediction)


def predict_unet_resnet50(model, image: np.ndarray, steps: int = 2) -> np.ndarray:
    """Run FastAI U-Net inference using temporary files required by test_dl."""
    current = legacy_mso_preprocess(image)
    prediction = current

    for _ in range(validate_steps(steps, 2)):
        prediction = _predict_unet_once(model, current)
        current = histogram_equalize_rgb(prediction)

    return ensure_rgb(prediction)


def _predict_unet_once(model, image: np.ndarray) -> np.ndarray:
    if cv2 is None:  # pragma: no cover - OpenCV is part of runtime requirements.
        raise RuntimeError("OpenCV is required for FastAI U-Net inference.")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        rgb = ensure_rgb(image)
        cv2.imwrite(str(tmp_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        dl = model.dls.test_dl(str(tmp_path))
        preds = model.get_preds(dl=dl)
        raw = _as_numpy(preds[0][0])

        if raw.ndim == 3 and raw.shape[0] in {1, 3}:
            raw = np.transpose(raw, (1, 2, 0))

        return output_to_uint8_image(raw, source_range=(-3.0, 3.0))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _as_numpy(value) -> np.ndarray:
    """Best-effort conversion from tensor-like values to numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
