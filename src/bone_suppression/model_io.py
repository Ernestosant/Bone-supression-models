"""Checkpoint loading with optional framework dependencies."""

from __future__ import annotations

import os
from pathlib import Path

from bone_suppression.registry import ModelSpec, get_model_spec

SUPPORTED_DEVICES = {"auto", "cpu"}


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded for inference."""


def load_model(model: str | ModelSpec, checkpoint_path: str | Path, device: str = "auto"):
    """Load a model checkpoint using the framework declared in the registry."""
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device {device!r}. Expected one of: auto, cpu.")

    spec = get_model_spec(model) if isinstance(model, str) else model
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if spec.framework == "fastai":
        try:
            from fastai.learner import load_learner
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            message = "Install the U-Net/FastAI requirements to load this model."
            raise ModelLoadError(message) from exc
        return load_learner(path, cpu=device == "cpu")

    if spec.framework == "tensorflow":
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise ModelLoadError("Install the TensorFlow requirements to load this model.") from exc
        if device == "cpu":
            _hide_tensorflow_gpus(tf)
        return tf.keras.models.load_model(path)

    raise ModelLoadError(f"Unsupported model framework: {spec.framework}")


def _hide_tensorflow_gpus(tf) -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):  # GPUs may already be initialized.
        return
