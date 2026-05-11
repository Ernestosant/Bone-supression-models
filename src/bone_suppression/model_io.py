"""Checkpoint loading with optional framework dependencies."""

from __future__ import annotations

from pathlib import Path

from bone_suppression.registry import ModelSpec, get_model_spec


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded for inference."""


def load_model(model: str | ModelSpec, checkpoint_path: str | Path):
    """Load a model checkpoint using the framework declared in the registry."""
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
        return load_learner(path)

    if spec.framework == "tensorflow":
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - depends on optional env.
            raise ModelLoadError("Install the TensorFlow requirements to load this model.") from exc
        return tf.keras.models.load_model(path)

    raise ModelLoadError(f"Unsupported model framework: {spec.framework}")
