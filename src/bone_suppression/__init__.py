"""Research utilities for chest X-ray bone suppression models."""

from bone_suppression.inference import predict_model, run_inference
from bone_suppression.registry import ModelSpec, get_model_spec, load_model_registry

__all__ = [
    "ModelSpec",
    "get_model_spec",
    "load_model_registry",
    "predict_model",
    "run_inference",
]

__version__ = "0.1.0"
