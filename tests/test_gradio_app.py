from __future__ import annotations

import numpy as np
import pytest

from bone_suppression import gradio_app
from bone_suppression.registry import ModelSpec


def test_predict_from_ui_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        gradio_app.predict_from_ui(np.zeros((8, 8, 3), dtype=np.uint8), "gan_mso2", "")


def test_predict_from_ui_calls_inference(monkeypatch) -> None:
    calls = {}

    def fake_run_inference(model_key, checkpoint_path, image, steps=None):
        calls["model_key"] = model_key
        calls["checkpoint_path"] = checkpoint_path
        calls["shape"] = image.shape
        calls["steps"] = steps
        return np.ones((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(gradio_app, "run_inference", fake_run_inference)

    output = gradio_app.predict_from_ui(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "gan_mso2",
        "models/checkpoints/gan_mso2.h5",
        2,
    )

    assert output.shape == (4, 4, 3)
    assert calls["model_key"] == "gan_mso2"
    assert calls["steps"] == 2


def test_available_model_choices_filters_pending_models() -> None:
    registry = {
        "gan_mso2": ModelSpec(
            key="gan_mso2",
            display_name="GAN",
            framework="tensorflow",
            architecture="Test",
            status="checkpoint_link_available",
            available=True,
            checkpoint_url="https://example.com/gan.h5",
            checkpoint_filename="gan.h5",
            default_steps=2,
            preprocessing=("RGB conversion",),
            notes="Available.",
        ),
        "unet_resnet50": ModelSpec(
            key="unet_resnet50",
            display_name="U-Net",
            framework="fastai",
            architecture="Test",
            status="pending_weights",
            available=False,
            checkpoint_url="https://example.com/unet.pkl",
            checkpoint_filename="unet.pkl",
            default_steps=2,
            preprocessing=("RGB conversion",),
            notes="Pending.",
        ),
    }

    assert gradio_app.available_model_choices(registry) == [("GAN", "gan_mso2")]
