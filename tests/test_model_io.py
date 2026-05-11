from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bone_suppression.model_io import load_model
from bone_suppression.registry import get_model_spec


def test_load_model_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        load_model(get_model_spec("gan_mso2"), "missing.keras", device="gpu")


def test_tensorflow_cpu_load_does_not_mutate_cuda_environment(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    calls: list[str] = []

    class FakeDevice:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            calls.append(f"enter:{self.name}")

        def __exit__(self, exc_type, exc, tb):
            calls.append(f"exit:{self.name}")

    class FakeLoadedModel:
        def __call__(self, *args, **kwargs):
            calls.append("predict")
            return "prediction"

    class FakeModels:
        @staticmethod
        def load_model(path):
            calls.append(f"load:{Path(path).name}")
            return FakeLoadedModel()

    fake_tf = SimpleNamespace(
        device=lambda name: FakeDevice(name),
        keras=SimpleNamespace(models=FakeModels),
    )

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    model = load_model(get_model_spec("gan_mso2"), checkpoint, device="cpu")
    result = model("image", training=True)

    assert result == "prediction"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls == [
        "enter:/CPU:0",
        "load:model.keras",
        "exit:/CPU:0",
        "enter:/CPU:0",
        "predict",
        "exit:/CPU:0",
    ]
