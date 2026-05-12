from __future__ import annotations

import json
from pathlib import Path

import pytest

from bone_suppression import registry as registry_module
from bone_suppression.registry import direct_checkpoint_url, get_model_spec, load_model_registry


def test_load_default_registry_contains_expected_models() -> None:
    registry = load_model_registry()

    assert set(registry) == {"gan_mso2", "unet_resnet50"}
    assert registry["gan_mso2"].checkpoint_filename == "gan_mso2_retrained_v1.keras"
    assert registry["gan_mso2"].device_support == ("cpu", "gpu")
    assert registry["gan_mso2"].available is True
    assert registry["unet_resnet50"].available is True


def test_unknown_model_key_lists_available_models() -> None:
    with pytest.raises(KeyError, match="Available models"):
        get_model_spec("missing-model")


def test_registry_validation_rejects_missing_fields(tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": [{"key": "broken"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required"):
        load_model_registry(registry_path)


def test_registry_can_load_from_environment_override(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "custom",
                        "display_name": "Custom",
                        "framework": "tensorflow",
                        "architecture": "Test architecture",
                        "status": "available",
                        "available": True,
                        "checkpoint_url": "https://example.com/model.h5",
                        "checkpoint_filename": "model.h5",
                        "checkpoint_sha256": "abc123",
                        "default_steps": 1,
                        "device_support": ["cpu"],
                        "examples": [],
                        "metrics": {},
                        "preprocessing": ["RGB conversion"],
                        "training_artifacts_url": "https://example.com/artifacts",
                        "notes": "Test registry.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(registry_module.REGISTRY_ENV_VAR, str(registry_path))

    registry = load_model_registry()

    assert set(registry) == {"custom"}


def test_registry_can_load_packaged_resource(monkeypatch) -> None:
    monkeypatch.delenv(registry_module.REGISTRY_ENV_VAR, raising=False)
    monkeypatch.setattr(registry_module, "_checkout_registry_path", lambda: Path("missing.json"))

    registry = load_model_registry()

    assert set(registry) == {"gan_mso2", "unet_resnet50"}


def test_direct_checkpoint_url_normalizes_google_drive_share_link() -> None:
    url = "https://drive.google.com/file/d/abc123/view?usp=sharing"

    direct = direct_checkpoint_url(url)

    assert direct == "https://drive.google.com/uc?export=download&id=abc123"
