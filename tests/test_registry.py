from __future__ import annotations

import json

import pytest

from bone_suppression.registry import get_model_spec, load_model_registry


def test_load_default_registry_contains_expected_models() -> None:
    registry = load_model_registry()

    assert set(registry) == {"gan_mso2", "unet_resnet50"}
    assert registry["gan_mso2"].available is True
    assert registry["unet_resnet50"].available is False


def test_unknown_model_key_lists_available_models() -> None:
    with pytest.raises(KeyError, match="Available models"):
        get_model_spec("missing-model")


def test_registry_validation_rejects_missing_fields(tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": [{"key": "broken"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required"):
        load_model_registry(registry_path)
