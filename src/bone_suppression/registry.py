"""Model registry helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bone_suppression.preprocessing import require_keys

REQUIRED_MODEL_FIELDS = {
    "key",
    "display_name",
    "framework",
    "architecture",
    "status",
    "available",
    "checkpoint_url",
    "checkpoint_filename",
    "default_steps",
    "preprocessing",
    "notes",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    framework: str
    architecture: str
    status: str
    available: bool
    checkpoint_url: str
    checkpoint_filename: str
    default_steps: int
    preprocessing: tuple[str, ...]
    notes: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpec:
        require_keys(payload, REQUIRED_MODEL_FIELDS, "Model registry entry")
        return cls(
            key=str(payload["key"]),
            display_name=str(payload["display_name"]),
            framework=str(payload["framework"]),
            architecture=str(payload["architecture"]),
            status=str(payload["status"]),
            available=bool(payload["available"]),
            checkpoint_url=str(payload["checkpoint_url"]),
            checkpoint_filename=str(payload["checkpoint_filename"]),
            default_steps=int(payload["default_steps"]),
            preprocessing=tuple(str(item) for item in payload["preprocessing"]),
            notes=str(payload["notes"]),
        )


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "model_registry.json"


def load_model_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Load and validate the JSON model registry."""
    registry_path = Path(path) if path is not None else default_registry_path()
    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    require_keys(payload, {"models"}, "Model registry")
    specs = [ModelSpec.from_dict(item) for item in payload["models"]]
    registry: dict[str, ModelSpec] = {}
    for spec in specs:
        if spec.key in registry:
            raise ValueError(f"Duplicate model key in registry: {spec.key}")
        registry[spec.key] = spec

    return registry


def get_model_spec(model_key: str, path: str | Path | None = None) -> ModelSpec:
    registry = load_model_registry(path)
    try:
        return registry[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model '{model_key}'. Available models: {available}.") from exc


def available_model_keys(path: str | Path | None = None) -> list[str]:
    return [key for key, spec in load_model_registry(path).items() if spec.available]
