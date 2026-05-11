"""Model registry helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from bone_suppression.preprocessing import require_keys

REGISTRY_ENV_VAR = "BONE_SUPPRESSION_MODEL_REGISTRY"
PACKAGED_REGISTRY = "resources/model_registry.json"

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
    """Return the editable-checkout registry path or an explicit override path."""
    env_path = os.getenv(REGISTRY_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    checkout_path = _checkout_registry_path()
    if checkout_path.exists():
        return checkout_path

    raise FileNotFoundError(
        "No filesystem registry path was found. Use load_model_registry() to load the packaged "
        f"registry resource, or set {REGISTRY_ENV_VAR} to an explicit registry JSON path."
    )


def load_model_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Load and validate the JSON model registry."""
    payload = _load_registry_payload(path)

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


def _load_registry_payload(path: str | Path | None = None) -> dict[str, Any]:
    if path is not None:
        return _load_registry_file(Path(path))

    env_path = os.getenv(REGISTRY_ENV_VAR)
    if env_path:
        return _load_registry_file(Path(env_path).expanduser())

    checkout_path = _checkout_registry_path()
    if checkout_path.exists():
        return _load_registry_file(checkout_path)

    registry_resource = resources.files("bone_suppression").joinpath(PACKAGED_REGISTRY)
    with registry_resource.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_registry_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _checkout_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "model_registry.json"
