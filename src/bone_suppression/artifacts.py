"""Artifact helpers for reproducible training runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest(
    model_key: str,
    checkpoint_path: str | Path,
    dataset_slug: str,
    split_path: str | Path,
    seed: int,
    hyperparameters: dict[str, Any],
    metrics_path: str | Path | None = None,
    training_environment: str = "kaggle-p100",
) -> dict[str, Any]:
    """Build the manifest recorded next to a trained checkpoint."""
    checkpoint = Path(checkpoint_path)
    return {
        "schema_version": 1,
        "model_key": model_key,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_slug": dataset_slug,
        "split_path": str(split_path),
        "seed": seed,
        "hyperparameters": hyperparameters,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "training_environment": training_environment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def current_git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
