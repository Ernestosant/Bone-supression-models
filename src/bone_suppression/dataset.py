"""Dataset pairing and deterministic split helpers for Kaggle JSRT/BSE data."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_SLUG = "hmchuong/xray-bone-shadow-supression"
DEFAULT_SOURCE_SUBDIR = "JSRT/JSRT"
DEFAULT_TARGET_SUBDIR = "BSE_JSRT/BSE_JSRT"
DEFAULT_SEED = 2026
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ImagePair:
    """A paired normal chest X-ray and bone-suppressed target."""

    id: str
    input_path: Path
    target_path: Path

    def to_json(self, dataset_root: Path) -> dict[str, str]:
        root = dataset_root.resolve()
        return {
            "id": self.id,
            "input": _relative_posix(self.input_path, root),
            "target": _relative_posix(self.target_path, root),
        }

    @classmethod
    def from_json(cls, payload: dict[str, str], dataset_root: Path) -> ImagePair:
        return cls(
            id=str(payload["id"]),
            input_path=dataset_root / payload["input"],
            target_path=dataset_root / payload["target"],
        )


def discover_pairs(
    dataset_root: str | Path,
    source_subdir: str = DEFAULT_SOURCE_SUBDIR,
    target_subdir: str = DEFAULT_TARGET_SUBDIR,
) -> list[ImagePair]:
    """Find matching JSRT source and BSE_JSRT target images by filename stem."""
    root = Path(dataset_root)
    source_root = root / source_subdir
    target_root = root / target_subdir
    if not source_root.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"Target image directory not found: {target_root}")

    sources = _indexed_images(source_root)
    targets = _indexed_images(target_root)
    shared_ids = sorted(set(sources) & set(targets))
    if not shared_ids:
        raise ValueError(
            "No paired images found. Expected matching filenames under "
            f"{source_subdir!r} and {target_subdir!r}."
        )

    return [
        ImagePair(id=item_id, input_path=sources[item_id], target_path=targets[item_id])
        for item_id in shared_ids
    ]


def build_split_payload(
    pairs: list[ImagePair],
    dataset_root: str | Path,
    seed: int = DEFAULT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    dataset_slug: str = DATASET_SLUG,
    source_subdir: str = DEFAULT_SOURCE_SUBDIR,
    target_subdir: str = DEFAULT_TARGET_SUBDIR,
) -> dict[str, Any]:
    """Return a serializable deterministic split payload."""
    _validate_fractions(train_fraction, validation_fraction)
    root = Path(dataset_root)
    ordered = sorted(pairs, key=lambda pair: pair.id)
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_fraction)
    validation_count = int(total * validation_fraction)
    test_count = total - train_count - validation_count

    train = shuffled[:train_count]
    validation = shuffled[train_count : train_count + validation_count]
    test = shuffled[train_count + validation_count :]

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    return {
        "schema_version": 1,
        "dataset_slug": dataset_slug,
        "source_subdir": source_subdir,
        "target_subdir": target_subdir,
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": round(1.0 - train_fraction - validation_fraction, 10),
        },
        "counts": {
            "total": total,
            "train": len(train),
            "validation": len(validation),
            "test": test_count,
        },
        "splits": {
            split_name: [pair.to_json(root) for pair in split_pairs]
            for split_name, split_pairs in splits.items()
        },
    }


def write_splits(
    output_path: str | Path,
    dataset_root: str | Path,
    seed: int = DEFAULT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    source_subdir: str = DEFAULT_SOURCE_SUBDIR,
    target_subdir: str = DEFAULT_TARGET_SUBDIR,
) -> dict[str, Any]:
    """Discover pairs and write a deterministic split JSON file."""
    pairs = discover_pairs(dataset_root, source_subdir=source_subdir, target_subdir=target_subdir)
    payload = build_split_payload(
        pairs,
        dataset_root,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        source_subdir=source_subdir,
        target_subdir=target_subdir,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_splits(splits_path: str | Path, dataset_root: str | Path) -> dict[str, list[ImagePair]]:
    """Load split JSON and resolve image paths against dataset_root."""
    path = Path(splits_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported split schema version: {payload.get('schema_version')!r}")

    root = Path(dataset_root)
    splits = payload.get("splits", {})
    required = {"train", "validation", "test"}
    missing = sorted(required - set(splits))
    if missing:
        raise ValueError(f"Split file is missing required split(s): {', '.join(missing)}.")

    return {
        split_name: [ImagePair.from_json(item, root) for item in splits[split_name]]
        for split_name in sorted(required)
    }


def split_counts(splits: dict[str, list[ImagePair]]) -> dict[str, int]:
    """Return counts for each split plus total."""
    counts = {name: len(items) for name, items in splits.items()}
    counts["total"] = sum(counts.values())
    return counts


def pair_to_record(pair: ImagePair) -> dict[str, str]:
    """Return an absolute-path record useful for debugging and reports."""
    return {
        "id": pair.id,
        "input_path": str(pair.input_path),
        "target_path": str(pair.target_path),
    }


def _indexed_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images[path.stem] = path
    return images


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a non-empty test split.")


def pair_asdict(pair: ImagePair) -> dict[str, Any]:
    """Return a JSON-friendly representation with string paths."""
    payload = asdict(pair)
    payload["input_path"] = str(payload["input_path"])
    payload["target_path"] = str(payload["target_path"])
    return payload
