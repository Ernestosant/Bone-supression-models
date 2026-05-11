"""Evaluation and example-panel generation for trained checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bone_suppression.artifacts import sha256_file, write_json
from bone_suppression.dataset import DATASET_SLUG, load_splits
from bone_suppression.inference import predict_model
from bone_suppression.metrics import aggregate_metrics, image_metrics
from bone_suppression.model_io import load_model
from bone_suppression.preprocessing import legacy_mso_preprocess, read_legacy_mso_image
from bone_suppression.registry import get_model_spec


def evaluate_checkpoint(
    model_key: str,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
    steps: int | None = None,
    limit: int | None = None,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Run inference over a split and write metrics plus optional predictions."""
    root = Path(dataset_root)
    output_root = Path(output_dir)
    predictions_dir = output_root / "predictions" / model_key
    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    split_pairs = load_splits(splits_path, root)
    if split not in split_pairs:
        available = ", ".join(sorted(split_pairs))
        raise KeyError(f"Unknown split {split!r}. Available splits: {available}.")
    pairs = split_pairs[split][:limit]

    spec = get_model_spec(model_key)
    model = load_model(spec, checkpoint_path, device=device)

    records: list[dict[str, Any]] = []
    for pair in pairs:
        input_image = _read_rgb(pair.input_path)
        target_image = read_legacy_mso_image(pair.target_path)

        start = time.perf_counter()
        prediction = predict_model(model, spec, input_image, steps=steps)
        elapsed = time.perf_counter() - start

        target_for_metric = _resize_like(target_image, prediction)
        metrics = image_metrics(prediction, target_for_metric)
        record = {
            "id": pair.id,
            "input": _safe_relative(pair.input_path, root),
            "target": _safe_relative(pair.target_path, root),
            **metrics,
        }
        if device == "cpu":
            record["cpu_seconds"] = elapsed
        else:
            record["inference_seconds"] = elapsed
        records.append(record)
        if save_predictions:
            Image.fromarray(prediction).save(predictions_dir / f"{pair.id}.png")

    aggregate = aggregate_metrics(records)
    payload = {
        "schema_version": 1,
        "dataset_slug": DATASET_SLUG,
        "model_key": model_key,
        "checkpoint_filename": Path(checkpoint_path).name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": split,
        "device": device,
        "steps": steps,
        "counts": {"evaluated": len(records)},
        "aggregate": aggregate,
        "records": records,
    }
    write_json(output_root / f"{model_key}_{split}_metrics.json", payload)
    return payload


def evaluate_checkpoint_steps(
    model_key: str,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    steps_values: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    split: str = "test",
    device: str = "cpu",
    limit: int | None = None,
    save_predictions: bool = True,
    step0_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Evaluate autoregressive inference metrics for multiple step counts.

    Step 0 is a no-model baseline: the original input resized to step0_size. Steps >= 1 run the
    model autoregressively, feeding each output back as the next input through predict_model.
    """
    normalized_steps = tuple(int(step) for step in steps_values)
    if any(step < 0 for step in normalized_steps):
        raise ValueError("Evaluation steps must be non-negative.")

    root = Path(dataset_root)
    output_root = Path(output_dir)
    split_pairs = load_splits(splits_path, root)
    if split not in split_pairs:
        available = ", ".join(sorted(split_pairs))
        raise KeyError(f"Unknown split {split!r}. Available splits: {available}.")
    pairs = split_pairs[split][:limit]

    spec = get_model_spec(model_key)
    model = None
    if any(step > 0 for step in normalized_steps):
        model = load_model(spec, checkpoint_path, device=device)

    step_payloads = []
    for step_count in normalized_steps:
        predictions_dir = output_root / "predictions" / model_key / f"steps_{step_count}"
        if save_predictions:
            predictions_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for pair in pairs:
            input_image = _read_rgb(pair.input_path)
            target_image = read_legacy_mso_image(pair.target_path)

            start = time.perf_counter()
            if step_count == 0:
                prediction = _resize_to_size(legacy_mso_preprocess(input_image), step0_size)
            else:
                prediction = predict_model(model, spec, input_image, steps=step_count)
            elapsed = time.perf_counter() - start

            target_for_metric = _resize_like(target_image, prediction)
            metrics = image_metrics(prediction, target_for_metric)
            record = {
                "id": pair.id,
                "input": _safe_relative(pair.input_path, root),
                "target": _safe_relative(pair.target_path, root),
                **metrics,
            }
            if device == "cpu":
                record["cpu_seconds"] = elapsed
            else:
                record["inference_seconds"] = elapsed
            records.append(record)
            if save_predictions:
                Image.fromarray(prediction).save(predictions_dir / f"{pair.id}.png")

        aggregate = aggregate_metrics(records)
        step_payload = {
            "steps": step_count,
            "counts": {"evaluated": len(records)},
            "aggregate": aggregate,
            "records": records,
        }
        step_path = output_root / f"{model_key}_{split}_steps_{step_count}_metrics.json"
        write_json(step_path, step_payload)
        step_payloads.append(step_payload)

    payload = {
        "schema_version": 1,
        "dataset_slug": DATASET_SLUG,
        "model_key": model_key,
        "checkpoint_filename": Path(checkpoint_path).name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": split,
        "device": device,
        "step0_size": list(step0_size),
        "step_results": step_payloads,
    }
    write_json(output_root / f"{model_key}_{split}_steps_metrics.json", payload)
    return payload


def build_comparison_examples(
    dataset_root: str | Path,
    splits_path: str | Path,
    prediction_dirs: dict[str, str | Path],
    output_dir: str | Path,
    split: str = "test",
    count: int = 3,
) -> list[Path]:
    """Create input/target/model-output panels for fixed examples from a split."""
    root = Path(dataset_root)
    split_pairs = load_splits(splits_path, root)
    pairs = split_pairs[split][:count]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for pair in pairs:
        images: list[tuple[str, np.ndarray]] = [
            ("Input", read_legacy_mso_image(pair.input_path)),
            ("Target", read_legacy_mso_image(pair.target_path)),
        ]
        for model_key, prediction_dir in prediction_dirs.items():
            pred_path = Path(prediction_dir) / f"{pair.id}.png"
            if pred_path.exists():
                images.append((model_key, _read_rgb(pred_path)))

        panel = _make_panel(images)
        output_path = output_root / f"{pair.id}_comparison.png"
        Image.fromarray(panel).save(output_path)
        written.append(output_path)
    return written


def build_step_comparison_examples(
    dataset_root: str | Path,
    splits_path: str | Path,
    prediction_root: str | Path,
    output_dir: str | Path,
    model_label: str,
    steps_values: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    split: str = "test",
    count: int = 3,
) -> list[Path]:
    """Create input/target/step-output panels for fixed examples."""
    root = Path(dataset_root)
    split_pairs = load_splits(splits_path, root)
    pairs = split_pairs[split][:count]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions = Path(prediction_root)

    written: list[Path] = []
    for pair in pairs:
        images: list[tuple[str, np.ndarray]] = [
            ("Input", read_legacy_mso_image(pair.input_path)),
            ("Target", read_legacy_mso_image(pair.target_path)),
        ]
        for step_count in steps_values:
            pred_path = predictions / f"steps_{step_count}" / f"{pair.id}.png"
            if pred_path.exists():
                images.append((f"{model_label} s{step_count}", _read_rgb(pred_path)))

        panel = _make_panel(images)
        output_path = output_root / f"{pair.id}_{model_label.lower().replace(' ', '_')}_steps.png"
        Image.fromarray(panel).save(output_path)
        written.append(output_path)
    return written


def merge_metric_files(metric_paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    """Merge per-model metric files into one compact results JSON."""
    models = []
    for path in metric_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        models.append(
            {
                "model_key": payload["model_key"],
                "checkpoint_filename": payload["checkpoint_filename"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "split": payload["split"],
                "device": payload["device"],
                "counts": payload["counts"],
                "aggregate": payload["aggregate"],
            }
        )
    merged = {"schema_version": 1, "models": models}
    write_json(output_path, merged)
    return merged


def _read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    width = reference.shape[1]
    height = reference.shape[0]
    resized = Image.fromarray(image).resize((width, height), Image.BICUBIC)
    return np.asarray(resized.convert("RGB"))


def _resize_to_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size[1], size[0]
    resized = Image.fromarray(image).resize((width, height), Image.BICUBIC)
    return np.asarray(resized.convert("RGB"))


def _make_panel(images: list[tuple[str, np.ndarray]], tile_size: int = 256) -> np.ndarray:
    label_height = 28
    tiles: list[Image.Image] = []
    for label, array in images:
        tile = Image.fromarray(array).convert("RGB").resize((tile_size, tile_size), Image.BICUBIC)
        canvas = Image.new("RGB", (tile_size, tile_size + label_height), "white")
        canvas.paste(tile, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), label, fill="black", font=ImageFont.load_default())
        tiles.append(canvas)

    panel = Image.new("RGB", (tile_size * len(tiles), tile_size + label_height), "white")
    for index, tile in enumerate(tiles):
        panel.paste(tile, (index * tile_size, 0))
    return np.asarray(panel)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
