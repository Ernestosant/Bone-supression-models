"""Quantitative image metrics for bone suppression evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, PSNR, and global SSIM for one prediction-target pair."""
    pred = _to_float01(prediction)
    tgt = _to_float01(target)
    if pred.shape != tgt.shape:
        raise ValueError(
            f"Metric inputs must have the same shape, got {pred.shape} and {tgt.shape}."
        )

    diff = pred - tgt
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    rmse = float(math.sqrt(mse))
    psnr = float("inf") if mse == 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": _global_ssim(pred, tgt),
    }


def aggregate_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Average per-image metric records, preserving infinity for perfect PSNR."""
    items = list(records)
    if not items:
        raise ValueError("Cannot aggregate an empty metric record list.")

    aggregate: dict[str, float] = {}
    for key in ("mae", "rmse", "psnr", "ssim", "cpu_seconds", "inference_seconds"):
        values = [float(item[key]) for item in items if key in item]
        if not values:
            continue
        if key == "psnr" and any(math.isinf(value) for value in values):
            aggregate[key] = float("inf") if all(math.isinf(value) for value in values) else float(
                np.mean([value for value in values if not math.isinf(value)])
            )
        else:
            aggregate[key] = float(np.mean(values))
    if "cpu_seconds" in aggregate:
        aggregate["cpu_seconds_per_image"] = aggregate.pop("cpu_seconds")
    if "inference_seconds" in aggregate:
        aggregate["inference_seconds_per_image"] = aggregate.pop("inference_seconds")
    return aggregate


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image).astype(np.float32)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] not in {1, 3}:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {array.shape}.")

    finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    if finite.size and finite.min() < 0.0:
        finite = finite * 0.5 + 0.5
    elif finite.size and finite.max() > 1.0:
        finite = finite / 255.0
    return np.clip(finite, 0.0, 1.0)


def _global_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = _to_luminance(prediction)
    tgt = _to_luminance(target)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_pred = float(np.mean(pred))
    mu_tgt = float(np.mean(tgt))
    var_pred = float(np.mean((pred - mu_pred) ** 2))
    var_tgt = float(np.mean((tgt - mu_tgt) ** 2))
    cov = float(np.mean((pred - mu_pred) * (tgt - mu_tgt)))

    numerator = (2.0 * mu_pred * mu_tgt + c1) * (2.0 * cov + c2)
    denominator = (mu_pred**2 + mu_tgt**2 + c1) * (var_pred + var_tgt + c2)
    if denominator == 0.0:
        return 1.0
    return float(max(min(numerator / denominator, 1.0), -1.0))


def _to_luminance(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 1:
        return image[..., 0]
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(image[..., :3], weights, axes=([-1], [0]))
