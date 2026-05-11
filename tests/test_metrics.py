from __future__ import annotations

import math

import numpy as np
import pytest

from bone_suppression.metrics import aggregate_metrics, image_metrics


def test_image_metrics_are_perfect_for_identical_images() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    metrics = image_metrics(image, image)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)


def test_image_metrics_detect_error() -> None:
    prediction = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.full((2, 2, 3), 255, dtype=np.uint8)

    metrics = image_metrics(prediction, target)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["psnr"] == pytest.approx(0.0)
    assert metrics["ssim"] < 0.01


def test_aggregate_metrics_averages_records() -> None:
    aggregate = aggregate_metrics(
        [
            {"mae": 0.1, "rmse": 0.2, "psnr": 20.0, "ssim": 0.8, "cpu_seconds": 1.0},
            {"mae": 0.3, "rmse": 0.4, "psnr": 30.0, "ssim": 0.6, "cpu_seconds": 3.0},
        ]
    )

    assert aggregate["mae"] == pytest.approx(0.2)
    assert aggregate["cpu_seconds_per_image"] == pytest.approx(2.0)


def test_aggregate_metrics_averages_inference_seconds() -> None:
    aggregate = aggregate_metrics(
        [
            {"mae": 0.1, "rmse": 0.2, "psnr": 20.0, "ssim": 0.8, "inference_seconds": 1.0},
            {"mae": 0.3, "rmse": 0.4, "psnr": 30.0, "ssim": 0.6, "inference_seconds": 3.0},
        ]
    )

    assert aggregate["mae"] == pytest.approx(0.2)
    assert aggregate["inference_seconds_per_image"] == pytest.approx(2.0)
