from __future__ import annotations

import numpy as np
import pytest

from bone_suppression.preprocessing import (
    ensure_rgb,
    histogram_equalize_rgb,
    legacy_mso_preprocess,
    normalize_to_minus_one_one,
    output_to_uint8_image,
    read_legacy_mso_image,
    resize_if_larger,
    validate_steps,
)


def test_ensure_rgb_expands_grayscale() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    rgb = ensure_rgb(image)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[..., 0], image)


def test_histogram_equalize_rgb_preserves_three_channels() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = 128

    equalized = histogram_equalize_rgb(image)

    assert equalized.shape == (8, 8, 3)
    assert equalized.dtype == np.uint8


def test_legacy_mso_preprocess_inverts_before_equalization() -> None:
    image = np.tile(np.arange(16, dtype=np.uint8).reshape(4, 4), (3, 1, 1)).transpose(1, 2, 0)

    preprocessed = legacy_mso_preprocess(image)

    assert preprocessed.shape == (4, 4, 3)
    assert preprocessed.dtype == np.uint8
    assert preprocessed[0, 0, 0] > preprocessed[-1, -1, 0]


def test_read_legacy_mso_image_handles_16_bit_targets(tmp_path) -> None:
    image = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 1024).astype(np.uint16)
    path = tmp_path / "target.png"
    from PIL import Image

    Image.fromarray(image).save(path)

    preprocessed = read_legacy_mso_image(path)

    assert preprocessed.shape == (4, 4, 3)
    assert preprocessed.dtype == np.uint8
    assert preprocessed.max() <= 255


def test_resize_if_larger_only_resizes_large_images() -> None:
    small = np.zeros((128, 128, 3), dtype=np.uint8)
    large = np.zeros((512, 300, 3), dtype=np.uint8)

    assert resize_if_larger(small).shape == (128, 128, 3)
    assert resize_if_larger(large).shape == (256, 256, 3)


def test_normalize_to_minus_one_one_range() -> None:
    image = np.array([[[0, 127, 255]]], dtype=np.uint8)

    normalized = normalize_to_minus_one_one(image)

    assert normalized.dtype == np.float32
    assert normalized.min() == pytest.approx(-1.0)
    assert normalized.max() == pytest.approx(1.0)


def test_output_to_uint8_image_preserves_uint8_scale_outputs() -> None:
    output = np.array([[[0.0, 127.0, 255.0]]], dtype=np.float32)

    image = output_to_uint8_image(output)

    assert image.dtype == np.uint8
    assert image.tolist() == [[[0, 127, 255]]]


def test_output_to_uint8_image_windows_out_of_range_outputs() -> None:
    output = np.linspace(-2.0, 3.0, num=25, dtype=np.float32).reshape(5, 5)

    image = output_to_uint8_image(output)

    assert image.dtype == np.uint8
    assert image.shape == (5, 5, 3)
    assert image.min() == 0
    assert image.max() == 255


def test_output_to_uint8_image_uses_explicit_source_range() -> None:
    output = np.array([[[-3.0, 0.0, 3.0]]], dtype=np.float32)

    image = output_to_uint8_image(output, source_range=(-3.0, 3.0))

    assert image.tolist() == [[[0, 127, 255]]]


def test_validate_steps_rejects_zero() -> None:
    with pytest.raises(ValueError):
        validate_steps(0, default=2)
