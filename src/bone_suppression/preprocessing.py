"""Image preprocessing utilities shared by model inference paths."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:  # pragma: no cover - fallback is exercised when OpenCV is unavailable.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


ArrayLikeImage = np.ndarray


def ensure_rgb(image: ArrayLikeImage) -> np.ndarray:
    """Return an image as H x W x 3 uint8 RGB."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "Expected a grayscale, RGB, RGBA, or single-channel image array; "
            f"received shape {array.shape}."
        )

    return to_uint8(array)


def to_uint8(image: ArrayLikeImage) -> np.ndarray:
    """Convert an image-like array to uint8 while preserving display range."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()

    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if array.size and array.max() <= 1.0 and array.min() >= 0.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def histogram_equalize_rgb(image: ArrayLikeImage) -> np.ndarray:
    """Equalize image intensity and return a three-channel RGB image."""
    rgb = ensure_rgb(image)

    if cv2 is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    equalized = _equalize_histogram(rgb.mean(axis=-1).astype(np.uint8))
    return np.stack([equalized, equalized, equalized], axis=-1)


def resize_if_larger(
    image: ArrayLikeImage,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Resize to target_size when an image exceeds that height or width."""
    rgb = ensure_rgb(image)
    target_height, target_width = target_size
    height, width = rgb.shape[:2]

    if height <= target_height and width <= target_width:
        return rgb

    if cv2 is not None:
        return cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    y_idx = np.linspace(0, height - 1, target_height).astype(int)
    x_idx = np.linspace(0, width - 1, target_width).astype(int)
    return rgb[np.ix_(y_idx, x_idx)]


def normalize_to_minus_one_one(image: ArrayLikeImage) -> np.ndarray:
    """Normalize image intensities from [0, 255] to [-1, 1]."""
    return ensure_rgb(image).astype(np.float32) / 127.5 - 1.0


def output_to_uint8_image(output: ArrayLikeImage) -> np.ndarray:
    """Convert model output in [0, 1] or [-1, 1] into uint8 RGB."""
    array = np.asarray(output)
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    if array.size and array.min() < 0.0:
        array = array * 0.5 + 0.5
    return ensure_rgb(np.clip(array, 0.0, 1.0))


def _equalize_histogram(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return gray.copy()

    cdf_min = nonzero[0]
    denom = gray.size - cdf_min
    if denom == 0:
        return gray.copy()

    lut = np.round((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.uint8)
    return lut[gray]


def validate_steps(steps: int | None, default: int) -> int:
    """Normalize user-provided iterative inference steps."""
    value = default if steps is None else int(steps)
    if value < 1:
        raise ValueError("Inference steps must be greater than or equal to 1.")
    return value


def require_keys(payload: dict, keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} is missing required field(s): {joined}.")
