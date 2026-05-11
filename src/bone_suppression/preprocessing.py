"""Image preprocessing utilities shared by model inference paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

try:  # pragma: no cover - fallback is exercised when OpenCV is unavailable.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


ArrayLikeImage = np.ndarray
LEGACY_MSO_PREPROCESSING = (
    "OpenCV 8-bit read, intensity inversion with 255-image, grayscale histogram "
    "equalization, and RGB expansion"
)


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


def legacy_mso_preprocess(image: ArrayLikeImage) -> np.ndarray:
    """Match the original MSO notebooks: invert intensities, then equalize grayscale."""
    rgb = ensure_rgb(image)
    return histogram_equalize_rgb(255 - rgb)


def read_cv2_uint8_rgb(path: str | Path) -> np.ndarray:
    """Read an image like the notebooks did with ``cv2.imread`` default flags."""
    image_path = Path(path)
    if cv2 is not None:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image could not be read: {image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if not image_path.exists():  # pragma: no cover - OpenCV is a runtime dependency.
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    return ensure_rgb(np.asarray(Image.open(image_path).convert("RGB")))


def read_legacy_mso_image(path: str | Path) -> np.ndarray:
    """Read and preprocess a JSRT/BSE image using the historical MSO notebook path."""
    return legacy_mso_preprocess(read_cv2_uint8_rgb(path))


def save_legacy_mso_image(input_path: str | Path, output_path: str | Path) -> None:
    """Write a notebook-compatible preprocessed image for cached training datasets."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(read_legacy_mso_image(input_path)).save(path)


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


def output_to_uint8_image(
    output: ArrayLikeImage,
    source_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert model output into uint8 RGB without saturating common ranges."""
    array = np.asarray(output)
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    if not array.size:
        return ensure_rgb(array)

    if source_range is not None:
        low, high = source_range
        if high <= low:
            raise ValueError("source_range high value must be greater than low value.")
        scaled = (array - low) / (high - low)
        return ensure_rgb(np.clip(scaled, 0.0, 1.0))

    array_min = float(array.min())
    array_max = float(array.max())
    if array_min >= -1.0 and array_max <= 1.0 and array_min < 0.0:
        array = array * 0.5 + 0.5
        return ensure_rgb(np.clip(array, 0.0, 1.0))
    if array_min >= 0.0 and array_max <= 1.0:
        return ensure_rgb(np.clip(array, 0.0, 1.0))
    if array_min >= 0.0 and array_max <= 255.0:
        return ensure_rgb(array)

    return ensure_rgb(_window_to_uint8(array))


def _window_to_uint8(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - low) / (high - low)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


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
