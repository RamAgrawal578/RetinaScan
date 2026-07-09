"""
Low-level image I/O and statistics helpers shared by preprocessing,
training, and prediction code paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageStats:
    """Basic per-image statistics used for validation and QA."""

    width: int
    height: int
    channels: int
    mean_brightness: float
    std_brightness: float
    is_blurry: bool
    blur_score: float


def read_image_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image from disk as a BGR NumPy array, returning None on
    failure instead of raising, so callers can produce a clean validation
    error rather than a stack trace."""
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("cv2.imread returned None for '%s'", path)
        return image
    except Exception as exc:  # noqa: BLE001 - convert any decode error
        logger.error("Failed to read image '%s': %s", path, exc)
        return None


def is_corrupted(path: str) -> bool:
    """Detect corrupted / truncated / unreadable image files."""
    image = read_image_bgr(path)
    return image is None or image.size == 0


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, cheap blur estimator.
    Lower values indicate a blurrier image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return _blur_score_from_gray(gray)


def _blur_score_from_gray(gray: np.ndarray) -> float:
    # CHANGED: factored out so compute_image_stats can reuse the gray
    # frame it already computed instead of converting to grayscale
    # twice on every single image (was a small but real, easily-avoided
    # cost multiplied across the whole dataset).
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_image_stats(
    image_bgr: np.ndarray, blur_threshold: float = 100.0
) -> ImageStats:
    """Compute size, brightness, and blur statistics for QA/logging."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = _blur_score_from_gray(gray)
    height, width = image_bgr.shape[:2]
    channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
    return ImageStats(
        width=width,
        height=height,
        channels=channels,
        mean_brightness=float(np.mean(gray)),
        std_brightness=float(np.std(gray)),
        is_blurry=blur_score < blur_threshold,
        blur_score=blur_score,
    )


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_with_aspect_ratio(
    image: np.ndarray, target_size: int, pad_color: int = 0
) -> np.ndarray:
    """Resize preserving aspect ratio, then pad to a square of
    `target_size` x `target_size` (letterboxing) rather than distorting
    the fundus image geometry."""
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas_shape: Tuple[int, ...]
    if image.ndim == 3:
        canvas_shape = (target_size, target_size, image.shape[2])
    else:
        canvas_shape = (target_size, target_size)
    canvas = np.full(canvas_shape, pad_color, dtype=image.dtype)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def center_crop(image: np.ndarray, crop_fraction: float = 0.9) -> np.ndarray:
    """Crop the central region of an image, discarding a border margin.
    Useful for removing the dark corners/vignette common in fundus
    photography."""
    height, width = image.shape[:2]
    crop_h, crop_w = int(height * crop_fraction), int(width * crop_fraction)
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to the [0, 1] float32 range."""
    return image.astype(np.float32) / 255.0