"""
End-to-end image preprocessing pipeline shared by training and inference.

Both `train.py` (via `training/dataset_loader.py`) and
`model/predictor.py` call `preprocess_image()` so that the exact same
transform is applied at train and inference time — a common and easy
source of silent accuracy loss when the two paths drift apart.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from preprocessing.clahe import apply_clahe
from preprocessing.filters import (
    adjust_brightness_contrast,
    denoise,
    sharpen,
)
from utils.config_loader import get_config
from utils.image_utils import (
    ImageStats,
    bgr_to_rgb,
    center_crop,
    compute_image_stats,
    is_corrupted,
    normalize_image,
    read_image_bgr,
    resize_with_aspect_ratio,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# CHANGED: cap OpenCV's internal thread pool to 1.
#
# This is the main fix for "preprocessing looks stuck for hours". OpenCV
# functions (denoise especially) spin up their own internal thread pool
# by default. This module is ALSO parallelized at the tf.data level in
# dataset_loader.py (many images processed concurrently). Stacking
# outer-level threads on top of OpenCV's own inner threads oversubscribes
# the CPU — way more threads than physical cores — which causes thread
# contention and context-switch overhead instead of speedup. On an
# 8GB CPU-only machine this can turn a 20-minute job into hours.
#
# Pinning OpenCV to 1 thread per call and letting the outer tf.data
# parallelism (dataset_loader.py) own the scaling instead is the correct
# division of labor here.
cv2.setNumThreads(1)

_DEFAULT_TARGET_SIZE: int = get_config().IMAGE_SIZE


@dataclass
class PreprocessResult:
    """Container for the preprocessed tensor plus the diagnostics gathered
    along the way, so the API layer can surface QA info without redoing
    the work."""

    image: Optional[np.ndarray]
    stats: Optional[ImageStats]
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.image is not None and not self.errors


def preprocess_image(
    path: str,
    target_size: int = _DEFAULT_TARGET_SIZE,
    apply_enhancement: bool = True,
    crop_fraction: float = 0.92,
    normalize: bool = True,
    compute_stats: bool = True,
) -> PreprocessResult:
    """Run the full preprocessing pipeline on a single image file:

    1. Corrupted-file detection
    2. BGR -> RGB conversion
    3. Center crop (removes fundus camera vignette border)
    4. Aspect-ratio-preserving resize + pad to `target_size`
    5. Denoise -> CLAHE contrast enhancement -> sharpen
    6. Brightness normalization
    7. [0, 1] float32 scaling (optional, disabled for display/report use)

    `compute_stats` controls whether blur/quality diagnostics are computed
    on the *original, full-resolution* image before resizing. That's
    useful QA info for a single inference request, but expensive
    (Laplacian variance etc. on multi-megapixel fundus photos) and adds up
    fast across a multi-thousand-image training set.
    `training/dataset_loader.py` now passes `compute_stats=False` for
    this reason; `model/predictor.py` should keep the default `True`.
    """
    errors: List[str] = []

    if is_corrupted(path):
        errors.append("File is corrupted or not a readable image.")
        return PreprocessResult(image=None, stats=None, errors=errors)

    image_bgr = read_image_bgr(path)
    stats = compute_image_stats(image_bgr) if compute_stats else None

    if stats is not None and stats.is_blurry:
        logger.warning(
            "Image '%s' flagged as blurry (score=%.1f)", path, stats.blur_score
        )

    image_rgb = bgr_to_rgb(image_bgr)
    image_rgb = center_crop(image_rgb, crop_fraction=crop_fraction)
    image_rgb = resize_with_aspect_ratio(image_rgb, target_size=target_size)

    if apply_enhancement:
        image_rgb = denoise(image_rgb, strength=5)
        image_rgb = apply_clahe(image_rgb, clip_limit=2.5)
        image_rgb = sharpen(image_rgb, amount=0.6)
        image_rgb = adjust_brightness_contrast(
            image_rgb, brightness=0.0, contrast=1.05
        )

    output = normalize_image(image_rgb) if normalize else image_rgb.astype(
        np.float32
    )

    return PreprocessResult(image=output, stats=stats, errors=errors)


def preprocess_batch(
    paths: List[str],
    target_size: int = _DEFAULT_TARGET_SIZE,
    apply_enhancement: bool = True,
    max_workers: int = 4,
) -> List[PreprocessResult]:
    """Convenience wrapper for preprocessing multiple files (e.g. batch
    evaluation scripts). Uses a thread pool since OpenCV calls release
    the GIL. Set max_workers=1 for strictly sequential behavior."""
    if max_workers <= 1:
        return [
            preprocess_image(
                path, target_size=target_size, apply_enhancement=apply_enhancement
            )
            for path in paths
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda p: preprocess_image(
                    p, target_size=target_size, apply_enhancement=apply_enhancement
                ),
                paths,
            )
        )
    return results