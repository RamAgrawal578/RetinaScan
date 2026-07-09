"""
Classical image filters used in the preprocessing pipeline: noise
reduction, sharpening, and brightness/contrast adjustment.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise(image_rgb: np.ndarray, strength: int = 7) -> np.ndarray:
    """Non-local means denoising — effective against the sensor noise
    common in lower-quality fundus camera captures, while preserving
    fine vessel edges better than a simple Gaussian blur."""
    return cv2.fastNlMeansDenoisingColored(
        image_rgb, None, h=strength, hColor=strength, templateWindowSize=7,
        searchWindowSize=21,
    )


def sharpen(image_rgb: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: subtract a blurred copy from the original to
    boost high-frequency detail (vessel edges, microaneurysms)."""
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(
        image_rgb, 1 + amount, blurred, -amount, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(
    image_rgb: np.ndarray, brightness: float = 0.0, contrast: float = 1.0
) -> np.ndarray:
    """Apply a linear brightness/contrast transform:
    output = image * contrast + brightness."""
    adjusted = image_rgb.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def estimate_and_correct_illumination(image_rgb: np.ndarray) -> np.ndarray:
    """Correct uneven illumination by dividing by a heavily blurred
    estimate of the background, then rescaling — reduces the common
    'bright center, dark edges' fundus vignetting artifact."""
    background = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=30)
    background = np.clip(background, 1, 255).astype(np.float32)
    corrected = (image_rgb.astype(np.float32) / background) * 128.0
    return np.clip(corrected, 0, 255).astype(np.uint8)
