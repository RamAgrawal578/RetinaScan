"""
Contrast Limited Adaptive Histogram Equalization (CLAHE) for retinal fundus
images. Fundus photographs frequently have uneven illumination (dark
periphery, bright optic disc), and CLAHE on the luminance channel improves
the visibility of vessels and lesions without blowing out highlights the
way global histogram equalization would.
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    image_rgb: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 8
) -> np.ndarray:
    """Apply CLAHE to the L channel of the LAB color space and merge back,
    preserving color while boosting local contrast."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_green_channel_clahe(
    image_rgb: np.ndarray, clip_limit: float = 3.0, tile_grid_size: int = 8
) -> np.ndarray:
    """The green channel carries the strongest vessel/lesion contrast in
    fundus photography. This variant enhances only the green channel and
    is useful as an auxiliary diagnostic view (not used for the primary
    CNN input, which stays in enhanced RGB)."""
    channels = list(cv2.split(image_rgb))
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    channels[1] = clahe.apply(channels[1])
    return cv2.merge(channels)
