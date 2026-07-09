"""
Tests for the preprocessing pipeline. These use synthetic, generated
images so the suite never depends on a real dataset or trained model.
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from preprocessing.clahe import apply_clahe
from preprocessing.filters import (
    adjust_brightness_contrast,
    denoise,
    sharpen,
)
from preprocessing.preprocess import preprocess_image
from utils.image_utils import (
    center_crop,
    compute_blur_score,
    compute_image_stats,
    is_corrupted,
    normalize_image,
    resize_with_aspect_ratio,
)


@pytest.fixture
def synthetic_fundus_image(tmp_path):
    """Generate a plausible, non-trivial fundus-like test image (a
    gradient disc on a dark background) and write it to a temp file."""
    size = 512
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= (center - 20) ** 2
    image[mask] = [180, 90, 60]  # warm fundus-like tone
    cv2.circle(image, (center, center), 40, (200, 150, 100), -1)  # optic disc
    noise = np.random.default_rng(42).integers(0, 15, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)

    path = tmp_path / "synthetic_fundus.png"
    cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture
def corrupted_image_path(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not a real image")
    return str(path)


class TestImageUtils:
    def test_is_corrupted_detects_bad_file(self, corrupted_image_path):
        assert is_corrupted(corrupted_image_path) is True

    def test_is_corrupted_accepts_valid_file(self, synthetic_fundus_image):
        assert is_corrupted(synthetic_fundus_image) is False

    def test_resize_with_aspect_ratio_produces_square_output(self):
        image = np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)
        resized = resize_with_aspect_ratio(image, target_size=224)
        assert resized.shape == (224, 224, 3)

    def test_center_crop_reduces_dimensions(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = center_crop(image, crop_fraction=0.8)
        assert cropped.shape[0] < 100
        assert cropped.shape[1] < 100

    def test_normalize_image_scales_to_unit_range(self):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        normalized = normalize_image(image)
        assert normalized.dtype == np.float32
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

    def test_compute_image_stats_returns_expected_fields(self, synthetic_fundus_image):
        image = cv2.imread(synthetic_fundus_image)
        stats = compute_image_stats(image)
        assert stats.width == 512
        assert stats.height == 512
        assert stats.blur_score >= 0

    def test_blur_score_is_lower_for_blurred_image(self, synthetic_fundus_image):
        image = cv2.imread(synthetic_fundus_image)
        sharp_score = compute_blur_score(image)
        blurred = cv2.GaussianBlur(image, (25, 25), 10)
        blurred_score = compute_blur_score(blurred)
        assert blurred_score < sharp_score


class TestFilters:
    def test_apply_clahe_preserves_shape_and_dtype(self, synthetic_fundus_image):
        image = cv2.cvtColor(cv2.imread(synthetic_fundus_image), cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(image)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8

    def test_denoise_preserves_shape(self, synthetic_fundus_image):
        image = cv2.cvtColor(cv2.imread(synthetic_fundus_image), cv2.COLOR_BGR2RGB)
        denoised = denoise(image)
        assert denoised.shape == image.shape

    def test_sharpen_preserves_shape_and_valid_range(self, synthetic_fundus_image):
        image = cv2.cvtColor(cv2.imread(synthetic_fundus_image), cv2.COLOR_BGR2RGB)
        sharpened = sharpen(image)
        assert sharpened.shape == image.shape
        assert sharpened.min() >= 0
        assert sharpened.max() <= 255

    def test_adjust_brightness_contrast_changes_pixel_values(self):
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        brighter = adjust_brightness_contrast(image, brightness=50, contrast=1.0)
        assert brighter.mean() > image.mean()


class TestPreprocessPipeline:
    def test_preprocess_image_valid_file_succeeds(self, synthetic_fundus_image):
        result = preprocess_image(synthetic_fundus_image, target_size=224)
        assert result.is_valid
        assert result.image.shape == (224, 224, 3)
        assert result.image.dtype == np.float32
        assert result.image.max() <= 1.0

    def test_preprocess_image_corrupted_file_fails_gracefully(self, corrupted_image_path):
        result = preprocess_image(corrupted_image_path, target_size=224)
        assert not result.is_valid
        assert result.image is None
        assert len(result.errors) > 0

    def test_preprocess_image_without_enhancement(self, synthetic_fundus_image):
        result = preprocess_image(
            synthetic_fundus_image, target_size=224, apply_enhancement=False
        )
        assert result.is_valid
        assert result.image.shape == (224, 224, 3)
