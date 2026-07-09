"""
Tests for the prediction service and validators. These deliberately run
without a trained model present, exercising the "no model available yet"
path that a fresh clone of this repository will hit before training.
"""
from __future__ import annotations

import io

from utils.validators import (
    validate_class_names,
    validate_prediction_output,
    validate_upload_request,
)


class TestUploadValidation:
    def test_rejects_missing_file(self):
        result = validate_upload_request(
            has_file=False,
            filename="",
            allowed_extensions=frozenset({"png", "jpg"}),
            content_length=None,
            max_content_length=10 * 1024 * 1024,
        )
        assert not result.is_valid

    def test_rejects_disallowed_extension(self):
        result = validate_upload_request(
            has_file=True,
            filename="scan.exe",
            allowed_extensions=frozenset({"png", "jpg"}),
            content_length=1000,
            max_content_length=10 * 1024 * 1024,
        )
        assert not result.is_valid
        assert any("Unsupported file type" in err for err in result.errors)

    def test_rejects_oversized_file(self):
        result = validate_upload_request(
            has_file=True,
            filename="scan.png",
            allowed_extensions=frozenset({"png", "jpg"}),
            content_length=20 * 1024 * 1024,
            max_content_length=10 * 1024 * 1024,
        )
        assert not result.is_valid

    def test_accepts_valid_request(self):
        result = validate_upload_request(
            has_file=True,
            filename="scan.png",
            allowed_extensions=frozenset({"png", "jpg"}),
            content_length=1000,
            max_content_length=10 * 1024 * 1024,
        )
        assert result.is_valid
        assert result.errors == []


class TestClassNameValidation:
    def test_rejects_empty_class_list(self):
        result = validate_class_names([])
        assert not result.is_valid

    def test_rejects_single_class(self):
        result = validate_class_names(["Healthy"])
        assert not result.is_valid

    def test_accepts_multiple_classes(self):
        result = validate_class_names(["Healthy", "Glaucoma", "AMD"])
        assert result.is_valid


class TestPredictionOutputValidation:
    def test_rejects_mismatched_lengths(self):
        result = validate_prediction_output([0.5, 0.5], ["Healthy", "Glaucoma", "AMD"])
        assert not result.is_valid

    def test_rejects_probabilities_not_summing_to_one(self):
        result = validate_prediction_output([0.2, 0.2], ["Healthy", "Glaucoma"])
        assert not result.is_valid

    def test_accepts_valid_probabilities(self):
        result = validate_prediction_output([0.7, 0.3], ["Healthy", "Glaucoma"])
        assert result.is_valid


class TestPredictionServiceWithoutTrainedModel:
    def test_handle_upload_reports_no_model_available(self, flask_app):
        from config.settings import get_config
        from services.prediction_service import PredictionService

        service = PredictionService(get_config("testing"))
        file_storage = _make_fake_image_upload()

        response = service.handle_upload_and_predict(file_storage, generate_report=False)

        # No model has been trained in the isolated test environment yet,
        # so this should fail cleanly with a descriptive error rather than
        # raising — this is the exact state a fresh clone starts in.
        assert response["success"] is False
        assert any(
            "No trained model" in err or "corrupted" in err.lower() or "readable" in err.lower()
            for err in response["errors"]
        )

    def test_health_status_reports_model_not_loaded(self, flask_app):
        from config.settings import get_config
        from services.prediction_service import PredictionService

        service = PredictionService(get_config("testing"))
        status = service.health_status()
        assert status["model_loaded"] is False


def _make_fake_image_upload():
    """Build a minimal valid PNG upload for the validation-path tests."""
    import numpy as np
    import cv2
    from werkzeug.datastructures import FileStorage

    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return FileStorage(
        stream=io.BytesIO(buffer.tobytes()),
        filename="test_scan.png",
        content_type="image/png",
    )
