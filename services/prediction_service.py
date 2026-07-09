"""
Prediction service: the single entry point the Flask routes call for
"upload an image, get a prediction" — keeps view functions thin and
keeps this logic independently unit-testable.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

from werkzeug.datastructures import FileStorage

from model.predictor import Predictor, PredictionResult
from services.report_service import generate_prediction_report
from utils.file_utils import save_uploaded_file
from utils.logger import get_logger
from utils.validators import validate_image_file, validate_upload_request

logger = get_logger(__name__)


class PredictionService:
    """Wraps a `Predictor` instance with upload handling, validation, and
    optional PDF report generation."""

    def __init__(self, config) -> None:
        self.config = config
        self.predictor = Predictor(config)

    def health_status(self) -> Dict:
        loaded = self.predictor.is_loaded or self.predictor.load()
        return {
            "model_loaded": loaded,
            "num_classes": len(self.predictor.class_names),
        }

    def handle_upload_and_predict(
        self, file_storage: FileStorage, generate_report: bool = False
    ) -> Dict:
        """Validate an uploaded file, run the prediction pipeline, and
        optionally generate a PDF report. Always returns a plain dict
        (JSON-serializable) with either the prediction or a list of
        validation errors."""
        content_length = None
        try:
            file_storage.stream.seek(0, 2)
            content_length = file_storage.stream.tell()
            file_storage.stream.seek(0)
        except (AttributeError, OSError):
            pass

        request_validation = validate_upload_request(
            has_file=file_storage is not None,
            filename=(file_storage.filename if file_storage else ""),
            allowed_extensions=self.config.ALLOWED_EXTENSIONS,
            content_length=content_length,
            max_content_length=self.config.MAX_CONTENT_LENGTH,
        )
        if not request_validation.is_valid:
            return {"success": False, "errors": request_validation.errors}

        saved_path = save_uploaded_file(
            file_storage, self.config.UPLOAD_DIR, self.config.ALLOWED_EXTENSIONS
        )
        if saved_path is None:
            return {
                "success": False,
                "errors": ["Failed to save the uploaded file."],
            }

        content_validation = validate_image_file(saved_path)
        if not content_validation.is_valid:
            return {"success": False, "errors": content_validation.errors}

        prediction: PredictionResult = self.predictor.predict(saved_path)
        if not prediction.is_valid:
            return {"success": False, "errors": prediction.errors}

        response: Dict = {"success": True, "prediction": asdict(prediction)}
        response["uploaded_image_path"] = saved_path

        if generate_report:
            report_path = generate_prediction_report(
                prediction, saved_path, self.config.REPORTS_DIR
            )
            response["report_path"] = report_path

        return response
