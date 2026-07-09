"""
REST API blueprint.

Endpoints:
    POST /api/predict     -> run a prediction on an uploaded image
    GET  /api/health       -> liveness / model-loaded status
    GET  /api/model-info   -> architecture, classes, training metadata
    GET  /api/version      -> app + model version
    GET  /api/metrics      -> latest evaluation metrics, if available
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/predict", methods=["POST"])
def api_predict():
    service = current_app.config["PREDICTION_SERVICE"]
    file_storage = request.files.get("image")
    generate_report = request.args.get("report", "false").lower() == "true"

    result = service.handle_upload_and_predict(
        file_storage, generate_report=generate_report
    )
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@api_bp.route("/health", methods=["GET"])
def api_health():
    service = current_app.config["PREDICTION_SERVICE"]
    status = service.health_status()
    return jsonify(
        {
            "status": "ok",
            "model_loaded": status["model_loaded"],
            "num_classes": status["num_classes"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@api_bp.route("/model-info", methods=["GET"])
def api_model_info():
    service = current_app.config["PREDICTION_SERVICE"]
    predictor = service.predictor
    predictor.is_loaded or predictor.load()
    return jsonify(
        {
            "architecture": predictor.version_info.get(
                "architecture", current_app.config["MODEL_ARCHITECTURE"]
            ),
            "class_names": predictor.class_names,
            "num_classes": len(predictor.class_names),
            "trained_at": predictor.version_info.get("trained_at"),
            "image_size": current_app.config["IMAGE_SIZE"],
        }
    )


@api_bp.route("/version", methods=["GET"])
def api_version():
    return jsonify(
        {
            "app_version": current_app.config["APP_VERSION"],
            "model_name": current_app.config["MODEL_NAME"],
        }
    )


@api_bp.route("/metrics", methods=["GET"])
def api_metrics():
    """Return the most recent test-set evaluation report, if
    `evaluate.py` has been run."""
    report_path = os.path.join(
        current_app.config["REPORTS_DIR"], "evaluation_report.json"
    )
    if not os.path.isfile(report_path):
        return (
            jsonify(
                {
                    "available": False,
                    "message": "No evaluation report found. Run "
                    "`python evaluate.py` after training.",
                }
            ),
            404,
        )

    with open(report_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return jsonify({"available": True, "metrics": metrics})
