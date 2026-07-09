"""Main blueprint: home, about, model-info, and contact pages."""
from __future__ import annotations

from flask import Blueprint, current_app, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/contact")
def contact():
    return render_template("contact.html")


@main_bp.route("/model-info")
def model_info_page():
    service = current_app.config["PREDICTION_SERVICE"]
    status = service.health_status()
    predictor = service.predictor
    info = {
        "architecture": predictor.version_info.get(
            "architecture", current_app.config["MODEL_ARCHITECTURE"]
        ),
        "trained_at": predictor.version_info.get("trained_at", "N/A"),
        "num_classes": status["num_classes"],
        "class_names": predictor.class_names,
        "app_version": current_app.config["APP_VERSION"],
        "model_loaded": status["model_loaded"],
    }
    return render_template("about.html", model_info=info)
