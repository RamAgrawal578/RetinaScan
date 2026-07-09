"""Predict blueprint: the human-facing upload -> result page flow."""
from __future__ import annotations

import os

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")


@predict_bp.route("/predict", methods=["POST"])
def predict_submit():
    service = current_app.config["PREDICTION_SERVICE"]
    file_storage = request.files.get("image")
    generate_report = request.form.get("generate_report") == "on"

    result = service.handle_upload_and_predict(
        file_storage, generate_report=generate_report
    )

    if not result.get("success"):
        for error in result.get("errors", ["Prediction failed."]):
            flash(error, "danger")
        return redirect(url_for("predict.predict_form"))

    uploaded_filename = os.path.basename(result["uploaded_image_path"])
    report_filename = (
        os.path.basename(result["report_path"]) if result.get("report_path") else None
    )

    return render_template(
        "result.html",
        prediction=result["prediction"],
        uploaded_filename=uploaded_filename,
        report_filename=report_filename,
    )


@predict_bp.route("/predict/image/<path:filename>")
def uploaded_image(filename: str):
    upload_dir = current_app.config["UPLOAD_DIR"]
    return send_from_directory(upload_dir, os.path.basename(filename))


@predict_bp.route("/predict/report/<path:filename>")
def download_report(filename: str):
    reports_dir = current_app.config["REPORTS_DIR"]
    safe_path = os.path.join(reports_dir, os.path.basename(filename))
    if not os.path.isfile(safe_path):
        flash("Report not found or has expired.", "warning")
        return redirect(url_for("predict.predict_form"))
    return send_file(safe_path, as_attachment=True, download_name="retina_report.pdf")
