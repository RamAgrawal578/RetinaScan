"""Centralized 404 / 500 / 413 error handling, registered on the app."""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from utils.logger import get_logger

logger = get_logger(__name__)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Not found"}), 404
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def payload_too_large(error):
        message = "Uploaded file exceeds the maximum allowed size."
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": message}), 413
        return render_template("404.html", message=message), 413

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception("Unhandled server error: %s", error)
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Internal server error"}), 500
        return render_template("500.html"), 500
