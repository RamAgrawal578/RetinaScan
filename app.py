"""
Flask application factory.

Usage:
    from app import create_app
    app = create_app()

Or run directly for local development:
    python app.py
"""
from __future__ import annotations

import os

from flask import Flask

from config.settings import get_config
from routes.api_routes import api_bp
from routes.error_handlers import register_error_handlers
from routes.main_routes import main_bp
from routes.predict_routes import predict_bp
from services.prediction_service import PredictionService
from utils.file_utils import ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app(config_name: str | None = None) -> Flask:
    """Application factory: builds and configures a Flask app instance.

    Kept free of side effects at import time (no model loading, no
    directory creation at module scope) so it is safe to import for
    testing, WSGI servers, and CLI tooling alike.
    """
    app = Flask(__name__)

    config_class = get_config(config_name)
    app.config.from_object(config_class)
    config_class.init_app(app)

    ensure_directories(
        app.config["DATASET_DIR"],
        app.config["SAVED_MODELS_DIR"],
        app.config["UPLOAD_DIR"],
        app.config["REPORTS_DIR"],
        app.config["LOGS_DIR"],
        app.config["TENSORBOARD_DIR"],
    )

    # --- Security-adjacent, "ready" configuration -----------------------
    # CSRF protection: enable automatically if flask-wtf is installed
    # (see requirements.txt). Kept optional so the API-only / stateless
    # JSON endpoints under /api are unaffected by form-based CSRF tokens.
    try:
        from flask_wtf import CSRFProtect

        csrf = CSRFProtect()
        csrf.init_app(app)
        csrf.exempt(api_bp)
        logger.info("CSRF protection enabled for HTML form routes.")
    except ImportError:
        logger.info(
            "flask-wtf not installed — CSRF protection skipped. "
            "Install it and it will activate automatically (CSRF-ready)."
        )

    # Rate limiting: enable automatically if flask-limiter is installed.
    if app.config.get("RATE_LIMIT_ENABLED"):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address

            limiter = Limiter(
                get_remote_address,
                app=app,
                default_limits=[app.config["RATE_LIMIT_DEFAULT"]],
            )
            app.config["LIMITER"] = limiter
            logger.info(
                "Rate limiting enabled: %s", app.config["RATE_LIMIT_DEFAULT"]
            )
        except ImportError:
            logger.info(
                "flask-limiter not installed — rate limiting skipped "
                "(rate-limit-ready; install to activate)."
            )

    # --- Service layer (holds the lazily-loaded model) -------------------
    # NOTE: the service/model layer (Predictor, Trainer, TrainingService)
    # uses dot-attribute config access (e.g. `config.SAVED_MODELS_DIR`) so
    # it can be reused unmodified from CLI scripts (train.py, predict.py)
    # that have no Flask app context. We therefore hand it `config_class`
    # itself rather than Flask's dict-like `app.config`.
    app.config["PREDICTION_SERVICE"] = PredictionService(config_class)

    # --- Blueprints --------------------------------------------------
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(api_bp)
    register_error_handlers(app)

    logger.info(
        "Flask app created (env=%s, debug=%s)",
        config_class.__name__,
        app.config["DEBUG"],
    )
    return app


# Module-level app object for WSGI servers (gunicorn `app:app`) that
# expect an importable instance rather than calling the factory.
app = create_app()


if __name__ == "__main__":
    # Local development server only — Render/production uses gunicorn
    # via the Procfile, never this block.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
