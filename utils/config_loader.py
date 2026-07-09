"""
Small helper for scripts (train.py, evaluate.py, predict.py) that need a
config object outside of a Flask application context.
"""
from __future__ import annotations

import os
from typing import Type

from config.settings import BaseConfig, get_config
from utils.file_utils import ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)


def load_runtime_config(env_name: str | None = None) -> Type[BaseConfig]:
    """Resolve the active config class and make sure its directories
    exist. Used by every standalone CLI entry point so they behave
    consistently with the Flask app."""
    config_class = get_config(env_name)
    ensure_directories(
        config_class.DATASET_DIR,
        config_class.SAVED_MODELS_DIR,
        config_class.UPLOAD_DIR,
        config_class.REPORTS_DIR,
        config_class.LOGS_DIR,
        config_class.TENSORBOARD_DIR,
    )
    logger.info(
        "Loaded '%s' configuration (env=%s)",
        config_class.__name__,
        env_name or os.environ.get("APP_ENV", "default"),
    )
    return config_class
