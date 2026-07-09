"""
Centralized logging configuration.

Every module in the project should call `get_logger(__name__)` instead of
constructing its own handlers, so log format, level, and destinations stay
consistent across the Flask app, training scripts, and tests.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def _configure_root(logs_dir: str | None = None) -> None:
    """Attach a console handler and a rotating file handler to the root
    logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logs_dir = logs_dir or os.environ.get("LOGS_DIR", "logs")
    try:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(logs_dir, "app.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only filesystem (some PaaS ephemeral containers) — console
        # logging alone is fine in that case.
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    _configure_root()
    return logging.getLogger(name)
