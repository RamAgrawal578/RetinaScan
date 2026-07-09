"""
Shared pytest configuration.

Environment variables are set at import time (before any other test
module imports `app`/`config`), because `config/settings.py` resolves
its directory paths once, at class-definition time, from the process
environment. This keeps the test suite from writing into the real
`saved_models/`, `uploads/`, etc. directories used by local development.
"""
from __future__ import annotations

import os
import tempfile

_TEST_ROOT = tempfile.mkdtemp(prefix="retinaai_test_")

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATASET_DIR", os.path.join(_TEST_ROOT, "dataset"))
os.environ.setdefault("SAVED_MODELS_DIR", os.path.join(_TEST_ROOT, "saved_models"))
os.environ.setdefault("UPLOAD_DIR", os.path.join(_TEST_ROOT, "uploads"))
os.environ.setdefault("REPORTS_DIR", os.path.join(_TEST_ROOT, "reports"))
os.environ.setdefault("LOGS_DIR", os.path.join(_TEST_ROOT, "logs"))
os.environ.setdefault("TENSORBOARD_DIR", os.path.join(_TEST_ROOT, "logs", "tensorboard"))
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def flask_app():
    from app import create_app

    app = create_app("testing")
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()
