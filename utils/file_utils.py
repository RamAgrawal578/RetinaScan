"""
Filesystem utilities: secure upload handling and automatic dataset class
discovery. Nothing in this module hardcodes disease class names — classes
are always derived from the folder structure under `dataset/`.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from utils.logger import get_logger

logger = get_logger(__name__)


def allowed_file(filename: str, allowed_extensions: frozenset) -> bool:
    """Check a filename's extension against the configured allow-list."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def generate_unique_filename(original_filename: str) -> str:
    """Build a collision-safe, path-traversal-safe filename that preserves
    the original extension."""
    safe_name = secure_filename(original_filename) or "upload"
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "png"
    return f"{uuid.uuid4().hex}.{ext}"


def save_uploaded_file(
    file_storage: FileStorage, upload_dir: str, allowed_extensions: frozenset
) -> Optional[str]:
    """Validate and persist an uploaded file. Returns the absolute path on
    disk, or None if the file failed validation."""
    if file_storage is None or file_storage.filename == "":
        logger.warning("Upload rejected: empty file field")
        return None

    if not allowed_file(file_storage.filename, allowed_extensions):
        logger.warning(
            "Upload rejected: disallowed extension for '%s'",
            file_storage.filename,
        )
        return None

    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    unique_name = generate_unique_filename(file_storage.filename)
    destination = os.path.join(upload_dir, unique_name)
    file_storage.save(destination)
    logger.info("Saved upload '%s' -> '%s'", file_storage.filename, destination)
    return destination


def discover_classes(dataset_dir: str) -> List[str]:
    """Automatically detect class names from immediate sub-directories of
    the dataset folder. Returns an empty list if no dataset is present yet
    (e.g. before the user has downloaded one from Kaggle)."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        logger.warning("Dataset directory '%s' does not exist", dataset_dir)
        return []

    classes = sorted(
        entry.name
        for entry in dataset_path.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not classes:
        logger.warning(
            "No class sub-folders found under '%s'. Place your dataset "
            "there before training.",
            dataset_dir,
        )
    return classes


def load_class_names(saved_models_dir: str, dataset_dir: str) -> List[str]:
    """Resolve class names for inference: prefer the list captured at
    training time (saved_models/class_names.json) and fall back to a live
    scan of the dataset directory."""
    import json

    class_file = Path(saved_models_dir) / "class_names.json"
    if class_file.is_file():
        try:
            with open(class_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read class_names.json: %s", exc)

    return discover_classes(dataset_dir)


def ensure_directories(*paths: str) -> None:
    """Create any of the given directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
