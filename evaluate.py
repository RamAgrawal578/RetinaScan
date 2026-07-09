#!/usr/bin/env python3
"""
Evaluate the trained model on the held-out test split.

Rebuilds the exact same deterministic train/val/test split used during
training (same seed + ratios), evaluates only on the test portion, and
writes:
    reports/confusion_matrix.png
    reports/roc_curves.png
    reports/precision_recall_curves.png
    reports/evaluation_report.json   (consumed by GET /api/metrics)

Usage:
    python evaluate.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf

from model.metrics import (
    compute_classification_report,
    save_confusion_matrix,
    save_precision_recall_curves,
    save_roc_curves,
)
from training.dataset_loader import build_dataset
from training.split import build_split
from utils.config_loader import load_runtime_config
from utils.file_utils import discover_classes, load_class_names
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_model(config):
    for filename in (
        f"{config.MODEL_NAME}_best.keras",
        f"{config.MODEL_NAME}_final.keras",
    ):
        path = os.path.join(config.SAVED_MODELS_DIR, filename)
        if os.path.isfile(path):
            logger.info("Loading model for evaluation: %s", path)
            return tf.keras.models.load_model(path)
    raise FileNotFoundError(
        "No trained model found in "
        f"'{config.SAVED_MODELS_DIR}'. Run `python train.py` first."
    )


def main() -> int:
    config = load_runtime_config()

    class_names = load_class_names(config.SAVED_MODELS_DIR, config.DATASET_DIR)
    if not class_names:
        logger.error(
            "No class names found (neither class_names.json nor a "
            "populated dataset/ directory). Cannot evaluate."
        )
        return 1

    try:
        model = _load_model(config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    split = build_split(
        dataset_dir=config.DATASET_DIR,
        class_names=class_names,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        test_ratio=config.TEST_SPLIT,
        seed=config.RANDOM_SEED,
    )

    if not split.test_paths:
        logger.error("Test split is empty — cannot evaluate.")
        return 1

    test_ds = build_dataset(
        split.test_paths,
        split.test_labels,
        image_size=config.IMAGE_SIZE,
        num_classes=len(class_names),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        augment=False,
        seed=config.RANDOM_SEED,
    )

    logger.info("Running inference on %d held-out test images...", len(split.test_paths))
    y_true, y_score = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_score.append(preds)
        y_true.append(labels.numpy())

    y_score = np.concatenate(y_score, axis=0)
    y_true_onehot = np.concatenate(y_true, axis=0)
    y_true_labels = np.argmax(y_true_onehot, axis=1)
    y_pred_labels = np.argmax(y_score, axis=1)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    report_dict = compute_classification_report(
        y_true_labels, y_pred_labels, class_names
    )
    save_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        class_names,
        os.path.join(config.REPORTS_DIR, "confusion_matrix.png"),
    )
    save_roc_curves(
        y_true_labels,
        y_score,
        class_names,
        os.path.join(config.REPORTS_DIR, "roc_curves.png"),
    )
    save_precision_recall_curves(
        y_true_labels,
        y_score,
        class_names,
        os.path.join(config.REPORTS_DIR, "precision_recall_curves.png"),
    )

    evaluation_report_path = os.path.join(
        config.REPORTS_DIR, "evaluation_report.json"
    )
    with open(evaluation_report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "num_test_samples": len(split.test_paths),
                "class_names": class_names,
                "classification_report": report_dict,
                "overall_accuracy": report_dict.get("accuracy"),
            },
            fh,
            indent=2,
        )

    logger.info(
        "Evaluation complete. Overall accuracy: %.4f. Report saved to '%s'.",
        report_dict.get("accuracy", 0.0),
        evaluation_report_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
