#!/usr/bin/env python3
"""
Train the retina disease classification model.

Usage:
    python train.py
    python train.py --env production
    APP_ENV=production python train.py

Automatically:
    - Detects dataset classes from `dataset/<ClassName>/*`
    - Splits train/val/test with class-stratified sampling
    - Resumes from the last checkpoint if one exists in `saved_models/`
    - Runs frozen-backbone transfer learning, then fine-tuning
    - Saves the best model, class names, version metadata, and curves
"""
from __future__ import annotations

import argparse
import sys

from services.training_service import TrainingService
from utils.config_loader import load_runtime_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the retina disease model.")
    parser.add_argument(
        "--env",
        default=None,
        help="Configuration environment: development | testing | production",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.env)

    logger.info(
        "Starting training run: architecture=%s, image_size=%d, "
        "batch_size=%d, epochs=%d (+%d fine-tune)",
        config.MODEL_ARCHITECTURE,
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        config.EPOCHS,
        config.FINE_TUNE_EPOCHS,
    )

    try:
        service = TrainingService(config)
        result = service.run()
    except RuntimeError as exc:
        logger.error("Training aborted: %s", exc)
        return 1

    logger.info(
        "Training complete. Architecture used: %s. Classes: %s",
        result["architecture"],
        result["class_names"],
    )
    logger.info(
        "Run `python evaluate.py` to generate the held-out test-set "
        "evaluation report (confusion matrix, ROC/PR curves, metrics)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
