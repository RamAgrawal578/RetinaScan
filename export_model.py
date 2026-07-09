#!/usr/bin/env python3
"""
Export the trained Keras model to portable formats for deployment
outside this repo (e.g. TF Serving, mobile, or a lighter-weight
inference server).

Usage:
    python export_model.py                     # exports SavedModel format
    python export_model.py --tflite             # also exports a .tflite file
    python export_model.py --tflite --quantize   # quantized .tflite (smaller/faster)
"""
from __future__ import annotations

import argparse
import os
import sys

import tensorflow as tf

from utils.config_loader import load_runtime_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained model.")
    parser.add_argument(
        "--tflite", action="store_true", help="Also export a TensorFlow Lite model."
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range quantization to the TFLite export.",
    )
    return parser.parse_args()


def _resolve_model_path(config) -> str:
    for filename in (
        f"{config.MODEL_NAME}_best.keras",
        f"{config.MODEL_NAME}_final.keras",
    ):
        path = os.path.join(config.SAVED_MODELS_DIR, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No trained model found in '{config.SAVED_MODELS_DIR}'. "
        "Run `python train.py` first."
    )


def main() -> int:
    args = parse_args()
    config = load_runtime_config()

    try:
        model_path = _resolve_model_path(config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    model = tf.keras.models.load_model(model_path)

    export_dir = os.path.join(
        config.SAVED_MODELS_DIR, f"{config.MODEL_NAME}_savedmodel"
    )
    model.export(export_dir)
    logger.info("Exported SavedModel format to '%s'", export_dir)

    if args.tflite:
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        if args.quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Applying dynamic-range quantization for TFLite export.")

        tflite_model = converter.convert()
        tflite_path = os.path.join(
            config.SAVED_MODELS_DIR, f"{config.MODEL_NAME}.tflite"
        )
        with open(tflite_path, "wb") as fh:
            fh.write(tflite_model)
        logger.info("Exported TFLite model to '%s'", tflite_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
