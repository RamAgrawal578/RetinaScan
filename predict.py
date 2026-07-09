#!/usr/bin/env python3
"""
Run a single prediction from the command line, without starting Flask.
Useful for quick sanity checks after training, or batch scripting.

Usage:
    python predict.py path/to/fundus_image.jpg
    python predict.py path/to/fundus_image.jpg --report
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from model.predictor import Predictor
from services.report_service import generate_prediction_report
from utils.config_loader import load_runtime_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single retina prediction.")
    parser.add_argument("image_path", help="Path to a fundus image file.")
    parser.add_argument(
        "--report", action="store_true", help="Also generate a PDF report."
    )
    parser.add_argument(
        "--no-gradcam", action="store_true", help="Skip Grad-CAM heatmap generation."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config()

    predictor = Predictor(config)
    if not predictor.load():
        logger.error(
            "No trained model available in '%s'. Run `python train.py` first.",
            config.SAVED_MODELS_DIR,
        )
        return 1

    result = predictor.predict(args.image_path, generate_gradcam=not args.no_gradcam)
    if not result.is_valid:
        logger.error("Prediction failed: %s", result.errors)
        return 1

    output = asdict(result)
    output.pop("gradcam_base64", None)  # too large for terminal output
    print(json.dumps(output, indent=2))

    if args.report:
        report_path = generate_prediction_report(
            result, args.image_path, config.REPORTS_DIR
        )
        print(f"\nPDF report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
