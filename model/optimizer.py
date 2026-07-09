"""
Optimizer construction, including mixed-precision policy setup and GPU
auto-detection with graceful CPU fallback.
"""
from __future__ import annotations

import tensorflow as tf

from utils.logger import get_logger

logger = get_logger(__name__)


def configure_hardware(use_mixed_precision: bool = True) -> str:
    """Detect available GPUs, enable memory growth (avoids TF grabbing all
    VRAM up front), and set the global mixed-precision policy. Returns a
    short string describing the resolved compute device for logging."""
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logger.warning("Could not set memory growth on %s: %s", gpu, exc)
        device_desc = f"GPU x{len(gpus)}"
    else:
        device_desc = "CPU"
        logger.warning(
            "No GPU detected — falling back to CPU. Training "
            "EfficientNetB7 on CPU will be significantly slower; "
            "consider a smaller batch size or a GPU-backed environment."
        )

    if use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision policy 'mixed_float16' enabled.")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        if use_mixed_precision and not gpus:
            logger.info(
                "Mixed precision requested but no GPU present; "
                "using float32 (mixed precision offers no benefit on CPU)."
            )

    logger.info("Compute device resolved: %s", device_desc)
    return device_desc


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Global random seed set to %d", seed)


def build_optimizer(
    learning_rate: float = 1e-4, use_mixed_precision: bool = True
) -> tf.keras.optimizers.Optimizer:
    """Construct the Adam optimizer, wrapped with loss scaling when mixed
    precision is active (required to avoid float16 gradient underflow)."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if (
        use_mixed_precision
        and tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    ):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        logger.info("Wrapped optimizer with LossScaleOptimizer for fp16.")

    return optimizer
