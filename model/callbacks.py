"""
Keras callback construction for the training pipeline: checkpointing,
early stopping, LR scheduling, TensorBoard, and CSV logging.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List

import tensorflow as tf

from utils.logger import get_logger

logger = get_logger(__name__)


def build_callbacks(
    saved_models_dir: str,
    logs_dir: str,
    tensorboard_dir: str,
    model_name: str = "retina_disease_model",
    early_stopping_patience: int = 8,
    reduce_lr_patience: int = 4,
    reduce_lr_factor: float = 0.5,
    min_learning_rate: float = 1e-7,
    monitor: str = "val_accuracy",
    monitor_mode: str = "max",
) -> List[tf.keras.callbacks.Callback]:
    """Build the full callback stack shared by the initial-training and
    fine-tuning phases."""
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        saved_models_dir, f"{model_name}_best.keras"
    )
    run_tensorboard_dir = os.path.join(tensorboard_dir, run_id)
    csv_log_path = os.path.join(logs_dir, f"training_log_{run_id}.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode=monitor_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=monitor_mode,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=run_tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path, append=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    logger.info(
        "Built %d callbacks (checkpoint='%s', tensorboard='%s', csv='%s')",
        len(callbacks),
        checkpoint_path,
        run_tensorboard_dir,
        csv_log_path,
    )
    return callbacks
