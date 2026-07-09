"""
Trainer: orchestrates the two-phase transfer-learning training run
(frozen-backbone warm-up, then fine-tuning), automatic checkpoint
resume, best-model selection, and model versioning/export.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf

from model.build_model import build_model, unfreeze_for_fine_tuning
from model.callbacks import build_callbacks
from model.losses import get_loss
from model.metrics import get_training_metrics, save_history_json, save_training_curves
from model.optimizer import build_optimizer, configure_hardware, set_global_seed
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """High-level training orchestrator used by `train.py`."""

    def __init__(self, config) -> None:
        self.config = config
        self.class_names: List[str] = []
        self.model: Optional[tf.keras.Model] = None
        self.used_architecture: str = config.MODEL_ARCHITECTURE
        self.history: Dict[str, List[float]] = {}

        set_global_seed(config.RANDOM_SEED)
        self.device_desc = configure_hardware(config.USE_MIXED_PRECISION)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _checkpoint_path(self) -> str:
        return os.path.join(
            self.config.SAVED_MODELS_DIR, f"{self.config.MODEL_NAME}_best.keras"
        )

    def build_or_resume(self, num_classes: int) -> tf.keras.Model:
        """Automatically resume from the last checkpoint if one exists;
        otherwise build a fresh model from the configured backbone."""
        checkpoint_path = self._checkpoint_path()

        if os.path.isfile(checkpoint_path):
            logger.info("Resuming from checkpoint: %s", checkpoint_path)
            self.model = tf.keras.models.load_model(checkpoint_path)
            self.used_architecture = self.config.MODEL_ARCHITECTURE
            return self.model

        logger.info("No checkpoint found — building a new model.")
        model, used_architecture = build_model(
            num_classes=num_classes,
            architecture=self.config.MODEL_ARCHITECTURE,
            fallback_order=self.config.ARCHITECTURE_FALLBACK_ORDER,
            image_size=self.config.IMAGE_SIZE,
            dropout_rate=self.config.DROPOUT_RATE,
            l2_reg=self.config.L2_REGULARIZATION,
            freeze_backbone=True,
            weights=getattr(self.config, "PRETRAINED_WEIGHTS", "imagenet"),
        )
        self.model = model
        self.used_architecture = used_architecture
        return model

    def compile_model(self, learning_rate: float) -> None:
        assert self.model is not None, "Call build_or_resume() first."
        optimizer = build_optimizer(
            learning_rate=learning_rate,
            use_mixed_precision=self.config.USE_MIXED_PRECISION,
        )
        self.model.compile(
            optimizer=optimizer,
            loss=get_loss("categorical_crossentropy"),
            metrics=get_training_metrics(),
        )

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------
    def train(
        self,
        train_dataset,
        val_dataset,
        class_names: List[str],
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, List[float]]:
        """Run phase 1 (frozen backbone) then phase 2 (fine-tuning),
        merging both histories and saving the final artifacts."""
        self.class_names = class_names
        num_classes = len(class_names)

        self.build_or_resume(num_classes)
        self.compile_model(self.config.LEARNING_RATE)

        callbacks = build_callbacks(
            saved_models_dir=self.config.SAVED_MODELS_DIR,
            logs_dir=self.config.LOGS_DIR,
            tensorboard_dir=self.config.TENSORBOARD_DIR,
            model_name=self.config.MODEL_NAME,
            early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE,
            reduce_lr_patience=self.config.REDUCE_LR_PATIENCE,
            reduce_lr_factor=self.config.REDUCE_LR_FACTOR,
            min_learning_rate=self.config.MIN_LEARNING_RATE,
        )

        logger.info(
            "Phase 1/2: transfer learning (frozen backbone) — %d epochs",
            self.config.EPOCHS,
        )
        history_phase1 = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )

        logger.info(
            "Phase 2/2: fine-tuning top layers — %d epochs",
            self.config.FINE_TUNE_EPOCHS,
        )
        self.model = unfreeze_for_fine_tuning(
            self.model, fine_tune_at_layer=self.config.FINE_TUNE_AT_LAYER
        )
        self.compile_model(self.config.FINE_TUNE_LEARNING_RATE)

        initial_epoch = len(history_phase1.history.get("loss", []))
        history_phase2 = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=initial_epoch + self.config.FINE_TUNE_EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )

        merged_history = {
            key: history_phase1.history.get(key, []) + history_phase2.history.get(key, [])
            for key in set(history_phase1.history) | set(history_phase2.history)
        }
        self.history = merged_history
        self._save_artifacts(merged_history)
        return merged_history

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------
    def _save_artifacts(self, history: Dict[str, List[float]]) -> None:
        os.makedirs(self.config.SAVED_MODELS_DIR, exist_ok=True)

        class_names_path = os.path.join(
            self.config.SAVED_MODELS_DIR, "class_names.json"
        )
        with open(class_names_path, "w", encoding="utf-8") as fh:
            json.dump(self.class_names, fh, indent=2)

        version_path = os.path.join(
            self.config.SAVED_MODELS_DIR, "model_version.json"
        )
        version_info = {
            "model_name": self.config.MODEL_NAME,
            "architecture": self.used_architecture,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "image_size": self.config.IMAGE_SIZE,
            "app_version": self.config.APP_VERSION,
            "device": self.device_desc,
        }
        with open(version_path, "w", encoding="utf-8") as fh:
            json.dump(version_info, fh, indent=2)

        history_path = os.path.join(self.config.LOGS_DIR, "training_history.json")
        save_history_json(history, history_path)

        curves_path = os.path.join(
            self.config.REPORTS_DIR, "training_curves.png"
        )
        save_training_curves(history, curves_path)

        final_model_path = os.path.join(
            self.config.SAVED_MODELS_DIR, f"{self.config.MODEL_NAME}_final.keras"
        )
        self.model.save(final_model_path)

        logger.info(
            "Training artifacts saved: model='%s', classes='%s', "
            "version='%s', history='%s'",
            final_model_path,
            class_names_path,
            version_path,
            history_path,
        )
