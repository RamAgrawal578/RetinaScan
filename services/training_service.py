"""
Training service: the single entry point `train.py` calls. Wires
together class discovery, dataset splitting, `tf.data` pipeline
construction, and the `Trainer` class, so the CLI script itself stays a
thin wrapper.
"""
from __future__ import annotations

from typing import Dict

from model.trainer import Trainer
from training.dataset_loader import build_train_val_test_datasets
from training.split import build_split, compute_class_weights
from utils.file_utils import discover_classes
from utils.logger import get_logger
from utils.validators import validate_class_names

logger = get_logger(__name__)


class TrainingService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> Dict:
        """Discover classes, build the stratified split, build `tf.data`
        pipelines, and run the full two-phase training loop."""
        class_names = discover_classes(self.config.DATASET_DIR)
        validation = validate_class_names(class_names)
        if not validation.is_valid:
            for error in validation.errors:
                logger.error(error)
            raise RuntimeError(
                "Dataset validation failed: " + "; ".join(validation.errors)
            )

        logger.info("Discovered %d classes: %s", len(class_names), class_names)

        split = build_split(
            dataset_dir=self.config.DATASET_DIR,
            class_names=class_names,
            train_ratio=self.config.TRAIN_SPLIT,
            val_ratio=self.config.VAL_SPLIT,
            test_ratio=self.config.TEST_SPLIT,
            seed=self.config.RANDOM_SEED,
        )

        class_weights = compute_class_weights(
            split.train_labels, num_classes=len(class_names)
        )

        train_ds, val_ds, test_ds = build_train_val_test_datasets(
            split,
            image_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            seed=self.config.RANDOM_SEED,
        )

        trainer = Trainer(self.config)
        history = trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            class_names=class_names,
            class_weight=class_weights,
        )

        return {
            "history": history,
            "class_names": class_names,
            "architecture": trainer.used_architecture,
            "test_dataset": test_ds,
        }
