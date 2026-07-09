"""
Metrics used during training (Keras metric objects) and post-hoc
evaluation (confusion matrix, ROC/PR curves, classification report) used
by `evaluate.py` and the training pipeline's test-set evaluation step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display server in prod
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def get_training_metrics() -> List[tf.keras.metrics.Metric]:
    """Metrics tracked live during `model.fit()`."""
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
    ]


def compute_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict:
    """sklearn classification report as a JSON-serializable dict
    (precision/recall/F1 per class, plus macro/weighted averages)."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> str:
    """Render and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to '%s'", output_path)
    return output_path


def _one_hot(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    """Explicit one-hot encoding that always yields one column per class.

    `sklearn.preprocessing.label_binarize` is deliberately not used here:
    for exactly two classes it collapses to a single column (its
    documented binary-classification behavior), which breaks a uniform
    one-vs-rest loop over `class_names` regardless of class count.
    """
    return np.eye(num_classes, dtype=np.float32)[y_true]


def save_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> str:
    """One-vs-rest ROC curve per class."""
    y_true_bin = _one_hot(y_true, len(class_names))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, class_name in enumerate(class_names):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i], y_score[:, i], name=class_name, ax=ax
        )
    ax.set_title("ROC Curves (One-vs-Rest)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves to '%s'", output_path)
    return output_path


def save_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> str:
    """One-vs-rest precision-recall curve per class — more informative
    than ROC alone for imbalanced medical-imaging classes."""
    y_true_bin = _one_hot(y_true, len(class_names))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, class_name in enumerate(class_names):
        PrecisionRecallDisplay.from_predictions(
            y_true_bin[:, i], y_score[:, i], name=class_name, ax=ax
        )
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved precision-recall curves to '%s'", output_path)
    return output_path


def save_training_curves(history: dict, output_path: str) -> str:
    """Plot loss and accuracy curves from a Keras History.history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train_accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves to '%s'", output_path)
    return output_path


def save_history_json(history: dict, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    return output_path
