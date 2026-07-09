"""
CNN model construction via transfer learning.

Architecture priority (CPU-first, per project requirements — tuned for
CPU-only training on a laptop with ~8GB RAM, no dedicated GPU):
    1. EfficientNetB0   (preferred — ~5.3M params, 224px native)
    2. MobileNetV3Large
    3. DenseNet121
    4. ResNet50V2

The build function always tries the configured `MODEL_ARCHITECTURE`
first and only falls back down the priority list if instantiating that
architecture fails outright (e.g. pretrained weights unreachable in a
fully offline environment) — it never silently substitutes a different
network without logging loudly.
"""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    MobileNetV3Large,
    ResNet50V2,
)

from utils.logger import get_logger

logger = get_logger(__name__)

_ARCHITECTURE_REGISTRY = {
    "EfficientNetB0": EfficientNetB0,
    "MobileNetV3Large": MobileNetV3Large,
    "DenseNet121": DenseNet121,
    "ResNet50V2": ResNet50V2,
}

_NATIVE_INPUT_SIZE = {
    "EfficientNetB0": 224,
    "MobileNetV3Large": 224,
    "DenseNet121": 224,
    "ResNet50V2": 224,
}


def native_input_size(architecture: str) -> int:
    """Return the architecture's canonical input resolution, used when the
    caller does not explicitly override IMAGE_SIZE."""
    return _NATIVE_INPUT_SIZE.get(architecture, 224)


def _instantiate_backbone(
    architecture: str,
    input_shape: Tuple[int, int, int],
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    backbone_cls = _ARCHITECTURE_REGISTRY[architecture]
    return backbone_cls(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )


def build_backbone(
    preferred_architecture: str,
    fallback_order: Tuple[str, ...],
    input_shape: Tuple[int, int, int],
    weights: str | None = "imagenet",
) -> Tuple[tf.keras.Model, str]:
    """Instantiate the preferred backbone, falling back down the priority
    list (and logging an explicit warning) only if the preferred choice
    cannot be built at all."""
    ordered_candidates = [preferred_architecture] + [
        arch for arch in fallback_order if arch != preferred_architecture
    ]

    last_error: Exception | None = None
    for architecture in ordered_candidates:
        if architecture not in _ARCHITECTURE_REGISTRY:
            logger.warning("Unknown architecture '%s', skipping", architecture)
            continue
        try:
            backbone = _instantiate_backbone(architecture, input_shape, weights=weights)
            if architecture != preferred_architecture:
                logger.warning(
                    "Preferred architecture '%s' unavailable — falling "
                    "back to '%s'.",
                    preferred_architecture,
                    architecture,
                )
            else:
                logger.info("Using architecture '%s'", architecture)
            return backbone, architecture
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to instantiate '%s': %s", architecture, exc
            )
            last_error = exc

    raise RuntimeError(
        "Could not instantiate any configured backbone architecture "
        f"from {ordered_candidates}"
    ) from last_error


def build_model(
    num_classes: int,
    architecture: str = "EfficientNetB0",
    fallback_order: Tuple[str, ...] = (
        "EfficientNetB0",
        "MobileNetV3Large",
        "DenseNet121",
        "ResNet50V2",
    ),
    image_size: int | None = None,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    freeze_backbone: bool = True,
    weights: str | None = "imagenet",
) -> Tuple[tf.keras.Model, str]:
    """Build the full classification model: backbone + custom head.

    `weights` defaults to `"imagenet"` for real training runs. Pass
    `weights=None` to build the same architecture with random
    initialization instead — useful for unit-testing the model wiring
    (input/output shapes, layer graph) without requiring network access
    to download pretrained weights.

    Returns the compiled-but-not-yet-`.compile()`-called Keras model and
    the name of the architecture that was actually used (which may differ
    from `architecture` if a fallback occurred).
    """
    resolved_size = image_size or native_input_size(architecture)
    input_shape = (resolved_size, resolved_size, 3)

    backbone, used_architecture = build_backbone(
        architecture, fallback_order, input_shape, weights=weights
    )
    backbone.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=input_shape, name="fundus_image")
    x = backbone(inputs, training=False if freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="head_dense_1",
    )(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(dropout_rate / 2, name="head_dropout_2")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs, outputs, name=f"retina_disease_{used_architecture.lower()}"
    )
    return model, used_architecture


def unfreeze_for_fine_tuning(
    model: tf.keras.Model, fine_tune_at_layer: int = -60
) -> tf.keras.Model:
    """Unfreeze the top `abs(fine_tune_at_layer)` layers of the backbone
    for a fine-tuning pass, keeping BatchNorm layers frozen so running
    statistics computed on ImageNet aren't destabilized by a small
    medical-imaging batch size."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        logger.warning(
            "No nested backbone sub-model found; unfreezing whole model."
        )
        model.trainable = True
        return model

    backbone.trainable = True
    freeze_until = len(backbone.layers) + fine_tune_at_layer
    for i, layer in enumerate(backbone.layers):
        if i < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    logger.info(
        "Fine-tuning enabled: last %d layers of '%s' unfrozen "
        "(BatchNorm layers kept frozen).",
        abs(fine_tune_at_layer),
        backbone.name,
    )
    return model