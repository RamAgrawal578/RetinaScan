"""
Training-time data augmentation.

Implemented as a small stack of `tf.keras.layers` so it runs on-device
(GPU/TPU) as part of the model graph via `tf.data`'s `.map()`, rather than
as a CPU-bound Python loop. Augmentation layers are no-ops during
inference (`training=False`), so the same model works unmodified for
prediction.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation_pipeline(seed: int = 42) -> tf.keras.Sequential:
    """Return a Sequential stack of augmentation layers appropriate for
    fundus images: geometric transforms that preserve diagnostic content
    (no aggressive color inversion, since color is diagnostically
    meaningful for conditions like AMD and DR)."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.08, seed=seed, fill_mode="constant"),
            layers.RandomZoom(0.10, seed=seed, fill_mode="constant"),
            layers.RandomTranslation(
                0.05, 0.05, seed=seed, fill_mode="constant"
            ),
            layers.RandomContrast(0.15, seed=seed),
            layers.RandomBrightness(0.10, seed=seed),
        ],
        name="augmentation_pipeline",
    )


def augment_dataset(
    dataset: tf.data.Dataset, seed: int = 42
) -> tf.data.Dataset:
    """Apply the augmentation pipeline to a labeled `tf.data.Dataset` of
    (image, label) pairs, running only during training."""
    pipeline = build_augmentation_pipeline(seed=seed)

    def _augment(images, labels):
        return pipeline(images, training=True), labels

    return dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
