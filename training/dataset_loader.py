"""
Builds `tf.data.Dataset` pipelines for train/validation/test splits.

Preprocessing (resize, crop, CLAHE, etc.) runs in NumPy/OpenCV via
`tf.py_function` (necessary since OpenCV isn't graph-traceable), while
augmentation runs as native Keras layers so it stays on-device and fast.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from preprocessing.augmentation import augment_dataset
from preprocessing.preprocess import preprocess_image
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

_cfg = get_config()
_CACHE_DIR = Path(_cfg.BASE_DIR) / ".tfcache"

# CHANGED: bounded parallelism instead of tf.data.AUTOTUNE.
#
# AUTOTUNE lets tf.data pick however many parallel threads it wants for
# the `.map()` call below, and each thread calls into OpenCV. Now that
# preprocessing/preprocess.py pins OpenCV to a single internal thread
# (cv2.setNumThreads(1)), it's safe — and much faster — to let *this*
# outer level own the parallelism instead of stacking two layers of
# threading on top of each other. Leave one core free for the main
# training thread / OS so the machine stays responsive.
_PARALLEL_CALLS = max(1, (os.cpu_count() or 4) - 1)


def _load_and_preprocess(path: str, image_size: int) -> np.ndarray:
    result = preprocess_image(
        path.numpy().decode("utf-8"),
        target_size=image_size,
        apply_enhancement=True,
        normalize=True,
        # CHANGED: skip per-image blur/quality stats during training —
        # that diagnostic is for single-image inference QA, runs on the
        # full-resolution original before resize, and adds real time
        # once multiplied across a multi-thousand-image dataset.
        compute_stats=False,
    )
    if not result.is_valid:
        logger.warning(
            "Skipping unreadable/invalid image during dataset load: %s",
            result.errors,
        )
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    return result.image


class _ProgressLogger:
    """Thread-safe counter that logs periodic progress while the
    (slow, CPU-bound) preprocessing pipeline runs.

    Previously the console printed one line ("first epoch will be slow
    while the cache is built") and then went completely silent — often
    for a long time on a large, merged dataset on CPU-only hardware.
    That silence is indistinguishable from an actual hang. This makes
    progress visible so "slow but working" is obvious.
    """

    def __init__(self, split_name: str, total: int):
        self.split_name = split_name
        self.total = total
        self.count = 0
        self.start = time.time()
        self.lock = threading.Lock()
        self.log_every = max(50, total // 25) if total else 100

    def tick(self) -> None:
        with self.lock:
            self.count += 1
            if self.count % self.log_every == 0 or self.count == self.total:
                elapsed = time.time() - self.start
                rate = self.count / elapsed if elapsed > 0 else 0.0
                remaining = self.total - self.count
                eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                logger.info(
                    "[%s] preprocessed %d/%d images (%.2f img/s, "
                    "elapsed %.1fm, ETA %.1fm)",
                    self.split_name,
                    self.count,
                    self.total,
                    rate,
                    elapsed / 60,
                    eta_min,
                )


def _make_tf_loader(image_size: int, num_classes: int, progress: "_ProgressLogger"):
    def _loader(path, label):
        def _wrapped(p):
            result = _load_and_preprocess(p, image_size)
            progress.tick()
            return result

        image = tf.py_function(
            func=_wrapped,
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape((image_size, image_size, 3))
        label_one_hot = tf.one_hot(label, depth=num_classes)
        return image, label_one_hot

    return _loader


def build_dataset(
    paths: List[str],
    labels: List[int],
    image_size: int,
    num_classes: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int = 42,
    cache_name: str | None = None,
) -> tf.data.Dataset:
    """Build a batched, prefetched `tf.data.Dataset` for one split.

    `cache_name`, if given, enables an on-disk `.cache()` keyed by that
    name ("train"/"val"/"test") so the expensive per-image CV pipeline
    only runs once instead of once per epoch.
    """
    if not paths:
        raise ValueError(
            "No image paths were provided — is the dataset directory "
            "populated? See dataset/README.md."
        )

    logger.info(
        "[%s] %d images queued for preprocessing (CPU-bound, first pass "
        "only — progress logs below).",
        cache_name or "split",
        len(paths),
    )

    path_ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )

    if shuffle:
        path_ds = path_ds.shuffle(
            buffer_size=min(len(paths), 2000), seed=seed,
            reshuffle_each_iteration=True,
        )

    progress = _ProgressLogger(cache_name or "split", total=len(paths))
    loader = _make_tf_loader(image_size, num_classes, progress)
    # CHANGED: bounded parallelism (see _PARALLEL_CALLS above) instead
    # of tf.data.AUTOTUNE.
    dataset = path_ds.map(loader, num_parallel_calls=_PARALLEL_CALLS)

    if cache_name:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = str(_CACHE_DIR / cache_name)
        dataset = dataset.cache(cache_path)
        logger.info(
            "Enabled disk cache for '%s' split at %s (first epoch will "
            "be slow while the cache is built; watch the "
            "'[%s] preprocessed X/Y images' lines above to confirm "
            "it's progressing).",
            cache_name,
            cache_path,
            cache_name,
        )

    dataset = dataset.batch(batch_size)

    if augment:
        dataset = augment_dataset(dataset, seed=seed)

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_train_val_test_datasets(
    split,
    image_size: int,
    batch_size: int,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Convenience wrapper taking a `training.split.DatasetSplit`."""
    num_classes = len(split.class_names)

    train_ds = build_dataset(
        split.train_paths, split.train_labels, image_size, num_classes,
        batch_size, shuffle=True, augment=True, seed=seed, cache_name="train",
    )
    val_ds = build_dataset(
        split.val_paths, split.val_labels, image_size, num_classes,
        batch_size, shuffle=False, augment=False, seed=seed, cache_name="val",
    )
    test_ds = build_dataset(
        split.test_paths, split.test_labels, image_size, num_classes,
        batch_size, shuffle=False, augment=False, seed=seed, cache_name="test",
    )
    return train_ds, val_ds, test_ds