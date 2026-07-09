"""
Deterministic, stratified-by-folder train/validation/test splitting.

Splits are computed at the file-path level (not via Keras' built-in
`validation_split`, which cannot produce a held-out *test* set), so the
same split logic feeds training, validation, and the final held-out
evaluation in `evaluate.py`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetSplit:
    train_paths: List[str]
    train_labels: List[int]
    val_paths: List[str]
    val_labels: List[int]
    test_paths: List[str]
    test_labels: List[int]
    class_names: List[str]


def _list_images(class_dir: Path) -> List[str]:
    return sorted(
        str(p)
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VALID_EXTENSIONS
    )


def build_split(
    dataset_dir: str,
    class_names: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Scan `dataset_dir/<class_name>/*` for each discovered class and
    produce a reproducible, per-class-stratified split so rare classes
    are represented proportionally in validation and test sets too."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0 "
            f"(got {train_ratio + val_ratio + test_ratio})"
        )

    rng = random.Random(seed)
    train_paths: List[str] = []
    train_labels: List[int] = []
    val_paths: List[str] = []
    val_labels: List[int] = []
    test_paths: List[str] = []
    test_labels: List[int] = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = Path(dataset_dir) / class_name
        images = _list_images(class_dir)
        if not images:
            logger.warning("Class '%s' has no images — skipping.", class_name)
            continue

        rng.shuffle(images)
        n_total = len(images)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total > 2 else 0
        n_train = min(n_train, n_total - (1 if n_total > 1 else 0))

        train_slice = images[:n_train]
        val_slice = images[n_train : n_train + n_val]
        test_slice = images[n_train + n_val :]

        # Guarantee the test slice is never empty when there's enough data.
        if not test_slice and len(val_slice) > 1:
            test_slice = [val_slice.pop()]

        train_paths.extend(train_slice)
        train_labels.extend([label_idx] * len(train_slice))
        val_paths.extend(val_slice)
        val_labels.extend([label_idx] * len(val_slice))
        test_paths.extend(test_slice)
        test_labels.extend([label_idx] * len(test_slice))

        logger.info(
            "Class '%s': %d train / %d val / %d test (of %d total)",
            class_name,
            len(train_slice),
            len(val_slice),
            len(test_slice),
            n_total,
        )

    return DatasetSplit(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
    )


def compute_class_weights(
    labels: List[int], num_classes: int
) -> Dict[int, float]:
    """Inverse-frequency class weights, handed to `model.fit(class_weight=)`
    to counteract the class imbalance typical of medical imaging datasets
    (e.g. far more Healthy than AMD samples)."""
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1

    total = sum(counts)
    weights: Dict[int, float] = {}
    for idx, count in enumerate(counts):
        if count == 0:
            weights[idx] = 1.0
            continue
        weights[idx] = total / (num_classes * count)

    logger.info("Computed class weights: %s", weights)
    return weights
