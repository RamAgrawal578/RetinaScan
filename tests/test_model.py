"""
Tests for model construction, losses, callbacks, and the optimizer
factory. These require TensorFlow; if it isn't installed yet (e.g. a
lightweight CI stage that only lints the web layer), the whole module is
skipped rather than failing the run.
"""
from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from model.build_model import (  # noqa: E402
    build_model,
    native_input_size,
    unfreeze_for_fine_tuning,
)
from model.losses import CategoricalFocalLoss, get_loss  # noqa: E402
from model.optimizer import build_optimizer, set_global_seed  # noqa: E402
from training.split import build_split, compute_class_weights  # noqa: E402


class TestBuildModel:
    def test_build_model_default_architecture_output_shape(self):
        model, used_architecture = build_model(
            num_classes=5,
            architecture="DenseNet201",
            image_size=224,
            weights=None,
        )
        assert used_architecture in (
            "DenseNet201",
            "EfficientNetB7",
            "EfficientNetV2L",
            "ResNet152V2",
        )
        assert model.output_shape == (None, 5)
        assert model.input_shape == (None, 224, 224, 3)

    def test_native_input_size_known_architectures(self):
        assert native_input_size("EfficientNetB7") == 600
        assert native_input_size("DenseNet201") == 224

    def test_unfreeze_for_fine_tuning_makes_backbone_trainable(self):
        model, _ = build_model(
            num_classes=3,
            architecture="DenseNet201",
            image_size=224,
            weights=None,
        )
        fine_tuned = unfreeze_for_fine_tuning(model, fine_tune_at_layer=-10)
        backbone = next(
            layer for layer in fine_tuned.layers if isinstance(layer, tf.keras.Model)
        )
        assert backbone.trainable is True


class TestLosses:
    def test_categorical_focal_loss_per_sample_is_nonnegative(self):
        """`Loss.call()` returns the raw per-sample loss (before Keras'
        automatic batch reduction), so we assert on that directly."""
        loss_fn = CategoricalFocalLoss()
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = tf.constant([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
        per_sample_loss = loss_fn.call(y_true, y_pred)
        assert per_sample_loss.shape == (2,)
        assert bool(tf.reduce_all(per_sample_loss >= 0))

    def test_categorical_focal_loss_call_reduces_to_scalar(self):
        """Invoking the loss normally (as Keras does during training)
        applies the standard batch reduction down to a scalar."""
        loss_fn = CategoricalFocalLoss()
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = tf.constant([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
        reduced_loss = loss_fn(y_true, y_pred)
        assert reduced_loss.shape == ()
        assert float(reduced_loss) >= 0

    def test_get_loss_factory_returns_focal_or_crossentropy(self):
        focal = get_loss("focal")
        assert isinstance(focal, CategoricalFocalLoss)
        crossentropy = get_loss("categorical_crossentropy")
        assert isinstance(crossentropy, tf.keras.losses.CategoricalCrossentropy)


class TestOptimizer:
    def test_build_optimizer_returns_adam_when_no_mixed_precision(self):
        optimizer = build_optimizer(learning_rate=1e-4, use_mixed_precision=False)
        assert isinstance(optimizer, tf.keras.optimizers.Adam)

    def test_set_global_seed_is_deterministic(self):
        set_global_seed(123)
        first = tf.random.uniform((3,)).numpy()
        set_global_seed(123)
        second = tf.random.uniform((3,)).numpy()
        assert (first == second).all()


class TestClassWeights:
    def test_compute_class_weights_favors_minority_class(self):
        labels = [0] * 90 + [1] * 10
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_build_split_respects_ratios_reasonably(self, tmp_path):
        for class_name, count in [("Healthy", 20), ("Glaucoma", 20)]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(count):
                (class_dir / f"img_{i}.png").write_bytes(b"\x89PNG\r\n")

        split = build_split(
            dataset_dir=str(tmp_path),
            class_names=["Healthy", "Glaucoma"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        total = len(split.train_paths) + len(split.val_paths) + len(split.test_paths)
        assert total == 40
        assert len(split.train_paths) > len(split.val_paths)
        assert len(split.train_paths) > len(split.test_paths)
