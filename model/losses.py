"""
Loss functions for the retina disease classifier.

Medical imaging datasets are almost always class-imbalanced (far more
"Healthy" samples than rare disease classes), so categorical focal loss
is offered alongside plain categorical crossentropy — focal loss
down-weights easy, well-classified examples so the model keeps learning
from the harder, minority-class samples throughout training.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class CategoricalFocalLoss(Loss):
    """Focal loss for multi-class classification (Lin et al., 2017).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        name: str = "categorical_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def get_loss(
    name: str = "categorical_crossentropy",
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
):
    """Factory returning a configured loss instance by name, used from
    `config/settings.py`-driven training scripts so the loss can be swapped
    without touching trainer.py."""
    if name == "focal":
        return CategoricalFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
