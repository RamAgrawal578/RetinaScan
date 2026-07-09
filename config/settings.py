"""
Central configuration for the Retina Disease Detection System.

All values are overridable via environment variables so the same codebase
runs unmodified in local development, CI, and on Render. See `.env.example`
for the full list of variables.

NOTE: Defaults below are tuned for CPU-only training on a laptop with
~8GB RAM (no dedicated GPU). If you later train on a machine with a GPU
and more RAM, override MODEL_ARCHITECTURE / IMAGE_SIZE / BATCH_SIZE via
environment variables instead of editing this file again.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Type

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # python-dotenv is an optional convenience for local development;
    # production environments set real environment variables directly.
    pass


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean-like environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class BaseConfig:
    """Shared configuration across all environments."""

    # --- Core Flask ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    JSON_SORT_KEYS: bool = False
    DEBUG: bool = False
    TESTING: bool = False

    # --- Filesystem layout ---
    DATASET_DIR: str = os.environ.get(
        "DATASET_DIR", str(BASE_DIR / "dataset")
    )
    SAVED_MODELS_DIR: str = os.environ.get(
        "SAVED_MODELS_DIR", str(BASE_DIR / "saved_models")
    )
    UPLOAD_DIR: str = os.environ.get(
        "UPLOAD_DIR", str(BASE_DIR / "uploads")
    )
    REPORTS_DIR: str = os.environ.get(
        "REPORTS_DIR", str(BASE_DIR / "reports")
    )
    LOGS_DIR: str = os.environ.get("LOGS_DIR", str(BASE_DIR / "logs"))
    TENSORBOARD_DIR: str = os.environ.get(
        "TENSORBOARD_DIR", str(BASE_DIR / "logs" / "tensorboard")
    )

    # --- Uploads / security ---
    MAX_CONTENT_LENGTH: int = _int_env(
        "MAX_UPLOAD_SIZE_MB", 10
    ) * 1024 * 1024
    ALLOWED_EXTENSIONS: frozenset = frozenset(
        {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
    )

    # --- Rate limiting (ready, wired up in app factory) ---
    RATE_LIMIT_ENABLED: bool = _bool_env("RATE_LIMIT_ENABLED", True)
    RATE_LIMIT_DEFAULT: str = os.environ.get(
        "RATE_LIMIT_DEFAULT", "60 per minute"
    )

    # --- Model / ML ---
    # CHANGED: EfficientNetB7 (66M params, 600px native) is far too heavy
    # for CPU-only training on 8GB RAM (5.83GB usable). EfficientNetB0 is
    # ~5.3M params, 224px native, and trains reliably on this hardware.
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "retina_disease_model")
    MODEL_ARCHITECTURE: str = os.environ.get(
        "MODEL_ARCHITECTURE", "EfficientNetB0"
    )
    # CHANGED: fallback order now goes from lightest to heaviest, and drops
    # anything B7/V2L/201/152-class since none of those are CPU-on-8GB safe.
    ARCHITECTURE_FALLBACK_ORDER = (
        "EfficientNetB0",
        "MobileNetV3Large",
        "DenseNet121",
        "ResNet50V2",
    )
    # CHANGED: matches EfficientNetB0's native input size. 600px images at
    # batch 8 is what triggered your mkl_conv_ops ABORTED crash.
    IMAGE_SIZE: int = _int_env("IMAGE_SIZE", 224)
    IMAGE_CHANNELS: int = 3
    # CHANGED: batch size 8 on 5.83GB usable RAM leaves almost no headroom
    # once EfficientNet activations + Windows + Python overhead are
    # accounted for. Batch size 4 is a safe starting point; drop to 2 if
    # you still see crashes or your system becomes unresponsive.
    BATCH_SIZE: int = _int_env("BATCH_SIZE", 4)
    # CHANGED: fewer epochs by default since CPU training is slow; combined
    # with early stopping this still converges, just costs less wall-clock
    # time per run. Raise this back up once you've confirmed things run
    # smoothly end-to-end.
    EPOCHS: int = _int_env("EPOCHS", 25)
    FINE_TUNE_EPOCHS: int = _int_env("FINE_TUNE_EPOCHS", 10)
    FINE_TUNE_AT_LAYER: int = _int_env("FINE_TUNE_AT_LAYER", -30)
    LEARNING_RATE: float = _float_env("LEARNING_RATE", 1e-4)
    FINE_TUNE_LEARNING_RATE: float = _float_env(
        "FINE_TUNE_LEARNING_RATE", 1e-5
    )
    DROPOUT_RATE: float = _float_env("DROPOUT_RATE", 0.4)
    L2_REGULARIZATION: float = _float_env("L2_REGULARIZATION", 1e-4)
    # CHANGED: mixed precision has zero benefit on CPU (it's a GPU tensor
    # core feature) and just adds cast overhead. Your log already showed
    # "Mixed precision requested but no GPU present" — this makes that
    # explicit instead of relying on a runtime fallback.
    USE_MIXED_PRECISION: bool = _bool_env("USE_MIXED_PRECISION", False)
    RANDOM_SEED: int = _int_env("RANDOM_SEED", 42)
    # "imagenet" (default, recommended) or "none" for random initialization
    # — useful for offline/air-gapped training environments without
    # network access to download pretrained weights.
    PRETRAINED_WEIGHTS: str | None = (
        None
        if os.environ.get("PRETRAINED_WEIGHTS", "imagenet").lower() == "none"
        else "imagenet"
    )

    TRAIN_SPLIT: float = _float_env("TRAIN_SPLIT", 0.70)
    VAL_SPLIT: float = _float_env("VAL_SPLIT", 0.15)
    TEST_SPLIT: float = _float_env("TEST_SPLIT", 0.15)

    EARLY_STOPPING_PATIENCE: int = _int_env("EARLY_STOPPING_PATIENCE", 6)
    REDUCE_LR_PATIENCE: int = _int_env("REDUCE_LR_PATIENCE", 3)
    REDUCE_LR_FACTOR: float = _float_env("REDUCE_LR_FACTOR", 0.5)
    MIN_LEARNING_RATE: float = _float_env("MIN_LEARNING_RATE", 1e-7)

    # Populated at runtime by utils.file_utils.discover_classes(); kept here
    # as a documented default only (never hardcoded into training/inference
    # logic).
    DEFAULT_CLASS_NAMES = (
        "Healthy",
        "Diabetic_Retinopathy",
        "Glaucoma",
        "AMD",
        "Cataract",
    )

    GRADCAM_LAST_CONV_LAYER: str = os.environ.get(
        "GRADCAM_LAST_CONV_LAYER", "top_conv"
    )

    APP_VERSION: str = os.environ.get("APP_VERSION", "1.0.0")

    @staticmethod
    def init_app(app) -> None:
        """Hook for subclasses / app factory to run environment-specific
        setup after the Flask app object is created."""
        return None


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    RATE_LIMIT_DEFAULT = "1000 per minute"


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    RATE_LIMIT_ENABLED = False


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False

    @staticmethod
    def init_app(app) -> None:
        if app.config["SECRET_KEY"] == "dev-secret-key-change-me":
            raise RuntimeError(
                "SECRET_KEY must be set via environment variable in "
                "production. Refusing to start with the default dev key."
            )


_CONFIG_MAP: dict[str, Type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config(env_name: str | None = None) -> Type[BaseConfig]:
    """Resolve a config class from an environment name (falls back to the
    FLASK_ENV / APP_ENV environment variables, then 'default')."""
    name = env_name or os.environ.get("APP_ENV") or os.environ.get(
        "FLASK_ENV", "default"
    )
    return _CONFIG_MAP.get(name.lower(), DevelopmentConfig)