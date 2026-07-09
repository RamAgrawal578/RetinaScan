# Architecture

## Overview

RetinaAI is a layered Flask application backed by a TensorFlow/Keras
transfer-learning model. The layers are:

```
routes/          (Blueprints — thin, HTTP-only)
   │
services/        (PredictionService, TrainingService, report generation)
   │
model/           (build_model, trainer, predictor, losses, metrics, callbacks, optimizer)
   │
preprocessing/   (preprocess, clahe, filters, augmentation)
   │
training/        (dataset_loader — tf.data pipelines, split — train/val/test)
   │
utils/           (logger, file_utils, image_utils, validators, config_loader)
```

Each layer only depends on layers below it. Routes never touch TensorFlow
or OpenCV directly — they call into `services/`, which orchestrates
`model/` and `preprocessing/`. This keeps the HTTP layer thin and testable,
and means `train.py`, `evaluate.py`, and `predict.py` can reuse the exact
same service/model code outside of a Flask request context.

## Model architecture

```
Input (image_size × image_size × 3)
   │
Backbone (EfficientNetB7 / EfficientNetV2L / DenseNet201 / ResNet152V2)
   │   — ImageNet-pretrained, frozen during phase 1, partially unfrozen
   │     for phase 2 fine-tuning (BatchNorm layers always kept frozen)
   │
GlobalAveragePooling2D
   │
BatchNormalization → Dropout
   │
Dense(512, relu, L2-regularized)
   │
BatchNormalization → Dropout
   │
Dense(num_classes, softmax, dtype=float32)   # float32 required under mixed precision
```

`num_classes` and the class *names* are never hardcoded — they're derived
at training time from `dataset/<ClassName>/` sub-folders
(`utils/file_utils.py:discover_classes`) and persisted to
`saved_models/class_names.json` for inference to reuse.

## Training pipeline

1. **Class discovery** — scan `dataset/` for sub-folders.
2. **Stratified split** — `training/split.py` shuffles each class
   independently (seeded, reproducible) into train/val/test so rare
   classes are represented in every split.
3. **Class weights** — inverse-frequency weights counteract imbalance.
4. **`tf.data` pipeline** — `training/dataset_loader.py` wraps the
   OpenCV/NumPy preprocessing pipeline in `tf.py_function` (since OpenCV
   isn't graph-traceable) and applies Keras augmentation layers natively
   on-device.
5. **Phase 1 (transfer learning)** — backbone frozen, only the head trains.
6. **Phase 2 (fine-tuning)** — the top N layers of the backbone unfreeze
   (BatchNorm layers stay frozen to protect ImageNet running statistics),
   and training continues at a lower learning rate.
7. **Callbacks** — `ModelCheckpoint` (best-only), `EarlyStopping`
   (restores best weights), `ReduceLROnPlateau`, `TensorBoard`,
   `CSVLogger`, `TerminateOnNaN`.
8. **Artifact export** — best/final `.keras` models, `class_names.json`,
   `model_version.json`, training history JSON, and loss/accuracy curves.

Re-running `train.py` automatically resumes from
`saved_models/<model_name>_best.keras` if it exists, rather than starting
over.

## Inference pipeline

`model/predictor.py`'s `Predictor` class lazily loads the model on first
use (not at Flask startup), so the app stays up even before a model has
been trained. A prediction runs:

1. The same `preprocessing/preprocess.py` pipeline used at training time.
2. A forward pass producing per-class probabilities.
3. Grad-CAM heatmap generation (see below).
4. Risk-level and plain-language explanation derived from the predicted
   class and confidence.

### Grad-CAM with a nested backbone

The backbone is a full Keras `Model` nested as a single layer inside the
outer classification model. Keras cannot directly wire an *intermediate*
tensor from inside that nested sub-model into a new `Model` built against
the outer model's input — the sub-model is an opaque boundary. The
predictor works around this by building a small model from the backbone's
*own* input to its target conv layer and its own final output (both in the
backbone's own graph), then manually replaying the outer model's head
layers on that output inside the same `GradientTape`, so gradients flow
correctly from the prediction back to the convolutional feature maps.

## Frontend

Server-rendered Jinja templates (no SPA framework) styled with a small
custom design system (`static/css/style.css`) layered on top of Bootstrap
5 for grid/utility classes. The visual motif throughout — logo, hero,
loading spinner, error pages — is a stylized ophthalmoscope/camera
aperture ring, chosen to tie the UI back to the actual subject matter
(retinal imaging instruments) rather than a generic dashboard template.
