# RetinaAI — Retina Disease Detection System

An AI-powered retinal fundus image screening system built with TensorFlow/Keras
transfer learning and Flask. Upload a fundus photograph and get a disease
classification, confidence score, Grad-CAM attention heatmap, and a
downloadable PDF report.

> **Medical disclaimer:** This is a research and educational project, not a
> certified medical device. It must never replace evaluation by a qualified
> ophthalmologist or other licensed medical professional.

---

## Features

- **Transfer learning** on an accuracy-first CNN backbone — EfficientNetB7 by
  default, with automatic fallback to EfficientNetV2L, DenseNet201, or
  ResNet152V2 if the preferred backbone can't be instantiated.
- **Two-phase training**: frozen-backbone warm-up, then fine-tuning of the
  top layers, with automatic checkpoint resume.
- **Professional preprocessing pipeline**: corruption/blur detection, center
  crop, aspect-ratio-preserving resize, denoising, CLAHE contrast
  enhancement, sharpening — identical at train and inference time.
- **Automatic class discovery** — drop any folder-structured dataset into
  `dataset/` and the app adapts; nothing is hardcoded.
- **Grad-CAM explainability** — every prediction includes a heatmap overlay
  showing which regions of the image influenced the result.
- **Downloadable PDF reports** via ReportLab.
- **REST API** (`/api/predict`, `/api/health`, `/api/model-info`,
  `/api/version`, `/api/metrics`) alongside a full HTML dashboard.
- **Render-ready deployment**: `render.yaml`, `Procfile`, `gunicorn.conf.py`.
- **48 passing pytest tests** covering preprocessing, validation, model
  construction, and API routes (see [Testing](#testing)).

---

## Folder structure

```
retina-ai/
├── app.py                  # Flask application factory
├── run.py                  # Production/local entry point (gunicorn imports run:app)
├── train.py                # CLI: train the model
├── evaluate.py              # CLI: evaluate on the held-out test split
├── predict.py               # CLI: single-image prediction
├── export_model.py          # CLI: export SavedModel / TFLite
├── requirements.txt / runtime.txt / render.yaml / Procfile / gunicorn.conf.py
├── config/                 # BaseConfig / Development / Testing / Production
├── model/                  # build_model, trainer, predictor, losses, metrics, callbacks, optimizer
├── training/                # dataset_loader (tf.data), split (train/val/test + class weights)
├── preprocessing/            # preprocess, clahe, filters, augmentation
├── services/                 # prediction_service, training_service, report_service
├── routes/                   # main, predict, api blueprints + error handlers
├── utils/                    # logger, file_utils, image_utils, validators, config_loader
├── templates/ / static/       # Jinja templates, CSS, JS
├── tests/                    # pytest suite
├── dataset/                   # EMPTY — see dataset/README.md
├── saved_models/               # trained model artifacts (gitignored)
└── docs/                       # architecture.md, api.md, deployment.md
```

> **Note on `config.py`:** the original spec called for both a root-level
> `config.py` *and* a `config/` package. Python cannot resolve both — a
> module and a package of the same name in the same directory is ambiguous,
> and the package always wins, silently orphaning the module. This repo
> keeps a single source of truth in `config/settings.py` instead.

---

## Quick start

```bash
git clone <your-repo-url>
cd retina-ai
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 1. Add a dataset

See [`dataset/README.md`](dataset/README.md) for Kaggle dataset links and
the required folder layout. In short:

```
dataset/
├── Healthy/
├── Diabetic_Retinopathy/
├── Glaucoma/
├── AMD/
└── Cataract/
```

### 2. Train

```bash
python train.py
```

This automatically discovers your classes, splits train/val/test
(70/15/15 by default), computes class weights, runs frozen-backbone
transfer learning, then fine-tunes the top layers, and saves:

- `saved_models/retina_disease_model_best.keras` (best checkpoint)
- `saved_models/retina_disease_model_final.keras`
- `saved_models/class_names.json`, `saved_models/model_version.json`
- `logs/training_history.json`, `reports/training_curves.png`

Re-running `python train.py` automatically **resumes** from the last
checkpoint if one exists.

### 3. Evaluate

```bash
python evaluate.py
```

Generates `reports/confusion_matrix.png`, `reports/roc_curves.png`,
`reports/precision_recall_curves.png`, and `reports/evaluation_report.json`
(consumed by `GET /api/metrics`).

### 4. Run the app

```bash
python run.py
```

Visit `http://localhost:5000`. Or use the CLI directly:

```bash
python predict.py path/to/image.jpg --report
```

---

## Environment variables

See [`.env.example`](.env.example) for the full list. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ARCHITECTURE` | `EfficientNetB7` | Preferred backbone (falls back automatically) |
| `PRETRAINED_WEIGHTS` | `imagenet` | Set to `none` for offline/air-gapped training |
| `IMAGE_SIZE` | `600` | Input resolution (matches EfficientNetB7's native size) |
| `BATCH_SIZE` | `8` | Lower this on CPU-only or memory-constrained machines |
| `EPOCHS` / `FINE_TUNE_EPOCHS` | `50` / `20` | Phase 1 / phase 2 epoch budgets |
| `SECRET_KEY` | *(dev default)* | **Must** be overridden in production |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size limit |

---

## Training on Render or another remote/persistent-disk host

Render's filesystem is ephemeral between deploys. If you plan to train (or
retrain) directly on Render, mount the persistent disk declared in
`render.yaml` and point `SAVED_MODELS_DIR` / `DATASET_DIR` at it via
environment variables, e.g.:

```
SAVED_MODELS_DIR=/opt/render/project/src/persisted_data/saved_models
DATASET_DIR=/opt/render/project/src/persisted_data/dataset
```

In practice, most people train locally or on a GPU notebook environment and
then deploy only the trained `.keras` file to Render alongside the app code.

---

## Deployment (Render)

1. Push this repo to GitHub.
2. In Render, create a new **Blueprint** from `render.yaml`, or a manual Web
   Service with:
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `gunicorn -c gunicorn.conf.py run:app`
   - Health check path: `/api/health`
3. Set `SECRET_KEY` (Render can auto-generate this) and any other variables
   from `.env.example` you want to override.
4. Either upload a pre-trained model into the persistent disk, or run
   training as a one-off Render job pointed at the same disk.

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | multipart `image` file → prediction JSON. `?report=true` also generates a PDF. |
| `/api/health` | GET | Liveness + whether a trained model is loaded. |
| `/api/model-info` | GET | Architecture, class names, training timestamp. |
| `/api/version` | GET | App + model version. |
| `/api/metrics` | GET | Latest `evaluate.py` report (404 until you run it). |

See [`docs/api.md`](docs/api.md) for full request/response examples.

---

## Testing

```bash
pytest                      # run everything
pytest --cov=. --cov-report=term-missing   # with coverage
pytest tests/test_model.py  # requires TensorFlow; auto-skipped if absent
```

The suite (48 tests) covers preprocessing/CLAHE/filters, validators, model
construction (architecture wiring, losses, optimizer, class weights, dataset
splitting), and every Flask route including the REST API — all using
synthetic in-memory images, so no dataset or trained model is required to
run it.

---

## Troubleshooting / FAQ

**"No trained model found" on `/health` or `/predict`.**
You haven't run `python train.py` yet, or `SAVED_MODELS_DIR` points
somewhere without a `*_best.keras` / `*_final.keras` file.

**Training is extremely slow / seems stuck.**
EfficientNetB7 at 600×600 is a large accuracy-first model — expect it to be
slow on CPU. Lower `BATCH_SIZE`, use a smaller `IMAGE_SIZE`, or switch
`MODEL_ARCHITECTURE` to `DenseNet201` for faster iteration while developing,
then switch back to `EfficientNetB7` for your final training run.

**`URL fetch failure ... 403 Forbidden` when building the model.**
Your environment can't reach `storage.googleapis.com` to download ImageNet
weights (common in restricted/offline sandboxes and some CI runners). Set
`PRETRAINED_WEIGHTS=none` to build with random initialization instead, or
run training somewhere with normal internet access.

**Grad-CAM heatmap is missing from a result.**
Check the logs for a "Grad-CAM target layer not found" warning —
`GRADCAM_LAST_CONV_LAYER` must match a real layer name inside your chosen
backbone (defaults assume EfficientNet's `top_conv`; e.g. DenseNet201 uses
`relu`, ResNet152V2 uses `post_relu`). The prediction itself still succeeds
even if the heatmap can't be generated.

**Out of memory during training.**
Lower `BATCH_SIZE`, or reduce `IMAGE_SIZE`. Gunicorn's worker count
(`gunicorn.conf.py`) should also stay low in production since each worker
holds its own full copy of the model in memory.

---

## Future improvements

- Multi-label support for images with co-occurring conditions.
- Model quantization-aware training for faster mobile/edge inference.
- Active-learning loop to prioritize which unlabeled images to annotate next.
- Multi-language UI.

---

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built as an AI/ML portfolio and research project applying transfer learning
to retinal fundus image screening.
