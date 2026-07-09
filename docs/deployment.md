# Deployment

## Render (primary target)

This repo ships a `render.yaml` Blueprint. From the Render dashboard:

1. **New → Blueprint**, point it at this repository.
2. Render reads `render.yaml` and provisions a web service with:
   - Build: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `gunicorn -c gunicorn.conf.py run:app`
   - Health check: `/api/health`
   - A small persistent disk mounted for model/dataset storage
3. Review/edit the environment variables Render pre-fills from
   `render.yaml` (`SECRET_KEY` is auto-generated; everything else has a
   sensible default matching `.env.example`).
4. Deploy.

### Getting a trained model onto Render

Render's default filesystem is ephemeral — it resets on every deploy. Two
options:

- **Recommended for most users:** train locally (or on a GPU
  notebook/cloud instance with real internet access to download ImageNet
  weights), then include the resulting `saved_models/*.keras` files when
  you push/deploy, or upload them directly onto the persistent disk
  declared in `render.yaml` via Render's shell.
- **Train on Render directly:** point `SAVED_MODELS_DIR` and
  `DATASET_DIR` at the mounted persistent disk path
  (`/opt/render/project/src/persisted_data/...`) via environment
  variables, upload your dataset there, and run `python train.py` as a
  one-off Render job. Expect this to be slow on Render's CPU-only plans
  for an EfficientNetB7 run — consider a smaller architecture or image
  size for a first pass.

### Manual Web Service (without Blueprint)

If you'd rather configure it by hand instead of using `render.yaml`:

| Setting | Value |
|---|---|
| Environment | Python 3 |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `gunicorn -c gunicorn.conf.py run:app` |
| Health Check Path | `/api/health` |

Set at minimum: `SECRET_KEY`, `APP_ENV=production`.

## Gunicorn tuning

`gunicorn.conf.py` defaults to 2 workers × 4 threads, a 120s timeout, and
`preload_app = False` (each worker loads its own copy of the model rather
than sharing one via fork — simpler and safer given TensorFlow's threading
model, at the cost of higher total memory use). Override via
`GUNICORN_WORKERS`, `GUNICORN_THREADS`, `GUNICORN_TIMEOUT` environment
variables if you have a larger/smaller instance.

## Running locally without Docker

```bash
pip install -r requirements.txt
cp .env.example .env
python run.py
```

## Environment parity checklist

Before deploying, confirm:

- [ ] `SECRET_KEY` is set to a real random value (not the dev default —
      `ProductionConfig.init_app` will refuse to start otherwise).
- [ ] `APP_ENV=production`.
- [ ] A trained model exists at `SAVED_MODELS_DIR` (check `/api/health`
      after deploy — `model_loaded` should be `true`).
- [ ] `MAX_UPLOAD_SIZE_MB` and `RATE_LIMIT_DEFAULT` match your expected
      traffic.
- [ ] If you enabled CSRF (`flask-wtf`) or rate limiting
      (`flask-limiter`), confirm they initialized (check the startup
      logs — both log a clear line either way).
