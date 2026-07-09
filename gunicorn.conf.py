"""
Gunicorn configuration for production deployment.

A large CNN (EfficientNetB7) held in memory per worker means worker
count must stay low relative to a typical web app — favor a small
number of workers with more threads, and a generous timeout for
first-request model loading and slower CPU-bound inference.
"""
import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Keep worker count low: each worker loads its own full copy of the
# model into memory. 2 is a safe default for most Render instance
# sizes; raise it only if you have confirmed available RAM headroom.
workers = int(os.environ.get("GUNICORN_WORKERS", 2))
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", 4))

# Model loading + CPU-bound CNN inference can take longer than the
# gunicorn default of 30s, especially on a cold instance.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))
graceful_timeout = 30
keepalive = 5

max_requests = 500
max_requests_jitter = 50

loglevel = os.environ.get("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"

preload_app = False  # load model per-worker rather than fork-sharing TF state
