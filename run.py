"""
Production/local run entry point.

`gunicorn` (see Procfile / gunicorn.conf.py) imports `run:app` directly;
this script also works standalone for a quick local check:

    python run.py
"""
from __future__ import annotations

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = app.config.get("DEBUG", False)
    app.run(host="0.0.0.0", port=port, debug=debug)
