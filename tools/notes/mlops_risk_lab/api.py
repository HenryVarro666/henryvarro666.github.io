"""Load the active immutable bundle once per process; restart after promotion."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from lab import Features, load_active, predict_loaded


def create_app(root=None):
    root = Path(root or os.environ.get("MLOPS_LAB_ROOT", "workspace"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.loaded = None
        try:
            app.state.loaded = load_active(root)
        except (ValueError, FileNotFoundError, KeyError):
            pass  # Liveness stays available; readiness refuses traffic.
        yield
        app.state.loaded = None

    app = FastAPI(title="Synthetic risk teaching lab", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz():
        if app.state.loaded is None:
            raise HTTPException(status_code=503, detail="No compatible active model; promote and restart")
        return {"status": "ready", "model_version": app.state.loaded[1]["run_id"]}

    @app.post("/predict")
    def predict(row: Features):
        readyz()
        return predict_loaded(*app.state.loaded, row.model_dump())

    return app


app = create_app()
