"""
Train Traffic Capture — extraction API.

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000      (from the project root)

Endpoints:
  GET  /health    liveness/readiness probe
  POST /extract   multipart video upload → cleaned CSV

The API reuses application.pipeline unchanged; it has no Gradio or Plotly
dependency. Interactive docs are served at /docs.

Environment variables:
  YOLO_MODEL_PATH   weights file (default: model_registry/passenger_capture_yolov8s.pt)
  MAX_UPLOAD_MB     reject uploads larger than this (default: 500)
  USE_GPU           "1" to run EasyOCR on CUDA (default: "0")
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

import application.config as config
from application.pipeline.export import to_csv
from application.pipeline.runner import Models, load_models, run_pipeline

logger = logging.getLogger("train_traffic_api")

ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".avi"})
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

_CHUNK_BYTES = 1024 * 1024

# ── Global model state ────────────────────────────────────────────────────────

_models: Models | None = None

# YOLO and EasyOCR instances are shared across requests and are not safe to
# call concurrently, so videos are processed one at a time. Extra requests
# wait here; scale out with more containers, not more threads.
_pipeline_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Loads the models once at startup. A missing or broken model file stops
    the server from starting — better than accepting uploads it cannot process."""
    global _models
    model_path = config.YOLO_MODEL_PATH
    if not Path(model_path).exists():
        raise RuntimeError(f"Model file not found at {model_path}. Set YOLO_MODEL_PATH.")
    logger.info("Loading models from %s", model_path)
    _models = load_models(model_path, use_gpu=USE_GPU)
    logger.info("Models ready. Detecting: %s", ", ".join(_models.class_names))
    yield
    _models = None


app = FastAPI(
    title="Train Traffic Capture API",
    description="Extracts passenger movement data from train simulator recordings as CSV.",
    version="1.0.0",
    lifespan=_lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", summary="Liveness and readiness probe")
def health() -> dict:
    if _models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return {"status": "ok", "classes": _models.class_names}


@app.post(
    "/extract",
    summary="Extract passenger events from a video",
    response_class=Response,
    responses={
        200: {"content": {"text/csv": {}}, "description": "Cleaned CSV, one row per passenger event."},
        400: {"description": "Empty upload or unsupported file extension."},
        413: {"description": "Upload exceeds MAX_UPLOAD_MB."},
        422: {"description": "The file could not be decoded as a video."},
        503: {"description": "Models are not loaded."},
    },
)
def extract(
    file: UploadFile = File(..., description="Screen recording (.mp4, .mov, .webm, .avi)."),
    target_fps: float = Query(
        config.DEFAULT_TARGET_FPS,
        ge=0.1,
        le=5.0,
        description="Frames sampled per second of video. Lower is faster.",
    ),
) -> Response:
    """Declared with plain `def` so FastAPI runs it in a worker thread — the
    pipeline is CPU-bound and would otherwise block the event loop (and /health)."""
    if _models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # OpenCV reads from a path, so the upload is spooled to a temp file first.
    tmp_path = _save_upload(file, suffix)
    try:
        started = time.perf_counter()
        with _pipeline_lock:
            try:
                df = run_pipeline(tmp_path, _models, target_fps)
            except ValueError as e:
                # video_io.open_video raises ValueError for undecodable files
                raise HTTPException(status_code=422, detail="The file could not be decoded as a video.") from e
        elapsed = time.perf_counter() - started
    finally:
        os.unlink(tmp_path)

    logger.info("Extracted %d rows from %s in %.1fs", len(df), file.filename, elapsed)

    stem = Path(file.filename or "video").stem
    return Response(
        content=to_csv(df),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{_ascii_safe(stem)}_passengers.csv"',
            "X-Row-Count": str(len(df)),
            "X-Processing-Seconds": f"{elapsed:.2f}",
        },
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _save_upload(file: UploadFile, suffix: str) -> str:
    """Copies the upload to a temp file, enforcing MAX_UPLOAD_BYTES.

    Returns the temp file path; the caller must delete it."""
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="train_traffic_api_") as tmp:
        tmp_path = tmp.name
        while chunk := file.file.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                break
            tmp.write(chunk)

    if written > MAX_UPLOAD_BYTES:
        os.unlink(tmp_path)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    if written == 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return tmp_path


def _ascii_safe(name: str) -> str:
    """Keeps a filename header-safe: ASCII letters, digits, dash and underscore only."""
    cleaned = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in name)
    return cleaned or "video"
