# Train Traffic Capture — extraction API image.
#
# Build from the project root (the build context must contain application/,
# api/ and model_registry/):
#   docker build -f api/Dockerfile -t train-traffic-api .

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics) links against libGL and glib at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch first: the default PyPI wheel for linux/amd64 bundles CUDA
# libraries and adds several GB to the image for no benefit on CPU hosts.
# PyPI stays as the primary index because the CPU index does not carry every
# transitive dependency; pip prefers the "+cpu" build when both match the pin.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0

COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

# Bake the EasyOCR weights into the image. Without this the first container
# start downloads them, which is slow and fails on hosts with no egress.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY application/ application/
COPY api/ api/
COPY model_registry/ model_registry/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# start-period covers model loading at boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker on purpose: each worker holds its own copy of both models.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
