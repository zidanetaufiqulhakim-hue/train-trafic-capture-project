# Backend API

A FastAPI service that accepts a video upload and returns the extracted CSV.
It wraps the same pipeline as the web app and has no dependency on Gradio or Plotly.

- Code: `api/main.py`
- Image definition: `api/Dockerfile`
- Interactive documentation (Swagger UI): `/docs` on a running instance

## Build and run the container

Build from the **project root** — the build context must include
`application/`, `api/` and `model_registry/`:

```bash
docker build -f api/Dockerfile -t train-traffic-api .
```

```bash
docker run --rm -p 8000:8000 train-traffic-api
```

```bash
curl http://localhost:8000/health
curl -F "file=@recording.mp4" http://localhost:8000/extract -o passengers.csv
```

The first build downloads PyTorch and takes several minutes; later builds reuse
the cached layers unless `api/requirements.txt` changes.

### What the image contains

| Layer | Reason |
|---|---|
| `python:3.10-slim` | Small base; matches the development Python version |
| `libgl1`, `libglib2.0-0` | Shared libraries OpenCV loads at import |
| CPU-only `torch` / `torchvision` | The default Linux x86-64 wheels bundle CUDA and add several GB that a CPU host never uses |
| `api/requirements.txt` | FastAPI, uvicorn, Ultralytics, EasyOCR, pandas |
| EasyOCR weights (`/opt/easyocr`) | Downloaded at build time so the container starts without internet access |
| `application/`, `api/`, `model_registry/` | The code and the YOLO weights |

`.dockerignore` is an allow-list: only those three directories are sent to the
Docker daemon, so datasets, notebooks and videos can never end up in the image.

The process runs as a non-root user (`appuser`, UID 1000) and the image defines
a `HEALTHCHECK` against `/health`.

### Configuration

| Environment variable | Default | Effect |
|---|---|---|
| `MAX_UPLOAD_MB` | `500` | Uploads larger than this are rejected with 413 |
| `YOLO_MODEL_PATH` | `model_registry/passenger_capture_yolov8s.pt` | Weights file to load |
| `USE_GPU` | `0` | `1` runs EasyOCR on CUDA. Requires a CUDA base image and GPU-enabled PyTorch, which this Dockerfile does not install |

```bash
docker run --rm -p 8000:8000 -e MAX_UPLOAD_MB=200 train-traffic-api
```

To serve retrained weights without rebuilding, mount them and point the variable at the mount:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/new_weights:/models:ro" \
  -e YOLO_MODEL_PATH=/models/best.pt \
  train-traffic-api
```

### Resources

Allow at least 2 GB of memory. Processing is CPU-bound: roughly 1.1 s per
sampled frame was measured on an Apple-silicon laptop CPU for a 2880×1800
recording, so a 5-minute video at the default 1 fps takes around 5–6 minutes.
Set client and proxy timeouts accordingly.

## Endpoints

### `GET /health`

Returns 200 once the models are loaded. Use it for container health checks and
load-balancer probes.

```json
{"status": "ok", "classes": ["passenger_increse", "station_id", "station_name", "station_numbering", "time"]}
```

### `POST /extract`

Processes one video and returns the CSV in the response body. The request
blocks until processing finishes.

**Request** — `multipart/form-data`

| Part | In | Type | Required | Description |
|---|---|---|---|---|
| `file` | form | file | yes | Screen recording. Extension must be `.mp4`, `.mov`, `.webm` or `.avi` |
| `target_fps` | query | float, 0.1–5.0 | no (default `1.0`) | Frames sampled per second of video. Lower is faster; higher catches short-lived rows |

**Response — 200** — `text/csv; charset=utf-8`

| Header | Example | Meaning |
|---|---|---|
| `Content-Disposition` | `attachment; filename="recording_passengers.csv"` | Suggested file name, derived from the upload |
| `X-Row-Count` | `13` | Number of data rows |
| `X-Processing-Seconds` | `22.93` | Time spent in the pipeline |

```csv
timestamp,time,station_id,station_name,passenger_increse
0.0,10:40:57,BD106,Bandung,2
1.006,10:41:06,BD106,Bandung,5
10.058,10:42:36,OCR MISSREAD,UNKNOWN ID,2
14.081,10:43:16,BD093,Dayeuh Kolot,
```

| Column | Type | Notes |
|---|---|---|
| `timestamp` | float | Seconds from the start of the video |
| `time` | string | In-game clock, `HH:MM:SS` or `HH:MM`; empty if unreadable |
| `station_id` | string | Destination station code, or `OCR MISSREAD` |
| `station_name` | string | Destination station name, or `UNKNOWN ID` |
| `passenger_increse` | integer | Passenger count; empty if unreadable. The spelling is intentional |

A video in which nothing is detected returns 200 with only the header row and
`X-Row-Count: 0`. See [Pipeline](pipeline.md#output) for how to interpret
partial rows.

**Errors** — JSON body `{"detail": "..."}`

| Status | When |
|---|---|
| 400 | The upload is empty, or its extension is not one of the four allowed |
| 413 | The upload exceeds `MAX_UPLOAD_MB` |
| 422 | The file could not be decoded as a video; `file` is missing; or `target_fps` is out of range |
| 503 | Models are not loaded |

### Examples

```bash
curl -f -F "file=@recording.mp4" \
     "http://localhost:8000/extract?target_fps=0.5" \
     -o passengers.csv
```

```python
import requests

with open("recording.mp4", "rb") as f:
    r = requests.post(
        "http://localhost:8000/extract",
        files={"file": f},
        params={"target_fps": 1.0},
        timeout=1800,
    )
r.raise_for_status()
print(r.headers["X-Row-Count"], "rows")
open("passengers.csv", "wb").write(r.content)
```

## Operational notes

- **No authentication.** The service trusts every caller. Run it on a private
  network or put it behind a gateway that handles authentication and TLS.
- **One video at a time per container.** The models are shared and not
  thread-safe, so requests are processed sequentially; additional requests wait.
  `/health` stays responsive during processing. To raise throughput, run more
  containers behind a load balancer rather than more uvicorn workers — each
  worker loads its own copy of both models.
- **Synchronous by design.** There is no job queue. For long videos, set proxy
  and client timeouts to match, or split the recording.
- **Upload limit.** `MAX_UPLOAD_MB` is enforced after the request body has been
  received. Set a matching body-size limit on the reverse proxy to reject
  oversized uploads before they are transferred.
- **Nothing is stored.** The upload is written to a temporary file and deleted
  when the request ends, including on failure.
- **Fail-fast startup.** If the weights file is missing the server exits
  instead of starting in a broken state.
- **Logs** go to stdout: one line per extraction with file name, row count and duration.
