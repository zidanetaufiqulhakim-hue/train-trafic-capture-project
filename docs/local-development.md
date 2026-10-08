# Local development

## Requirements

- Python 3.10 or newer (developed on 3.10; the Space runs 3.12)
- About 3 GB of disk for PyTorch, EasyOCR and their weights
- `ffmpeg` — only for the web app's YouTube input
- Git LFS, to fetch the model weights when cloning: `git lfs install`

## Setup

```bash
git clone https://github.com/zidanetaufiqulhakim-hue/train-trafic-capture-project
cd train-trafic-capture-project

conda create -n yolo_env python=3.10 -y    # or: python -m venv .venv
conda activate yolo_env                    #     source .venv/bin/activate

pip install -r requirements.txt            # web app
pip install -r api/requirements.txt        # API (adds FastAPI and uvicorn)
```

Check that the weights are the real file and not an LFS pointer:

```bash
ls -lh model_registry/passenger_capture_yolov8s.pt    # ≈ 22 MB
```

If it is a few hundred bytes, run `git lfs pull`.

## Run the web app

```bash
python app.py
```

Open <http://127.0.0.1:7860>. The status line shows "Loading models…" for 10–20
seconds on the first page load (EasyOCR downloads its weights to `~/.EasyOCR`
the first time it ever runs), then "Models ready".

To run it in the background with logs in a file:

```bash
python app.py > /tmp/app.log 2>&1 &
```

## Run the API

```bash
uvicorn api.main:app --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@recording.mp4" "http://127.0.0.1:8000/extract?target_fps=1" -o passengers.csv
```

Interactive documentation is at <http://127.0.0.1:8000/docs>. Add `--reload`
while editing code. Run commands from the project root so that `application`
and `api` are importable. See [Backend API](api.md) for the full reference and
the container build.

## Use the pipeline from Python

```python
import application.config as config
from application.pipeline.runner import load_models, run_pipeline
from application.pipeline.export import to_csv

models = load_models(config.YOLO_MODEL_PATH)
df = run_pipeline("recording.mp4", models, target_fps=1.0)
print(to_csv(df))
```

## Changing behaviour

Everything tunable is in `application/config.py`.

| To… | Change |
|---|---|
| Add a station | Add `"BDxxx": "Name"` to `STATION_LOOKUP` |
| Accept noisier station codes (or be stricter) | `MAX_EDIT_DISTANCE` (default 2) |
| Detect more / fewer boxes | `YOLO_CONF_THRESHOLD` (default 0.3) |
| Fix rows being merged or split | `ROW_Y_TOLERANCE` in pixels (default 20) |
| Collapse repeats over a longer window | `TEMPORAL_DEDUP_WINDOW_SEC` (default 5.0) |
| Change the default sampling rate | `DEFAULT_TARGET_FPS` (default 1.0) |
| Use different weights | Replace the file in `model_registry/`, or set the `YOLO_MODEL_PATH` environment variable |

## Things that look like bugs and are not

- **`passenger_increse`** is the class name inside the trained weights. Do not
  correct the spelling anywhere.
- **`clean_time` assigns seconds, hours, minutes** in that order. It matches
  the order EasyOCR returns the clock's digit groups.
- **`station_name` and `station_numbering` are detected but never read.** Names
  come from `STATION_LOOKUP`.
- **`theme=` and `css=` are passed to `app.launch()`**, not `gr.Blocks()`.
  Gradio 6 requires this.
- **Every `yield` in `_handle_analyse` returns 10 values**, one per output
  component. Adding an output means updating every yield.

## Where things are

| Path | Contents |
|---|---|
| `application/pipeline/` | Extraction logic — see [Pipeline](pipeline.md) |
| `application/analytics/` | Aggregations and Plotly charts for the web app |
| `application/app.py` | Gradio UI and event handlers |
| `api/main.py` | FastAPI service |
| `notebooks/passenger_detector_training.ipynb` | Dataset preparation, training and evaluation — see [Data and training](data-and-training.md) |
| `notebooks/preprocessing_pipeline.py` | Early single-file prototype of the pipeline; kept for reference, not imported |
