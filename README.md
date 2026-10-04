---
title: Train Traffic Capture
emoji: 🚆
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 6.29.1
python_version: "3.10"
pinned: false
license: mit
---

# Train Traffic Capture

Extract passenger movement data from Japanese train simulator recordings:
YOLO detects on-screen UI elements, EasyOCR reads the values, and the pipeline
produces a cleaned CSV plus five interactive charts — passenger flow timeline,
crowd heatmap, destination share by time interval, total volume by station,
and overall station split.

Upload a local MP4/MOV/WEBM/AVI file, or paste a YouTube URL.

## Run locally

```bash
pip install -r requirements.txt
python application/app.py
```

Opens on http://127.0.0.1:7860.

## Project layout

```
application/
  app.py                 Gradio UI entry
  config.py              Thresholds, station lookup, model path
  pipeline/              video → YOLO → OCR → cleaning → rows → postprocessing
  analytics/             aggregation + chart builders
notebooks/
  runs/.../best.pt       YOLOv8 weights (21 MB, loaded at startup)
```
