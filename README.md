---
title: Train Traffic Capture
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: '3.12'
app_file: app.py
pinned: false
---

# Train Traffic Capture

**Turn a screen recording of a train simulator into a spreadsheet of passenger demand.**

## TL;DR

- **What it does.** You give it a video of the simulator's station panel. It
  gives back a CSV with one row per passenger event: when it happened, how many
  passengers, and which station they are travelling to.
- **How it works.** A fine-tuned YOLOv8s model finds the text on each frame,
  EasyOCR reads it, and a set of plain rules cleans the result and removes repeats.
- **How to use it.** Either the web app, which adds charts
  ([live demo](https://huggingface.co/spaces/zhakim/train-traffic-capture)),
  or a REST API in a Docker container.
- **Try it now.**

  ```bash
  pip install -r requirements.txt
  python app.py        # open http://127.0.0.1:7860
  ```

---

## Contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [What you get](#what-you-get)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Model performance](#model-performance)
- [Documentation](#documentation)
- [Limitations](#limitations)

## The problem

In the simulator, each station has a panel listing the passengers waiting
there. A row looks like this:

```
1 pax    L-20.1    [BD108] (4-13) Hasan Sadikin
```

It says one new passenger is waiting for line L-20.1, heading to station BD108,
Hasan Sadikin. Rows appear as passengers arrive and vanish as they board.

That panel is the only place this information is shown, and the game cannot
export it. Finding out which destinations are busiest, or when demand peaks,
means watching the screen and typing numbers into a spreadsheet by hand.

This project does that job automatically from a screen recording.

## How it works

```
 video ──▶ sample 1 frame/sec ──▶ FIND text regions ──▶ READ each region ──▶ CLEAN and group ──▶ CSV
                                     (YOLOv8s)             (EasyOCR)          (Python rules)
```

The work is split into three stages, each doing one thing:

| Stage | Tool | Its job |
|---|---|---|
| **Find** | Fine-tuned YOLOv8s | Locate each piece of text and say what kind of field it is: passenger count, station code, or clock |
| **Read** | EasyOCR | Turn each cropped region into characters |
| **Clean** | Python rules | Check values, correct common OCR mistakes, group fields into rows, and drop repeats of the same event |

Two ideas keep the output reliable:

- **Known answers are looked up, not read.** The list of stations is fixed, so
  a station code misread as `BD1O6` is corrected to the nearest real code,
  `BD106`, and the station *name* comes from a lookup table rather than OCR.
- **A blank beats a wrong value.** When something cannot be read with
  confidence, the row is marked (`OCR MISSREAD`) instead of guessed.

The reasoning behind each choice is in [Problem and design](docs/design.md),
and the stage-by-stage detail is in [Pipeline](docs/pipeline.md).

## What you get

```csv
timestamp,time,station_id,station_name,passenger_increse
0.0,10:40:57,BD106,Bandung,2
1.006,10:41:06,BD106,Bandung,5
2.012,10:41:16,BD108,Hasan Sadikin,3
```

| Column | Meaning |
|---|---|
| `timestamp` | Seconds from the start of the video |
| `time` | The in-game clock at that moment |
| `station_id` | Destination station code |
| `station_name` | Destination station name |
| `passenger_increse` | Number of passengers in the event |

> **About the spelling.** `passenger_increse` is misspelled on purpose. It is the
> class name stored inside the trained model, so every layer above it uses the
> same spelling. Please do not correct it.

The web app also draws five charts from this table: a passenger timeline, a
station-by-time heatmap, destination share per time interval, total volume per
station, and overall station share.

## Getting started

There are two ways to run the project. Both use the same extraction code.

### Option 1 — Web app

For exploring a recording interactively. Upload a video or paste a YouTube
link, then browse the table and charts and download the CSV.

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>. The first start takes 10–20 seconds while the
models load.

To host it yourself, see [Deploying to Hugging Face Spaces](docs/deploy-space.md).

### Option 2 — REST API in Docker

For calling the extraction from another program. Send a video, receive the CSV.

```bash
# 1. Build the image (run from the project root)
docker build -f api/Dockerfile -t train-traffic-api .

# 2. Start the service
docker run --rm -p 8000:8000 train-traffic-api

# 3. Send a video
curl -F "file=@recording.mp4" http://localhost:8000/extract -o passengers.csv
```

| Endpoint | Purpose |
|---|---|
| `POST /extract` | Upload a video (`.mp4`, `.mov`, `.webm`, `.avi`), get the CSV back |
| `GET /health` | Check that the service is up and the models are loaded |
| `GET /docs` | Interactive API documentation |

The full reference, configuration options and operational notes are in
[Backend API](docs/api.md). To run the API without Docker, see
[Local development](docs/local-development.md).

## Project structure

```
app.py                    Web app entry point (the file Hugging Face Spaces runs)
requirements.txt          Web app dependencies
packages.txt              System packages for the Space (ffmpeg)

application/              Shared Python package
  config.py               Every threshold, the station table, the model path
  app.py                  Gradio user interface
  pipeline/               Extraction: video → detection → OCR → cleaning → CSV
  analytics/              Aggregations and charts for the web app

api/                      REST service
  main.py                 FastAPI application
  requirements.txt        API dependencies
  Dockerfile              Container image

model_registry/           Trained model weights (stored with Git LFS)
notebooks/                Model training notebook
docs/                     Documentation
```

The web app and the API are thin layers over `application/pipeline/`, so a fix
or improvement to the extraction reaches both.

## Model performance

The detector is YOLOv8s, fine-tuned from COCO weights on 110 training images
derived from 32 hand-labelled screenshots. Results on the validation set:

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| All classes | 0.936 | 0.883 | **0.948** |
| `time` | 0.921 | 0.950 | 0.988 |
| `station_id` | 0.929 | 0.859 | 0.958 |
| `passenger_increse` | 0.944 | 0.857 | 0.958 |
| `station_name` | 0.945 | 0.869 | 0.918 |
| `station_numbering` | 0.941 | 0.883 | 0.916 |

These figures are optimistic: the validation images overlap with the training
images. [Data and training](docs/data-and-training.md) explains why, along with
how the dataset was built and how each training setting was chosen. The full
training run is in
[`notebooks/passenger_detector_training.ipynb`](notebooks/passenger_detector_training.ipynb).

## Documentation

| Document | What it answers |
|---|---|
| [Problem and design](docs/design.md) | What problem is being solved, and why this design? |
| [Data and training](docs/data-and-training.md) | How was the data collected and prepared? How and why was the model configured? |
| [Pipeline](docs/pipeline.md) | What happens to a video at each stage, and why? |
| [Deploying to Hugging Face Spaces](docs/deploy-space.md) | How do I publish the web app? |
| [Backend API](docs/api.md) | How do I build, run and call the API container? |
| [Local development](docs/local-development.md) | How do I set up and run everything on my machine? |

## Limitations

- **One network.** The station table covers codes `BD093`–`BD120`. Other
  networks need their stations added to `application/config.py`; a different
  game UI needs new training data.
- **One event per sampled frame.** If several rows are on screen at once, only
  the most complete one is kept.
- **Speed.** About one second of processing per second of video on a laptop
  CPU at the default sampling rate.
- **No authentication on the API.** Run it on a private network or behind a gateway.

Details and workarounds are in [Pipeline](docs/pipeline.md#known-limitations)
and [Backend API](docs/api.md#operational-notes).
