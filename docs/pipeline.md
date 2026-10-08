# Pipeline

What happens to a video between upload and CSV, and why each stage exists.
All code is in `application/pipeline/`; every constant mentioned here is defined
in `application/config.py`.

## Overview

```
video file
   │
   ▼
video_io         sample 1 frame per second
   │
   ▼  for each sampled frame ──────────────────────────────────────────┐
detection        YOLOv8s → boxes with a class and a confidence         │
   │             (station_name / station_numbering boxes dropped)      │
   ▼                                                                   │
ocr              crop → grayscale → 2× upscale → CLAHE → EasyOCR       │
   │                                                                   │
   ▼                                                                   │
cleaning         raw text → validated value, or None                   │
   │                                                                   │
   ▼                                                                   │
assembly         group boxes into rows by vertical position            │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘
   ▼  once, after the last frame
postprocessing   4 passes: per-frame dedup → drop empty → temporal dedup → fill names
   │
   ▼
export           fixed column order → CSV
```

`runner.py` ties the stages together. It offers two equivalent entry points:
`run_pipeline()` returns the final DataFrame (used by the API), and
`iter_pipeline()` is a generator that yields progress after every frame and the
DataFrame last (used by the web app to show a live progress line).

## Stages

### 1. Frame sampling — `video_io.py`

Reads every Nth frame, where N = video frame rate ÷ `DEFAULT_TARGET_FPS` (1.0).
A 60 fps recording is read once every 60 frames.

**Why:** the panel changes every few seconds, not every frame. Sampling makes
cost proportional to video duration rather than frame count, and it is the main
speed control exposed to users (0.5–3 fps in the web app, 0.1–5 in the API).
Each frame is tagged with its offset in seconds, which becomes the `timestamp` column.

### 2. Detection — `detection.py`

Runs the fine-tuned YOLOv8s model and keeps boxes with confidence ≥
`YOLO_CONF_THRESHOLD` (0.3). Boxes whose class is in `SKIPPED_CLASSES`
(`station_name`, `station_numbering`) are dropped immediately.

**Why a low threshold:** a missed box loses an event for good, whereas a
spurious box usually produces text that fails cleaning and is discarded later.
The stages after detection are the filter, so detection leans towards recall.

**Why skip two classes:** the station name is derived from the ID, and the
numbering adds nothing once the ID is known. Not reading them saves two OCR
calls per row and removes the least reliable class from the output.

Class names are read from the weights file at load time, so the pipeline cannot
drift out of sync with training.

### 3. OCR — `ocr.py`

Each remaining box is cropped from the **full-resolution** frame, then:

1. converted to grayscale,
2. upscaled 2× with bicubic interpolation,
3. contrast-enhanced with CLAHE,
4. passed to EasyOCR.

**Why:** UI text is small and
EasyOCR is noticeably more accurate on larger glyphs. CLAHE raises local
contrast without the ringing that a sharpening kernel adds to compressed video.
Cropping first means OCR sees one field at a time, with nothing nearby to confuse it.

### 4. Cleaning — `cleaning.py`

One cleaner per class. Each returns a typed value or `None`.

| Class | Rule | Example |
|---|---|---|
| `time` | Extract digit groups. Two groups → `HH:MM`. Three → `HH:MM:SS`, assigned as **seconds, hours, minutes**. | `30 03 29` → `03:29:30` |
| `station_id` | Strip punctuation, fix letter-for-digit confusions in the numeric part (`B`→8, `O`→0, `S`→5), force the `BD` prefix, snap to the nearest known ID within `MAX_EDIT_DISTANCE` (2). Ties → `None`. | `[BD1O6]` → `BD106` |
| `passenger_increse` | Keep digits, cast to integer. | `3 pax` → `3` |

**Why the odd time order:** the clock shows seconds as a small superscript
beside the large `HH:MM`, and EasyOCR returns that small group first. The order
in `clean_time` matches what the OCR engine produces. It looks wrong and is correct.

**Why snap to a closed set:** OCR errors on a five-character code are common,
but the valid codes are known. Matching against them corrects most single- and
double-character errors. Rejecting ties avoids assigning passengers to the wrong station.

### 5. Row assembly — `assembly.py`

The detector returns a flat list of boxes with no notion of rows. Assembly
sorts the table boxes by vertical centre and starts a new row whenever the gap
to the previous box exceeds `ROW_Y_TOLERANCE` (20 px). The clock is handled
separately: it belongs to the whole frame, so its value is attached to every row.

`station_name` is filled from `STATION_LOOKUP[station_id]` here.

**Why geometry rather than a "row" class:** rows in this UI have a fixed
height and never wrap, so a one-dimensional clustering is enough, costs nothing
to annotate, and can be tuned with one number.

### 6. Post-processing — `postprocessing.py`

Runs once over all rows from all frames. Order matters.

| Pass | What it does | Why |
|---|---|---|
| 1. Same-timestamp dedup | Keeps the most complete row per frame | Row grouping sometimes emits a real row plus a fragment of it |
| 2. Drop empty rows | Removes rows with neither a count nor a station | Nothing was read; the row carries no information |
| 3. Temporal dedup | Removes a row identical to the previous one (same station and count) within `TEMPORAL_DEDUP_WINDOW_SEC` (5 s) | The same row stays on screen across several sampled frames |
| 4. Fill names | Re-derives `station_name`; marks unreadable IDs as `OCR MISSREAD` / `UNKNOWN ID` | Makes gaps explicit in the CSV instead of leaving blanks |

Pass 1 runs before pass 3 so that fragments cannot break up a run of identical
rows and let duplicates through.

### 7. Export — `export.py`

Writes the columns in a fixed order
(`timestamp, time, station_id, station_name, passenger_increse`), with counts
as integers and timestamps rounded to milliseconds. Both the web app's download
button and the API call this function.

## Output

```csv
timestamp,time,station_id,station_name,passenger_increse
0.0,10:40:57,BD106,Bandung,2
10.058,10:42:36,OCR MISSREAD,UNKNOWN ID,2
14.081,10:43:16,BD093,Dayeuh Kolot,
```

- An empty `passenger_increse` means the station was read but the count was not.
- `OCR MISSREAD` means a count was read but the station code could not be
  matched to a known station. The analytics charts exclude these rows.
- A video with no detections yields a header-only CSV.

## Performance

On an Apple-silicon laptop CPU, a 2880×1800 recording takes roughly 1.1 seconds
per sampled frame (20 sampled frames in 23 s). At the default 1 fps that is
slightly slower than real time. Most of that is EasyOCR; lowering the sampling
rate is the most effective way to speed a run up.

## Known limitations

- **One event per frame.** Pass 1 keeps a single row per sampled frame. When the
  panel shows several rows at once, the others are lost. This was accepted
  because fragments were a bigger problem than multi-row frames in the
  recordings tested; it is the first thing to revisit for busier stations.
- **Occasional clock misreads.** A misread digit gives a time slightly out of
  sequence. `timestamp` (video offset) is always monotonic and can be used to
  order rows or to spot such values.
- **`BD103` is ambiguous.** The code maps to two stations that only the
  numbering tells apart. Since numbering is not read, it resolves to "Pungkur".
- **Stations outside `STATION_LOOKUP`** are reported as `OCR MISSREAD`.
- **Models are not thread-safe.** One video is processed at a time per process.
