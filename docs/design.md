# Problem and design

## Problem statement

The train simulator shows passenger demand in a station panel: a list where
each row reads like `1 pax · waiting for line L-20.1 · next stop [BD108] (4-13) Hasan Sadikin`.
Rows appear as new passengers arrive and disappear as they board. The in-game
clock sits in the top toolbar.

That panel is the only place the information exists. The game does not export
it, so answering questions such as "which destinations draw the most passengers
from this station?" or "when does demand peak?" means watching the panel and
writing numbers down by hand.

The goal of this project is to replace that manual step: given a screen
recording of the panel, produce a table with one row per passenger event.

| Column | Meaning |
|---|---|
| `timestamp` | Seconds from the start of the video to the frame the event was read from |
| `time` | In-game clock on that frame, `HH:MM:SS` (or `HH:MM` if seconds were not read) |
| `station_id` | Destination station code, e.g. `BD108` |
| `station_name` | Destination station name, looked up from `station_id` |
| `passenger_increse` | Number of passengers in the row |

### What makes it hard

- **The panel moves.** It is a draggable, resizable window, so the text is not at
  fixed pixel coordinates. Cropping fixed regions does not work.
- **Rows repeat a pattern.** One frame can contain several rows, each with the
  same kinds of field. The system has to know which count belongs to which station.
- **Fields look alike.** `[BD108]`, `(4-13)` and `[37]` are all short bracketed
  codes sitting next to each other.
- **A video repeats itself.** The same row stays on screen for many seconds, so
  naive frame-by-frame reading produces the same event dozens of times.
- **Recordings vary.** Resolution, compression (especially YouTube re-encodes)
  and brightness differ between captures.

### What makes it tractable

The source is rendered UI text, not a photograph. Fonts are consistent, text is
always horizontal, there is no perspective distortion or motion blur, and the
set of stations is finite and known in advance.

## Design thinking

### Split the job in three: find, read, reason

| Stage | Tool | Decides |
|---|---|---|
| Find | Fine-tuned YOLOv8s | *Where* each piece of text is and *what kind of field* it is |
| Read | EasyOCR | *Which characters* are in a crop |
| Reason | Plain Python rules | Whether the value is valid, which row it belongs to, whether it is a repeat |

Each stage does the one thing it is good at. The detector is the only place
field identity is decided; OCR never sees more than one field at a time, so it
cannot mix a station code with a platform number. Everything after OCR is
deterministic and can be read, tested and changed without retraining anything.

**Why not OCR the whole frame?** Full-frame OCR returns every string on screen
— toolbar buttons, money, map labels — with no indication of what each one is.
Telling a station code from a line number would then depend on fragile regular
expressions and on positions that change whenever the panel is dragged.

**Why not a single end-to-end model?** A model that goes straight from pixels
to a table would need far more labelled data than the 32 screenshots this
project was trained on, and its mistakes would be opaque. With the three-stage
design, a wrong value can be traced to a specific stage.

### Use what is already known instead of reading it

Station names vary in length and proved to be the hardest class for the
detector (see [Data and training](data-and-training.md#results)). But the name
is fully determined by the station code, and the network's stations are known.
So the pipeline never reads the name: it reads the code and looks the name up
in `STATION_LOOKUP`.

The same closed set repairs noisy codes. An OCR result such as `BD1O6` (letter O for zero) is
snapped to the nearest known code (`BD106`) by edit distance. If two known codes
are equally close, the value is rejected — a blank is easier to deal with
downstream than a confident wrong answer.

The detector is still trained on five classes, including the two the pipeline
does not read (`station_name`, `station_numbering`). Teaching it what those
regions are stops it from labelling them as something else.

### Prefer a missing value to a wrong one

Every cleaner returns `None` when it cannot produce a valid value. Rows where
the station could not be recovered are kept and marked `OCR MISSREAD` rather
than dropped or guessed, so a reader of the CSV can see exactly how much was lost.

### One pipeline, two surfaces

All extraction logic lives in `application/pipeline/` and knows nothing about
how it is being called. The Gradio app and the REST API are thin shells around
it and share one CSV serialiser, so they always return the same data for the
same video. `application/config.py` holds every threshold and the station
table, so tuning never requires touching pipeline code.

### Sample frames, then deduplicate rows

Processing cost scales with frames, not with video length, and on-screen state
changes slowly. The pipeline reads one frame per second by default and removes
repeats afterwards by comparing consecutive rows. This is simpler than tracking
objects across frames and is good enough for a list that changes every few seconds.

## Scope and limits

- The station table covers one network (codes `BD093`–`BD120`). A new network
  needs new entries in `STATION_LOOKUP`; a different UI needs new training data.
- One event is kept per sampled frame. If several rows are visible at once, only
  the most complete one survives (see [Pipeline](pipeline.md#known-limitations)).
- Detection confidence is used as a threshold but is not written to the CSV.
- The model was validated on a small set that overlaps the training set, so its
  reported accuracy is optimistic (see [Data and training](data-and-training.md#known-weaknesses)).
