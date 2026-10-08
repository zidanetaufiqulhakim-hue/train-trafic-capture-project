# Train passenger traffic capture — project plan

## 1. Overview

The product takes a screen recording of a train-sim game UI (e.g. the station/platform panel shown in the reference screenshot), detects five recurring text regions per frame, reads their text with OCR, and lets the user turn that into a structured, filterable, downloadable CSV — plus quick charts. Delivered as a Streamlit app.

**Source material matters here**: this is *rendered UI text* from a game, not a photo of physical signage. That's good news for OCR accuracy (crisp, consistent font, no perspective distortion, no motion blur), but it introduces its own failure modes: the panel can be dragged/resized, the visible row count varies, and windows can overlap other UI chrome. The plan below is built around that reality rather than generic "text in the wild" assumptions.

## 2. Class taxonomy (grounded in the reference screenshot)

| Class | Example from screenshot | Where it lives | Notes | OCR'd? |
|---|---|---|---|---|
| `time` | `07:37`, `Saturday July 18, 2026` | Fixed top toolbar | Usually one instance per frame, fixed screen position | Yes |
| `station_id` | `BD100`, `BD102`, `BD110` | Bracketed code before station name | Alphanumeric, fixed short pattern; fuzzy-matched to closed set post-OCR | Yes |
| `station_numbering` | `(4-18)`, `(4-11)` | Parenthetical, sometimes absent | YOLO detects it but OCR is **skipped** — not in output schema | No |
| `station_name` | `Moch.Toha`, `Tegal Lega`, `Pasteur` | Free text after id/numbering | YOLO detects it but OCR is **skipped** — derived from `STATION_LOOKUP` keyed on `station_id` | No |
| `passenger_increse` | `3`, `10` | Left-hand list column | Numeric only (unit stripped); repeats per row | Yes |

**Output schema** (columns in the final DataFrame/CSV): `timestamp`, `time`, `station_id`, `station_name`, `passenger_increse`.

Key design implication: several of these classes appear **multiple times per frame** in a **list** (passenger rows, next-stop rows), not just once. The pipeline needs to treat each frame as "N instances of 5 classes," not "1 instance of 5 classes."

**Note on `station_name` / `station_numbering`**: Both are still detected by YOLO (useful for bounding-box training accuracy metrics), but neither goes through EasyOCR. `station_name` is resolved entirely from `STATION_LOOKUP`, a hardcoded dict of known `station_id → name` pairs. `station_numbering` is discarded — the lookup key is `station_id` alone, so the numbering adds no information once the ID is cleanly matched. If the game adds a station with an ambiguous ID (e.g. two names map to the same code), `station_numbering` would need to be re-introduced as a disambiguator and the lookup replaced with a two-key table.

## 3. System workflow (as specified, refined)

1. User uploads a screen-recording video.
2. User selects which classes they want captured (checkboxes: time, station_id, station_numbering, station_name, passenger).
3. User clicks **Capture**.
4. Backend samples frames → YOLO detects & classifies text regions → EasyOCR reads each region → post-processing validates/cleans/deduplicates.
5. Result is assembled into a structured table with one column per class (plus metadata: frame id, video timestamp, confidence).
6. User optionally groups the data (by station, by time bucket, etc.) with an aggregation choice.
7. User downloads raw CSV and/or grouped CSV.
8. User picks a chart type (histogram, scatter, line, pie) over the grouped data, previews it, and downloads the image.

## 4. Architecture / pipeline

```
video upload
   │
   ▼
frame sampling (fixed FPS, e.g. 1–2 fps via frame_interval)
   │
   ▼
YOLO detector (5 classes: time, station_id, station_numbering, station_name, passenger_increse)
   │  → boxes + class + confidence, possibly many boxes per class per frame
   │  station_name & station_numbering boxes are detected but immediately skipped (no OCR)
   ▼
crop + preprocess each active box
   │  → grayscale → 2× upscale (INTER_CUBIC) → CLAHE contrast enhancement → back to BGR
   ▼
EasyOCR per crop (time, station_id, passenger_increse only)
   │
   ▼
per-class text cleaning:
   │  time          → normalize to HH:MM or HH:MM:SS
   │  station_id    → strip non-alphanumeric → force "BD" prefix → edit-distance
   │                   snap to nearest key in STATION_LOOKUP (returns None if ambiguous)
   │  passenger_increse → strip non-numeric → cast to int
   ▼
y-band row assembly (Approach A, §15)
   │  group boxes by y_center proximity; time is extracted globally, not per-row
   ▼
station_name lookup from STATION_LOOKUP[station_id]
   │
   ▼
post-processing pipeline (in order):
   │  1. drop_same_timestamp_duplicates — keep most-complete row per frame timestamp
   │  2. drop_missing_rows             — drop rows where all key OCR fields are null
   │  3. temporal_deduplication        — remove consecutive identical rows within 5 s window
   │  4. impute_missing_station_id_and_names — fill derived station_name; mark null IDs
   ▼
structured table (pandas DataFrame) → Streamlit preview
   │
   ├─→ user column selection → raw CSV
   ├─→ user group-by + aggregation → grouped CSV
   └─→ chart builder (histogram / scatter / line / pie) → image download
```

The "row assembly" step is the part that's easy to underestimate: a `pax` value and a `station_name` are two independent detections. Associating the right `pax` with the right `station_name` in a given row requires either (a) fixed relative geometry within the panel (e.g. same row-index / y-band), or (b) training YOLO to detect the *row* as a container and the fields as children within it. See §6.6.

## 5. Requirements

### 5.1 Functional
- Accept common video formats (mp4, mov, webm) and screen-recording frame rates.
- Detect and classify the 5 text classes across frames without requiring a fixed camera/window position (panel is draggable/resizable).
- Extract text per class via OCR with a confidence score attached to every extracted value.
- Let the user choose, before running capture, which classes to include (reduces compute and noise).
- Assemble output into rows with consistent columns; support multiple instances per frame (multiple `pax` rows, multiple next-stop rows).
- Deduplicate repeated detections across consecutive frames of the same on-screen state.
- Group-by with a chosen aggregation function (sum, count, mean, first/last).
- Export raw and grouped CSV.
- Generate and export histogram, scatter, line, and pie chart views of the grouped data.

### 5.2 Non-functional
- Processing time should be proportional to sampled frames, not raw video length — sampling rate is the main lever for speed vs completeness.
- The app should degrade gracefully: low-confidence values are flagged, not silently dropped or silently wrong.
- The pipeline should be re-trainable if the game UI changes (new patch, new panel layout) without a full rebuild — annotation and training process should be repeatable, not one-off.

## 6. Answers to your specific questions

### 6.1 UX/UI — Streamlit flow
Structure it as a linear wizard (matches the mockup above), not one dense page:
1. **Upload** — drag-and-drop, show video duration and estimated frame count once uploaded.
2. **Select fields** — checkboxes for the 5 classes, shown *before* capture so the user isn't waiting on classes they don't need.
3. **Capture** — a single button, with a progress bar (`st.progress`) since this can take a while on longer videos. Show frame-by-frame progress, not a spinner — users trust visible progress much more on multi-minute jobs.
4. **Preview table** — `st.dataframe` with a confidence column, sortable, with low-confidence rows visually flagged (e.g. conditional formatting). This is the trust-building step: let the user *see* what got extracted before committing to a download.
5. **Group by** — a `st.selectbox` for the group column(s) and an aggregation choice, live-updating the preview.
6. **Export** — `st.download_button` for raw CSV and grouped CSV, separately, so the user isn't forced to pick one.
7. **Visualize** — chart-type selector (histogram / scatter / line / pie), rendered with Plotly or matplotlib, with its own download button for the image.

General UX principles that matter most here: show confidence/uncertainty visibly rather than hiding it, let the user preview before committing to an export, and keep field selection *before* the expensive step so they're not paying compute for data they don't want.

### 6.2 Do you need your own annotated dataset, or is there a pretrained model?
You need your own. Off-the-shelf YOLO (COCO-pretrained) detects generic objects like "person" or "car," not "station_id" or "pax count" — those are domain-specific text classes that don't exist in any pretrained checkpoint. What you *can* reuse is the pretrained backbone: fine-tune a COCO-pretrained YOLOv8/YOLOv11 on your own annotated screenshots rather than training from scratch. That still needs your own labeled data, just less of it (transfer learning cuts the data requirement substantially versus training from zero).

One thing worth exploiting because your source is a *rendered game UI*, not a photo: you can generate a meaningful chunk of synthetic training data by scripting the game (or replaying saved states) to produce screenshots with known ground-truth text, rather than hand-labeling everything. Real, hand-labeled screenshots are still needed for validation and to cover edge cases (overlapping windows, scrollable lists, different zoom levels), but synthetic generation can reduce the manual annotation burden significantly for a UI this structured.

Annotation tool: LabelImg works, but for this task Roboflow or CVAT are worth considering too — they support YOLO-format export directly, plus augmentation (brightness, scale, slight UI-position jitter) baked into the export step.

### 6.3 Dataset folder structure

Standard YOLO layout:

```
dataset/
├── data.yaml                 # class names + paths, read by YOLO at train time
├── images/
│   ├── train/
│   │   ├── clip01_f00012.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── clip01_f00012.txt   # YOLO format: class_id x_center y_center w h (normalized)
│   │   └── ...
│   ├── val/
│   └── test/
└── raw_videos/                # original screen recordings, not used directly by YOLO
    └── clip01.mp4
```

`data.yaml`:
```yaml
train: images/train
val: images/val
test: images/test
nc: 5
names: ['passenger_increse', 'station_id', 'station_name', 'station_numbering', 'time']
```
*(class names stored in model metadata alphabetically; the pipeline reads them dynamically from `yolo_model.names` at load time)*

Recommended split: ~70/20/10 train/val/test, split by **video clip**, not by individual frame — frames from the same clip are highly correlated (same panel position, same session), so splitting by frame leaks information between train and val and inflates your apparent accuracy.

### 6.4 Turning captured text into CSV
After OCR produces raw strings, each goes through a class-specific cleaner before reaching the table:

- `time`: digits extracted with `re.findall(r'\d+', ...)`. Two groups → `HH:MM`; three groups → `HH:MM:SS`. Returns `None` on failure (not an empty string — downstream code can distinguish missing from zero).
- `station_id`: strip non-alphanumeric → force `"BD"` prefix → run edit-distance snap via `_closest_valid_station_id` against `STATION_LOOKUP` keys. Returns `None` if no match within `station_id_max_edit_distance` (default 2) or if two IDs are tied (ambiguous). This is stronger than a bare regex: it catches OCR substitutions (B→8, O→0, S→5) and extra/missing digits uniformly.
- `station_numbering`: **not cleaned — OCR is skipped entirely** for this class.
- `station_name`: **not OCR'd** — resolved from `STATION_LOOKUP.get(station_id)` after the station_id is cleaned. Returns `None` if the station_id is unknown.
- `passenger_increse`: strip all non-numeric characters, cast to `int`. Returns `None` if nothing numeric remains.

Final DataFrame columns: `timestamp, time, station_id, station_name, passenger_increse`. Confidence scores per class are not currently stored in the output schema (the raw YOLO confidence is used internally for the 0.3 detection threshold but is not propagated to the CSV).

### 6.5 Letting the user select their own fields (Streamlit)
Two selection points, not one:
- **Before capture**: which classes to detect/OCR at all (saves compute, keeps the job focused).
- **After capture, before export**: `st.multiselect` over the resulting DataFrame's columns, so the user can drop columns like confidence scores or frame metadata from the final CSV without re-running capture. Keep this separate from the pre-capture selection — a user might want to *detect* everything but only *export* three columns.

### 6.6 Preventing class mixing and missed detections
This is fundamentally a detection-quality problem, addressed at three layers:

1. **Model layer** — train with enough examples per class that visually similar classes (`station_id` vs `station_numbering`, both short alphanumeric-ish strings) are well separated; check the confusion matrix specifically for these pairs after training, not just overall mAP. Use a confidence threshold and Non-Max Suppression tuned per class, since classes like `passenger` (small, repeating) behave differently than `time` (single instance, high visual regularity).
2. **Geometric/structural layer** — since this is a fixed-layout UI (not free-form photography), encode the known spatial relationships as a rule, not just hope the model learns it: `station_id` and `station_name` on the same row should sit at a consistent x-offset and the same y-band. If a detection's class disagrees with its expected position relative to other detections in the same row, flag it for review rather than accepting it silently. This catches mixing that slips past the model.
3. **Post-OCR validation layer** — the regex/format checks in §6.4 act as a second independent check: if a box was classified as `time` but the OCR text doesn't match `HH:MM`, that's a strong signal of misclassification, not just a bad OCR read.

For **missed detections**: track objects across consecutive frames (simple IOU-based tracking) so a station that's briefly occluded or has a low-confidence frame can still be filled in from neighboring frames, rather than silently disappearing from the output. Also report a per-video "expected vs captured" sanity count where possible (e.g. if the panel shows a fixed row count, check the extracted rows against it).

### 6.7 Group-by feature
Straightforward `pandas.groupby` on top of the assembled DataFrame, exposed in Streamlit as: pick group column(s) (`station_name`, `station_id`, or a derived hourly bucket from `time`) and an aggregation (`sum`, `count`, `mean`, `first`). Output is itself just another DataFrame, so it reuses the same preview/download/chart components as the raw table — no separate code path needed.

## 7. Visualization module
Given grouped data, offer:
- **Histogram** — distribution of a numeric column (e.g. passenger counts).
- **Scatter** — two numeric columns (e.g. time bucket vs passenger sum).
- **Line** — trend over time (e.g. passenger sum per hour).
- **Pie** — share of a metric across a categorical group (e.g. passenger share by station).

Use Plotly for interactivity in-app (hover, zoom) with a static image export (`fig.write_image`, needs `kaleido`) for the download button.

## 8. Suggested milestones

| Phase | Duration | Output |
|---|---|---|
| 1. Class taxonomy lock + annotation setup | 3–5 days | Finalized class rules, annotation tool configured |
| 2. Dataset build (synthetic + hand-labeled) | 1–2 weeks | Labeled dataset in YOLO format, train/val/test split |
| 3. YOLO training + confusion-matrix tuning | 1 week | Trained detector, per-class precision/recall report |
| 4. OCR + validation layer | 3–5 days | Per-class regex validators, cleaned extraction pipeline |
| 5. Row assembly + temporal dedup/tracking | 3–5 days | Structured DataFrame builder |
| 6. Streamlit app (upload → capture → preview → export) | 1 week | Working end-to-end app |
| 7. Group-by + visualization module | 3–5 days | Charting UI with export |
| 8. Testing on held-out recordings, mixing-rate measurement | 1 week | Accuracy report, tuning backlog |

## 9. Risks & trade-offs

- **Panel is movable/resizable** → position-based heuristics (§6.6) need to be relative, not absolute pixel coordinates.
- **Variable row count per frame** → sampling rate and dedup logic must handle scroll/pagination if the passenger list can exceed the visible panel.
- **Sampling rate vs completeness** → higher FPS sampling catches more transient states but multiplies compute cost and duplicate rows; start conservative (2–3 fps) and raise only if missed-row testing shows gaps.
- **Synthetic data realism** → synthetic screenshots reduce labeling effort but must be mixed with real recordings so the model doesn't overfit to clean synthetic renders and fail on real capture artifacts (compression, scaling).

## 10. Division of responsibility: YOLO vs OCR

This is worth stating explicitly since it determines where engineering effort should go:

- **YOLO detects AND classifies.** Its output per box is `(bounding_box, class_label, confidence)` — it decides both *where* a piece of text is and *which of the 5 classes it belongs to*. This is the only place in the pipeline where class identity is decided.
- **EasyOCR only reads characters** for three of the five classes: `time`, `station_id`, and `passenger_increse`. It has no concept of class — it reads whatever crop it receives. It cannot fix a class mistake.
- **`station_name` and `station_numbering` bypass OCR entirely.** YOLO still detects these boxes (useful for evaluation metrics), but the pipeline immediately skips them. `station_name` is resolved from a hardcoded lookup table; `station_numbering` is discarded.

Implication: **all class-mixing prevention work belongs in the YOLO layer**. The post-OCR cleaning in §6.4 is a *secondary* check (e.g. a box classified as `time` that doesn't parse to `HH:MM` returns `None`), not a replacement for correct upstream classification. The edit-distance snap on `station_id` is a noise-correction step, not a class-correction step — it assumes the box is already correctly classified as `station_id`.

## 11. Deduplication strategy (implemented)

**Current approach: row-level post-processing in three sequential passes** (not frame-level image hashing — that's a possible future optimization, but not what's in the pipeline today).

### 11.1 Pass 1 — `drop_same_timestamp_duplicates`
Within a single sampled frame, YOLO's y-band grouping (§15) sometimes produces more than one row dict for the exact same timestamp — a full/real row plus a partial echo (e.g. a misread badge fragment sitting near a real row). This pass keeps only the most complete row per timestamp, scored by count of non-null key fields (`passenger_increse`, `station_id`, `station_name`). Ties keep the first-seen row (stable sort).

Caveat: if the UI legitimately shows two distinct real entries in one frame with the same timestamp, this collapses them to one. There's no way to distinguish a genuine second entry from a partial echo at this stage. This is an acceptable trade-off given the current single-station UI layout.

### 11.2 Pass 2 — `drop_missing_rows`
Drops rows where all key OCR'd fields (`passenger_increse`, `station_id`) are null. `station_name` is excluded from this check since it's derived, not OCR'd — a null `station_name` just means the station_id wasn't in the lookup, which is a data quality flag, not a reason to drop the row.

### 11.3 Pass 3 — `temporal_deduplication`
Removes consecutive identical rows (same `station_id` + `passenger_increse`) within a 5-second window. This handles the case where the same on-screen state persists across multiple sampled frames — the second and subsequent occurrences of an identical `(station_id, passenger_increse)` pair within 5 s are dropped.

NaN-safe: null values are temporarily filled with a sentinel `"__MISSING__"` so that `NaN == NaN` evaluates as True (two missing values in the same column → treated as identical, not different).

### 11.4 Pass 4 — `impute_missing_station_id_and_names`
Final sweep: re-derives `station_name` from `station_id` for any rows that slipped through with a null name, and fills null `station_id` cells with the string `"OCR MISSREAD"` so downstream CSV consumers don't get silent NaN values in a key column.

**Future optimization**: pre-filter at the frame level using perceptual image hashing (`imagehash.phash`) before YOLO runs — most of a static gameplay recording is pixel-identical frames, so this would cut YOLO inference time substantially. The current approach dedupes rows after the fact, not frames before inference.

## 12. Fine-tuning methodology (COCO-pretrained YOLOv8)

### 12.1 Setup
- Base checkpoint: `yolov8s.pt` (small) — pretrained on COCO. The trained model lives at `notebooks/runs/detect/train_passenger_capture/yolov8s_mps_run/weights/best.pt`.
- 5 classes (`nc: 5`): `passenger_increse`, `station_id`, `station_name`, `station_numbering`, `time` (alphabetical order as stored in model metadata). The backbone and neck weights are fine-tuned, not frozen.

### 12.2 Training procedure
1. **Freeze-then-unfreeze warmup (optional but recommended).** Freeze the backbone for the first ~10 epochs (train only the head), then unfreeze everything for the remaining epochs. This avoids the pretrained features getting destroyed by large early gradients before the new head has stabilized.
2. **Full fine-tune.** Continue training all layers with a lower learning rate than a from-scratch run (e.g. `lr0: 0.001–0.005` instead of the default `0.01`), since you don't want to overwrite the useful pretrained low-level features too aggressively.
3. **Epochs.** Start around 100–150 epochs with early stopping (`patience`) on validation mAP — domain-narrow datasets like this tend to converge faster than general-purpose detection tasks, so watch for overfitting past ~80 epochs if the dataset is small.
4. **Augmentation, tuned for UI text (not natural photos).** Default YOLO augmentations (mosaic, heavy rotation, large color jitter) are built for natural-scene photography and will actively hurt you here — the UI is axis-aligned and color is meaningful (semantic status colors). Dial down/disable:
   - Rotation and shear → near zero (UI text is always horizontal).
   - Aggressive HSV/color jitter → reduce, since color can carry meaning in this UI.
   - Mosaic → use sparingly or disable; it can fragment small text regions unnaturally.
   - Keep: scale jitter, slight translation, and synthetic panel-position jitter (moving/resizing the captured window in synthetic generation) — this directly targets the real-world variation you'll see (draggable/resizable panel).
5. **Class balance check.** Passenger and next-stop rows repeat many times per frame while `time` appears once — this naturally skews instance counts per class. Verify per-class instance counts in the final dataset and oversample underrepresented classes (or their source frames) rather than letting `time` be starved of examples.

### 12.3 Evaluation
- Track per-class precision/recall and the confusion matrix specifically, not just overall mAP — mAP can look healthy while masking exactly the class-mixing failure mode you care about most (e.g. station_id ↔ station_numbering confusion).
- Hold out entire clips for validation/test (not random frames, per §6.3) so the model is evaluated on panel positions/sessions it hasn't memorized.
- Re-run confusion-matrix review after every retrain — this is your primary signal for whether §6.6's geometric/validation safety nets are still doing meaningful work or whether the model itself has closed the gap.

### 12.4 Iteration loop
Since this UI may change across game versions/patches (per the open question in §10 originally), structure training as a repeatable pipeline (versioned dataset, versioned `data.yaml`, tracked run configs — e.g. via Ultralytics' built-in run logging) rather than a one-off script, so a UI update means "add new labeled frames and re-run," not "rebuild from scratch."

## 13. Annotation guidelines

### 13.1 Tooling and workflow
- Use **CVAT or Roboflow** over raw LabelImg for this dataset — both support YOLO-format export directly, let multiple people annotate the same project with review/QA stages, and (Roboflow especially) let you version datasets as you add frames over time, which matters given §12.4's iterate-as-the-UI-changes plan. LabelImg is fine for a very small first batch but becomes a bottleneck once you're managing hundreds of frames with QA passes.
- Extract candidate frames from your videos first (e.g. every Nth frame, or run your frame-hashing dedup from §11 *before* annotation too — no point hand-labeling near-duplicate frames), then hand-pick a diverse subset covering different panel positions, row counts, and content variety rather than annotating sequentially through a video.

### 13.2 Bounding box rules — one rule per class, to prevent mixing at the source

Ambiguity in annotation guidelines is the single biggest cause of a model that mixes classes later, since the model can only be as consistent as its training labels.

- **`time`** — box the clock display only (e.g. `07:37`), not the date line above it. If you want the date captured too, treat it as a separate class rather than merging it into `time` — merging two different text formats into one class is exactly the kind of ambiguity that causes downstream regex validation (§6.4) to fail unpredictably.
- **`station_id`** — box only the bracketed code (`[BD102]`), excluding the brackets themselves if you plan to strip them in post-processing anyway — pick one convention (include or exclude brackets) and apply it to every single instance. Inconsistent bracket inclusion across annotators is a common, subtle source of noisy training data.
- **`station_numbering`** — box only the parenthetical (`(4-18)`). Since this field is often **absent**, do not force a box where there is no numbering — an empty/missing instance is correct ground truth, not a labeling gap. Don't be tempted to box the nearest text as a placeholder.
- **`station_name`** — box the free-text name only, right up to (but not including) the next class's start. This is the class most prone to boundary creep — annotators will be tempted to include a little of the surrounding whitespace or a trailing character from an adjacent field. Set a hard rule: the box ends exactly at the last visible character of the name, no padding.
- **`passenger_increse`** — box the numeric count only (the `pax` unit is stripped in post-processing, but boxing the whole `N pax` string is also fine since the cleaner strips non-numeric chars). Since these repeat many times per frame (§12.2 class balance note), keep every repetition in the row, don't sample only one per frame.

### 13.3 Handling multiple instances per frame
Since `passenger` and the next-stop list produce many boxes per frame, annotate **every visible instance**, not a representative sample — the model needs to learn "this class can and does appear N times in one image," and sparse labeling here teaches it the opposite. Use each tool's list/track view to keep row order consistent for your own review, even though YOLO training itself doesn't use that order.

### 13.4 Negative and hard-negative examples
Include some frames where a class is **legitimately absent** (e.g. no `station_numbering` on any row, or the panel closed and no station data visible at all) so the model learns absence is normal rather than always forcing a low-confidence guess. Also deliberately include frames where visually similar classes sit close together (`station_id` immediately next to `station_numbering`) since these are your highest-risk mixing cases per §6.6 — the model needs concentrated exposure to exactly the boundary it's most likely to get wrong.

### 13.5 Quality control
- Have a second person (or a second pass by the same annotator after a break) review a sample of completed frames against the class rules in §13.2 — annotation drift over a long labeling session is common and hard to self-catch in the moment.
- Spot-check bracket/whitespace consistency across the dataset before training — a quick script that pulls a sample of cropped `station_id` boxes and displays them together will immediately reveal if some include brackets and some don't.
- After the first training run, use the confusion matrix (§12.3) to identify which class pairs are actually being confused, then go back and add more annotated examples specifically at that boundary rather than annotating uniformly across all classes for round two.

## 15. Row association: how the model knows what belongs on one row

This is a real gap to be explicit about: **YOLO itself has no concept of "row."** It outputs a flat, unordered list of independent detections per frame — each box just has a class and a confidence, with no built-in relationship to any other box. Grouping `station_id`, `station_numbering`, `station_name`, and `passenger` into one logical row is entirely a **post-processing step you build**, not something the detector gives you for free. There are two viable ways to do it, and they're not mutually exclusive.

### 15.1 Approach A — geometric clustering after detection (recommended starting point)
After YOLO returns all boxes for a frame, group them into rows using the **y-coordinate band** each box falls into, since this UI lays fields out in horizontal rows:

1. Take the vertical center (`y_center`) of every detected box in the frame.
2. Sort all boxes by `y_center`.
3. Cluster boxes whose `y_center` values fall within a small tolerance of each other (e.g. within half a row-height) into the same row group — this is a simple 1D clustering problem, not full spatial reasoning.
4. Within a row group, order fields left-to-right by `x_center` to disambiguate if needed (e.g. confirming `station_id` precedes `station_name`, not just that they're vertically aligned).
5. A row is only "complete" if it has the expected combination of classes for that panel section (e.g. a next-stop row has `station_id` + `station_numbering`? + `station_name`; a waiting-list row has `passenger` alone). Rows missing an expected required field get flagged low-confidence rather than silently assembled with a gap.

This is *not* something YOLO does — it's plain post-detection logic (a few dozen lines of coordinate math), which is exactly why it needs to be designed carefully rather than assumed to "just work" from the model.

### 15.2 Approach B — train YOLO to detect the row itself as a container
Add a 6th class, e.g. `row`, that boxes the entire horizontal row as one region (the same idea as a "table row" detector). Then:
- Any field-level detection whose box center falls inside a `row` box is assigned to that row.
- This is more robust than pure y-banding because it lets the model itself learn the panel's actual row boundaries (including uneven spacing, or panel styles where rows aren't perfectly uniform height), rather than you hand-tuning a tolerance value.
- Costs more: it's another class to annotate consistently (box the full row extent, a similar boundary-discipline problem to `station_name` in §13.2), and it makes annotation slower since every frame needs both field-level *and* row-level boxes.

### 15.3 Recommendation
Start with **Approach A** — it's cheap, fast to implement, and this UI's rows are visually regular (fixed row height, no free-form layout), so simple y-band clustering should work well. Move to Approach B only if testing shows the panel has irregular row spacing, variable row heights (e.g. wrapped station names spanning two lines), or overlapping rows that geometric clustering can't reliably separate. Since Approach A's clustering logic is decoupled from the YOLO model itself, adopting it now doesn't block adding Approach B later — you'd just be replacing the y-band heuristic with row-box lookups without touching the rest of the pipeline.

## 16. Open questions for next decision point
- Fixed set of game UI layouts, or does the panel design change across game versions/mods? This affects how future-proof the trained model needs to be.
- Expected video length/volume per capture session — affects whether sampling-rate defaults need to be more aggressive.
- Any need for multi-station or multi-window frames in a single video, i.e. can more than one station panel be open at once?
