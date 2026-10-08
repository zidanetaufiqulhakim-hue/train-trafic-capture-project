# Data and training

How the detector's training set was built, how the model was configured and
why, what it scored, and where it is weak.

The work lives in [`notebooks/passenger_detector_training.ipynb`](../notebooks/passenger_detector_training.ipynb). The run that produced the shipped
weights is recorded in
`notebooks/runs/detect/train_passenger_capture/yolov8s_mps_run/` (`args.yaml`
holds the full configuration, `results.csv` the per-epoch metrics). Datasets
and run artifacts are kept locally and are not part of the deployed repository.

## Dataset

### Collection

Training images are screenshots of the simulator with a station panel open,
captured on a Retina display (2880 px wide) during play sessions in August 2026.
72 screenshots were collected into `data/raw_images/`.

Screenshots were chosen over extracted video frames because consecutive video
frames are near-identical: hand-picking moments gives more variety per labelled
image — different panel positions, row counts, stations and clock values.

### Annotation

32 of the screenshots were labelled in **Label Studio** and exported in YOLO
format (`data/annotated_dataset/`). Every visible instance of five classes was boxed:

| ID | Class | What is boxed | Read by OCR? |
|---|---|---|---|
| 0 | `passenger_increse` | The passenger count at the start of a row (`3 pax`) | Yes |
| 1 | `station_id` | The bracketed station code (`[BD108]`) | Yes |
| 2 | `station_name` | The station's name (`Hasan Sadikin`) | No — looked up from the ID |
| 3 | `station_numbering` | The parenthesised number (`(4-13)`) | No — discarded |
| 4 | `time` | The clock in the top toolbar | Yes |

The class is named `passenger_increase` in the Label Studio export and
`passenger_increse` in `dataset.yaml`. The second spelling is the one baked
into the weights, which is why the rest of the codebase keeps it.

### Augmentation

The 32 labelled images were expanded offline with Albumentations. Each
transform was applied to a random share of the images, chosen to mimic the ways
real recordings differ from the screenshots:

| Transform | Share of images | Variation it stands in for |
|---|---|---|
| Brightness / contrast jitter (±0.2) | 35% | Display and in-game brightness settings |
| Scale up 5–10% | 25% | Different recording resolutions and window sizes |
| JPEG compression (quality 60–90) | 20% | Compressed recordings and YouTube re-encodes |
| Translation (±3%) | 15% | The panel sitting at different positions |
| Gaussian noise | 5% | Imperfect captures; stops reliance on pixel-perfect edges |

The noise transform ran at Albumentations' default strength: the installed
version did not accept the `var_limit` argument the notebook passes.

Augmented files were written back into the same folder, and the step was run
more than once, so later passes also sampled earlier outputs (file names such
as `…_aug_57733_aug_56476.png` are a transform of a transform).

Rotation, flips, shear and mosaic were deliberately left out: UI text is always
upright and axis-aligned, and mirrored or tiled text is not something the model
will ever see.

### Split

Training used `data/cleaned_dataset/`:

| Split | Images | `passenger_increse` | `station_id` | `station_name` | `station_numbering` | `time` | Total boxes |
|---|---|---|---|---|---|---|---|
| train | 110 | 425 | 430 | 430 | 390 | 66 | 1,741 |
| val | 34 | 117 | 137 | 137 | 120 | 20 | 531 |

`time` has far fewer instances than the other classes because there is one
clock per frame but several rows. It is also the visually simplest class — one
fixed-format widget — and it ended up the best-performing one, so no
rebalancing was done.

## Training configuration

```python
model = YOLO("yolov8s.pt")
model.train(
    data="data/cleaned_dataset/dataset.yaml",
    epochs=180, patience=25, batch=16, imgsz=640, device="mps", workers=4,
    freeze=10,
    optimizer="auto", lr0=0.001, cos_lr=True, warmup_epochs=3,
    mosaic=0.0, mixup=0.0, fliplr=0.0, flipud=0.0, degrees=0.0,
    shear=0.0, perspective=0.0, copy_paste=0.0, hsv_h=0.0,
    hsv_s=0.15, hsv_v=0.2, translate=0.05, scale=0.15,
    save_period=10, seed=42, deterministic=True,
)
```

| Setting | Value | Why |
|---|---|---|
| Base model | `yolov8s.pt` (COCO-pretrained) | No public model knows these classes, but pretrained low-level features (edges, corners, strokes) transfer. The "small" variant is large enough to separate look-alike fields and small enough to run on CPU at inference time (11.1 M parameters, 22 MB). |
| `freeze` | `10` | Freezes the backbone; only the neck and head train. With ~110 training images, a fully trainable network would memorise them. Freezing is the main guard against overfitting here. |
| `optimizer` | `auto` | Ultralytics picks AdamW for short runs like this one. Note that `auto` also sets the learning rate itself and ignores `lr0`: the run used ≈0.0011, which happens to be close to the 0.001 requested. |
| `cos_lr`, `warmup_epochs` | `True`, `3` | Warm-up protects pretrained weights from large early gradients; cosine decay lets the head settle at the end. |
| `epochs`, `patience` | `180`, `25` | An upper bound with early stopping. Validation mAP kept improving slowly, so early stopping never triggered and all 180 epochs ran. |
| `imgsz` | `640` | Ultralytics default. Frames are downscaled for detection only; OCR crops are taken from the full-resolution frame, so text legibility for reading does not depend on this. |
| `mosaic`, `mixup`, `copy_paste` | off | These stitch or blend images. They would break rows apart and place fields in arrangements the UI never produces. |
| `fliplr`, `flipud`, `degrees`, `shear`, `perspective` | off | The UI is always upright and axis-aligned. |
| `hsv_h` | `0` | Hue is fixed by the game's theme; shifting it teaches nothing. |
| `hsv_s`, `hsv_v`, `translate`, `scale` | `0.15`, `0.2`, `0.05`, `0.15` | Small amounts of the variation that does occur: display brightness, panel position, window size. |
| `seed`, `deterministic` | `42`, `True` | Reproducible runs, so a change in metrics can be attributed to a change in data or settings. |
| `save_period` | `10` | Keeps intermediate checkpoints for inspecting training later. |
| `device` | `mps` | Trained on an Apple-silicon laptop GPU. The run took about 96 minutes. |

## Results

Validation of the shipped weights (`best.pt`) on the 34-image validation split,
taken from the end of the training log:

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **All** | 531 | 0.936 | 0.883 | **0.948** | 0.769 |
| `time` | 20 | 0.921 | 0.950 | 0.988 | 0.856 |
| `station_id` | 137 | 0.929 | 0.859 | 0.958 | 0.787 |
| `passenger_increse` | 117 | 0.944 | 0.857 | 0.958 | 0.729 |
| `station_name` | 137 | 0.945 | 0.869 | 0.918 | 0.722 |
| `station_numbering` | 120 | 0.941 | 0.883 | 0.916 | 0.751 |

The project target was mAP@0.5 ≥ 0.85. Training and validation losses fell
together without diverging, and `time` — the class with the fewest examples —
scored highest.

To regenerate these figures:

```bash
yolo val model=model_registry/passenger_capture_yolov8s.pt data=data/cleaned_dataset/dataset.yaml
```

The notebook's curves and commentary were written against a separate run
(`yolov8s_mps_run-5`, mAP@0.5 0.968 on 27 images). In that run `station_name`
had clearly the lowest recall (0.873, every other class above 0.94), most
likely because name length varies far more than any other field.

**This finding shaped the pipeline:** rather than collect more data to fix a
field whose value is already implied by the station code, the pipeline stopped
reading names and looks them up instead. In the shipped weights too, the three
classes the pipeline reads have the three highest mAP@0.5 scores. Their recall
(0.86–0.95) is the figure to improve next, since a missed box is a lost value.

## Known weaknesses

These are properties of the current dataset, recorded so the numbers above are
read correctly and the next training round can address them.

1. **Validation overlaps training.** The split was made after augmentation, per
   image. All 19 source screenshots present in the validation split also appear
   in the training split, as the original or an augmented copy. Validation
   scores therefore measure how well the model recognises screenshots it has
   largely seen, and overstate accuracy on new recordings.
2. **Translated images kept their original labels.** The offline augmentation
   copies each label file unchanged. That is correct for brightness, compression
   and noise, and for scaling (YOLO coordinates are relative). It is not correct
   for translation: in that 15% of augmented images the boxes are offset from
   the text by up to 3% of the image size.
3. **Small source set.** Everything derives from 32 labelled screenshots of one
   network, at one display resolution, in one UI theme.
4. **The dataset is not fully reproducible from the notebook.** The split cell
   writes to `data/train` and `data/val`; training read `data/cleaned_dataset`.

## Retraining

When the UI changes or a new network is added:

1. Capture new screenshots and label them in Label Studio with the same five classes.
2. Split by **source screenshot** before augmenting, so no screenshot contributes
   to both splits. Ideally hold out a whole recording session.
3. Augment the training split only. Either drop the translation transform or
   apply it with Albumentations' `bbox_params` so labels move with the image.
4. Train with the configuration above and compare per-class recall, not just mAP.
5. Copy the new `best.pt` to `model_registry/passenger_capture_yolov8s.pt` (or
   point `YOLO_MODEL_PATH` at it) and add any new stations to `STATION_LOOKUP`.
