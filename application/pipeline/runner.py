"""
Pipeline orchestrator: loads models and runs the full video-to-DataFrame pipeline.

Public functions:
  load_models(model_path, use_gpu)  -> Models
  run_pipeline(...)                 -> pd.DataFrame   (blocking, with optional callback)
  iter_pipeline(...)                -> Generator       (yields frame progress, then final DataFrame)

iter_pipeline is the preferred entry point for UIs because it lets the caller
stream progress updates to the screen on every frame without threading or callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Union

import numpy as np
import pandas as pd
import easyocr
from ultralytics import YOLO

import application.config as config
from application.pipeline import video_io, detection, ocr, cleaning, assembly, postprocessing


@dataclass
class Models:
    """Holds the loaded YOLO and EasyOCR instances plus the class name list.

    Constructed once at startup by load_models() and reused for every video."""
    yolo: YOLO
    ocr_reader: easyocr.Reader
    class_names: list[str]


def load_models(model_path: str, use_gpu: bool = False) -> Models:
    """Loads YOLO and EasyOCR. Class names are read from the model's own metadata.

    Raises FileNotFoundError if model_path does not exist.
    Raises RuntimeError if the model fails to load for any other reason."""
    yolo = YOLO(model_path)
    # Use the model's own class name list so the pipeline stays in sync with training
    class_names = [yolo.names[i] for i in sorted(yolo.names.keys())]
    ocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=use_gpu)
    return Models(yolo=yolo, ocr_reader=ocr_reader, class_names=class_names)


def run_pipeline(
    video_path: str,
    models: Models,
    target_fps: float,
    on_progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Processes a video file end-to-end and returns a cleaned DataFrame.

    Calls on_progress(fraction, message) after each frame if provided —
    intended for driving a UI progress bar.

    Raises ValueError if the video cannot be opened."""
    total = video_io.count_sampled_frames(video_path, target_fps)
    cap = video_io.open_video(video_path)
    all_rows: list[dict] = []

    try:
        for i, (timestamp, frame) in enumerate(
            video_io.iter_sampled_frames(cap, target_fps), start=1
        ):
            rows = _process_frame(frame, timestamp, models)
            all_rows.extend(rows)
            if on_progress:
                on_progress(i / total, f"Frame {i} of {total}")
    finally:
        cap.release()

    if not all_rows:
        return pd.DataFrame(columns=config.OUTPUT_COLUMNS)

    raw_df = pd.DataFrame(all_rows)
    return postprocessing.run_postprocessing(
        raw_df, config.STATION_LOOKUP, config.TEMPORAL_DEDUP_WINDOW_SEC
    )


def iter_pipeline(
    video_path: str,
    models: Models,
    target_fps: float,
) -> Generator[Union[tuple[int, int], pd.DataFrame], None, None]:
    """Generator version of run_pipeline — preferred for UI use.

    Yields (frame_index, total_frames) after processing each frame so the caller
    can update a progress indicator without callbacks or threads.
    Yields the final cleaned DataFrame as the last item.

    Raises ValueError if the video cannot be opened."""
    total = video_io.count_sampled_frames(video_path, target_fps)
    cap = video_io.open_video(video_path)
    all_rows: list[dict] = []

    try:
        for i, (timestamp, frame) in enumerate(
            video_io.iter_sampled_frames(cap, target_fps), start=1
        ):
            all_rows.extend(_process_frame(frame, timestamp, models))
            yield i, total
    finally:
        cap.release()

    if not all_rows:
        yield pd.DataFrame(columns=config.OUTPUT_COLUMNS)
        return

    raw_df = pd.DataFrame(all_rows)
    yield postprocessing.run_postprocessing(
        raw_df, config.STATION_LOOKUP, config.TEMPORAL_DEDUP_WINDOW_SEC
    )


def _process_frame(frame: np.ndarray, timestamp: float, models: Models) -> list[dict]:
    """Runs one frame through detection → OCR → cleaning → row assembly.

    Kept private because callers should use run_pipeline(), which handles
    video reading, progress tracking, and post-processing."""
    boxes = detection.detect_boxes(
        frame,
        models.yolo,
        models.class_names,
        config.YOLO_CONF_THRESHOLD,
        config.SKIPPED_CLASSES,
    )

    ocr_results: dict[int, object] = {}
    for i, box in enumerate(boxes):
        crop = frame[box.y1:box.y2, box.x1:box.x2]
        if crop.size == 0:
            continue
        processed_crop = ocr.preprocess_crop(crop)
        raw_text = ocr.read_text(processed_crop, models.ocr_reader)

        if box.cls_name == "time":
            ocr_results[i] = cleaning.clean_time(raw_text)
        elif box.cls_name == "station_id":
            ocr_results[i] = cleaning.clean_station_id(
                raw_text, config.STATION_LOOKUP, config.MAX_EDIT_DISTANCE
            )
        elif box.cls_name == "passenger_increse":
            ocr_results[i] = cleaning.clean_passenger_count(raw_text)
        else:
            ocr_results[i] = raw_text.strip() or None

    return assembly.assemble_rows(
        boxes, ocr_results, timestamp, config.STATION_LOOKUP, config.ROW_Y_TOLERANCE
    )
