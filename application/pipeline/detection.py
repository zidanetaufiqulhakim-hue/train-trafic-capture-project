"""
YOLO inference on a single video frame.

detect_boxes() is the only public function.
It returns a typed list of Box objects — no raw YOLO internals leak out of this module.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Box:
    """One detected bounding box from YOLO."""
    cls_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    y_center: float   # used for row-grouping in assembly.py
    confidence: float


def detect_boxes(
    frame: np.ndarray,
    model: YOLO,
    class_names: list[str],
    conf_threshold: float,
    skipped_classes: frozenset[str],
) -> list[Box]:
    """Runs YOLO on a frame and returns detected boxes, excluding skipped classes.

    skipped_classes are those YOLO detects but the pipeline does not OCR
    (station_name and station_numbering — see config.SKIPPED_CLASSES)."""
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    boxes: list[Box] = []

    for raw in results.boxes:
        cls_idx = int(raw.cls[0])
        if cls_idx >= len(class_names):
            continue
        cls_name = class_names[cls_idx]
        if cls_name in skipped_classes:
            continue
        x1, y1, x2, y2 = map(int, raw.xyxy[0])
        boxes.append(Box(
            cls_name=cls_name,
            x1=x1, y1=y1, x2=x2, y2=y2,
            y_center=(y1 + y2) / 2,
            confidence=float(raw.conf[0]),
        ))

    return boxes
