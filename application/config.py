"""
Single source of truth for all pipeline constants.

Edit this file to tune thresholds, add stations, or point to a new model.
Nothing else in the codebase hardcodes these values.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent

# Deployed weights live in model_registry/, not in the training run directory.
# Set the YOLO_MODEL_PATH environment variable to point at different weights
# without editing code (e.g. a volume-mounted model in the API container).
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    str(_PROJECT_ROOT / "model_registry" / "passenger_capture_yolov8s.pt"),
)

# Column order of the final DataFrame / CSV.
OUTPUT_COLUMNS = ["timestamp", "time", "station_id", "station_name", "passenger_increse"]

# ── Pipeline tuning ───────────────────────────────────────────────────────────

DEFAULT_TARGET_FPS = 1.0       # frames per second to sample from the video
YOLO_CONF_THRESHOLD = 0.3      # minimum YOLO detection confidence to keep a box
ROW_Y_TOLERANCE = 20           # pixel tolerance for grouping detections into the same row
MAX_EDIT_DISTANCE = 2          # max Levenshtein distance to snap a noisy station ID to a known one
TEMPORAL_DEDUP_WINDOW_SEC = 5.0  # identical consecutive rows within this window are collapsed

OCR_LANGUAGES = ["en"]

# ── Class taxonomy ────────────────────────────────────────────────────────────

# These two classes are detected by YOLO but intentionally not OCR'd:
#   - station_name  → derived from STATION_LOOKUP keyed on station_id
#   - station_numbering → not needed once station_id is known (see project plan §2)
SKIPPED_CLASSES = frozenset({"station_name", "station_numbering"})

# ── Station lookup ────────────────────────────────────────────────────────────
#
# Maps station_id (after OCR cleaning) → human-readable station name.
# Used instead of reading station_name text via OCR (see project plan §2 for rationale).
#
# NOTE: BD103 is ambiguous — it could be 'Pungkur' or 'Asia Afrika' depending on
# station_numbering, which is no longer read. Defaulted to 'Pungkur'; update if wrong.

STATION_LOOKUP: dict[str, str] = {
    "BD093": "Dayeuh Kolot",
    "BD095": "Bojongasih",
    "BD096": "Pasawahan",
    "BD097": "Mengger",
    "BD098": "Wates",
    "BD099": "Mekar Agung",
    "BD100": "Moch. Toha",
    "BD101": "Pasar Montana",
    "BD102": "Tegal Lega",
    "BD103": "Pungkur",
    "BD105": "Banceuy",
    "BD106": "Bandung",
    "BD107": "Pasir Kaliki",
    "BD108": "Hasan Sadikin",
    "BD109": "Sukabungah",
    "BD110": "PVJ",
    "BD112": "Geger Kalong",
    "BD113": "UPI",
    "BD114": "Terminal Ledeng",
    "BD115": "Isola",
    "BD116": "Isolakahuripan",
    "BD117": "Gudangkahuripan",
    "BD118": "Kolonel Masturi",
    "BD119": "Manoko",
    "BD120": "Lembang",
}
