"""
Groups per-frame OCR detections into logical table rows by vertical position.

YOLO returns a flat list of independent boxes per frame — it has no concept of
"row". This module builds that structure by clustering boxes whose y_center values
fall within ROW_Y_TOLERANCE pixels of each other (see project plan §15, Approach A).

assemble_rows() is the only public function.
"""

from .detection import Box


def assemble_rows(
    boxes: list[Box],
    ocr_results: dict[int, object],
    timestamp: float,
    station_lookup: dict[str, str],
    row_y_tolerance: int,
) -> list[dict]:
    """Groups detected boxes into logical rows and builds the output row dicts.

    ocr_results maps box index → cleaned value (str, int, or None).

    Returns a list of dicts, each with:
      timestamp, time, station_id, station_name, passenger_increse

    Time is extracted globally (it's a HUD element common to the whole frame,
    not part of any specific data row). Station name is derived from the lookup
    table — it is never read from OCR."""

    # Pair each box with its cleaned OCR value
    tagged = [
        (box.y_center, box.cls_name, ocr_results.get(i))
        for i, box in enumerate(boxes)
    ]

    # Extract the frame-level time before row grouping
    frame_time = next(
        (value for _, cls, value in tagged if cls == "time" and value),
        None,
    )

    # Only table-data classes participate in y-band grouping
    table_items = sorted(
        [(y, cls, value) for y, cls, value in tagged if cls != "time"],
        key=lambda d: d[0],
    )

    # Cluster by y_center proximity
    row_groups: list[dict] = []
    current_row: dict = {}
    current_y: float | None = None

    for y, cls_name, value in table_items:
        if current_y is not None and abs(y - current_y) > row_y_tolerance:
            row_groups.append(current_row)
            current_row = {}
        current_row[cls_name] = value
        current_y = y

    if current_row:
        row_groups.append(current_row)

    # Build final row dicts with all output columns
    rows = []
    for group in row_groups:
        station_id = group.get("station_id")
        rows.append({
            "timestamp": timestamp,
            "time": frame_time,
            "station_id": station_id,
            "station_name": station_lookup.get(station_id) if station_id else None,
            "passenger_increse": group.get("passenger_increse"),
        })

    return rows
