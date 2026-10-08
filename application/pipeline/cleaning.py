"""
Per-class text cleaners: raw OCR strings → typed, validated values.

Each function returns None (not an empty string) when the value is unreadable,
so callers can distinguish "nothing was detected" from "something was read but is zero".

Public functions:
  clean_time(raw)                               → "HH:MM" | "HH:MM:SS" | None
  clean_station_id(raw, station_lookup, max_d)  → "BD106" | None
  clean_passenger_count(raw)                    → int | None
"""

import re


# ── Public cleaners ───────────────────────────────────────────────────────────


def clean_time(raw: str) -> str | None:
    """Parses OCR time output into HH:MM or HH:MM:SS.

    The in-game clock is read by EasyOCR in the visual order: seconds, hours, minutes
    (the layout of the UI's clock widget), so 3-group parsing assigns them in that order.
    Returns None if no recognisable digit groups are found."""
    if not raw:
        return None
    groups = re.findall(r"\d+", raw)
    if len(groups) == 2:
        h, m = int(groups[0]), int(groups[1])
        return f"{h:02d}:{m:02d}"
    if len(groups) >= 3:
        try:
            s = int(groups[0]) % 60
            h = int(groups[1]) % 24
            m = int(groups[2]) % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
        except ValueError:
            return None
    return None


def clean_station_id(raw: str, station_lookup: dict[str, str], max_edit_distance: int) -> str | None:
    """Cleans a noisy OCR'd station code and snaps it to the nearest known station ID.

    Steps:
      1. Strip non-alphanumeric characters (OCR often adds stray brackets, spaces)
      2. Correct common digit substitutions in the numeric suffix (B→8, O→0, S→5)
      3. Force the 'BD' prefix (all stations in this network use it)
      4. Find the closest match in station_lookup using Levenshtein edit distance

    Returns None if:
      - The text has no numeric part after cleaning
      - No known station is within max_edit_distance edits
      - Two stations are tied (ambiguous read is worse than a missing value)"""
    if not raw:
        return None

    stripped = re.sub(r"[^A-Za-z0-9]", "", raw)
    match = re.match(r"^[A-Za-z]*(.*)$", stripped)
    if not match:
        return None

    numeric_suffix = match.group(1)
    numeric_suffix = numeric_suffix.replace("B", "8").replace("O", "0").replace("S", "5")

    digit_match = re.search(r"\d+", numeric_suffix)
    if not digit_match:
        return None

    candidate = "BD" + digit_match.group()
    return _snap_to_known_station(candidate, station_lookup, max_edit_distance)


def clean_passenger_count(raw: str) -> int | None:
    """Strips non-numeric characters and casts to int. Returns None if nothing numeric remains."""
    if not raw:
        return None
    digits = re.sub(r"[^0-9]", "", raw)
    return int(digits) if digits else None


# ── Internal helpers ──────────────────────────────────────────────────────────


def _snap_to_known_station(
    candidate: str, station_lookup: dict[str, str], max_edit_distance: int
) -> str | None:
    """Returns the single closest station ID within max_edit_distance, or None if ambiguous/too far."""
    best_dist = max_edit_distance + 1
    best_ids: list[str] = []

    for known_id in station_lookup:
        dist = _levenshtein(candidate, known_id)
        if dist < best_dist:
            best_dist = dist
            best_ids = [known_id]
        elif dist == best_dist:
            best_ids.append(known_id)

    if best_dist > max_edit_distance or len(best_ids) != 1:
        return None
    return best_ids[0]


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance (insertions, deletions, substitutions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(
                prev[j] + 1,           # deletion
                curr[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]
