"""
Four-pass DataFrame cleaning pipeline applied after all frames are processed.

run_postprocessing() is the only public function — it calls the four passes in order.
Each pass is a private function so it can be read and reasoned about independently.

Pass order matters:
  1. drop_same_timestamp_duplicates — collapse partial echo rows before dedup
  2. drop_missing_rows              — remove truly empty rows
  3. temporal_deduplication         — remove consecutive identical rows
  4. impute_missing_station_names   — fill derived fields last, after dedup
"""

import pandas as pd


def run_postprocessing(
    df: pd.DataFrame,
    station_lookup: dict[str, str],
    temporal_dedup_window_sec: float,
) -> pd.DataFrame:
    """Applies all four cleaning passes in order. Returns a cleaned DataFrame."""
    if df.empty:
        return df
    df = _drop_same_timestamp_duplicates(df)
    df = _drop_missing_rows(df)
    df = _temporal_deduplication(df, temporal_dedup_window_sec)
    df = _impute_missing_station_names(df, station_lookup)
    return df


def _drop_same_timestamp_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the most data-complete row for each frame timestamp.

    Within a single sampled frame, y-band grouping sometimes produces one valid
    row plus a partial echo (e.g. a misread badge fragment next to a real entry).
    This pass keeps the row with the most non-null key fields and discards the rest."""
    key_cols = [c for c in ["passenger_increse", "station_id", "station_name"] if c in df.columns]
    completeness = df[key_cols].notna().sum(axis=1)
    ranked = df.assign(_rank=completeness).sort_values(
        ["timestamp", "_rank"], ascending=[True, False], kind="mergesort"
    )
    return (
        ranked.drop_duplicates(subset="timestamp", keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def _drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where every key OCR field is null (nothing was read from the frame)."""
    key_cols = [c for c in ["passenger_increse", "station_id"] if c in df.columns]
    return df.dropna(subset=key_cols, how="all").reset_index(drop=True)


def _temporal_deduplication(df: pd.DataFrame, max_gap_seconds: float) -> pd.DataFrame:
    """Removes consecutive rows with identical station_id + passenger_increse within the time window.

    NaN equality: two missing values in the same column are treated as equal
    (both filled with '__NULL__') so partially-read rows don't escape the filter."""
    key_cols = [c for c in ["station_id", "passenger_increse"] if c in df.columns]
    if not key_cols:
        return df

    filled = df[key_cols].fillna("__NULL__")
    is_same = (filled == filled.shift(1)).all(axis=1)
    time_gap = df["timestamp"] - df["timestamp"].shift(1)
    is_duplicate = is_same & (time_gap <= max_gap_seconds)

    return df[~is_duplicate].reset_index(drop=True)


def _impute_missing_station_names(df: pd.DataFrame, station_lookup: dict[str, str]) -> pd.DataFrame:
    """Re-derives station_name from station_id for any rows still missing it.

    Null station_id cells are labelled 'OCR MISSREAD' rather than left blank
    so the CSV is unambiguous — a blank cell would look like missing data."""
    if "station_id" not in df.columns or "station_name" not in df.columns:
        return df

    df["station_name"] = df["station_id"].map(
        lambda sid: station_lookup.get(sid) if pd.notna(sid) else "UNKNOWN ID"
    )
    df["station_id"] = df["station_id"].fillna("OCR MISSREAD")
    return df
