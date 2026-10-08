"""
Passenger data aggregation for the analytics section.

Three public functions:
  aggregate_by_station(df)                     -> station totals + % share
  aggregate_by_time_interval(df, interval_min) -> per-bucket station breakdown + % share
  aggregate_time_series(df)                    -> per-event time series by destination station

All functions silently exclude 'OCR MISSREAD' rows — those have no known station
and would pollute the charts with an 'unknown' segment.
"""

import pandas as pd


def aggregate_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """Total passenger increases per station across the whole session.

    Returns columns: station_name, total_passengers, pct_of_total
    Sorted by total_passengers descending."""
    empty = pd.DataFrame(columns=["station_name", "total_passengers", "pct_of_total"])

    if df.empty or "passenger_increse" not in df.columns:
        return empty

    valid = df[
        df["station_id"].notna()
        & (df["station_id"] != "OCR MISSREAD")
        & df["passenger_increse"].notna()
    ].copy()

    if valid.empty:
        return empty

    totals = (
        valid.groupby("station_name")["passenger_increse"]
        .sum()
        .reset_index()
        .rename(columns={"passenger_increse": "total_passengers"})
    )
    grand_total = totals["total_passengers"].sum()
    totals["pct_of_total"] = (totals["total_passengers"] / grand_total * 100).round(1)

    return totals.sort_values("total_passengers", ascending=False).reset_index(drop=True)


def aggregate_by_time_interval(df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """Passenger increases per station within equal time buckets.

    Buckets are formed by flooring the in-game clock to the nearest interval_minutes.
    E.g. interval=5 → '10:40', '10:45', '10:50' ...

    Returns columns: time_bucket, station_name, passengers, pct_of_bucket
    Sorted by time_bucket ascending, then passengers descending."""
    empty = pd.DataFrame(columns=["time_bucket", "station_name", "passengers", "pct_of_bucket"])

    if df.empty or "time" not in df.columns or "passenger_increse" not in df.columns:
        return empty

    valid = df[
        df["station_id"].notna()
        & (df["station_id"] != "OCR MISSREAD")
        & df["time"].notna()
        & df["passenger_increse"].notna()
    ].copy()

    if valid.empty:
        return empty

    valid["_minutes"] = valid["time"].map(_parse_time_to_minutes)
    valid = valid.dropna(subset=["_minutes"])

    if valid.empty:
        return empty

    valid["time_bucket"] = valid["_minutes"].map(
        lambda m: _format_bucket(int(m), interval_minutes)
    )

    grouped = (
        valid.groupby(["time_bucket", "station_name"])["passenger_increse"]
        .sum()
        .reset_index()
        .rename(columns={"passenger_increse": "passengers"})
    )

    bucket_totals = grouped.groupby("time_bucket")["passengers"].transform("sum")
    grouped["pct_of_bucket"] = (grouped["passengers"] / bucket_totals * 100).round(1)

    return grouped.sort_values(
        ["time_bucket", "passengers"], ascending=[True, False]
    ).reset_index(drop=True)


def _parse_time_to_minutes(time_str: str) -> float | None:
    """Converts 'HH:MM' or 'HH:MM:SS' to minutes since midnight."""
    try:
        parts = str(time_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (IndexError, ValueError):
        return None


def _format_bucket(minutes: int, interval: int) -> str:
    """Floors minutes to the nearest interval and formats as 'HH:MM'."""
    bucketed = (minutes // interval) * interval
    return f"{bucketed // 60:02d}:{bucketed % 60:02d}"


def aggregate_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Passenger increases plotted against the in-game clock, grouped by destination station.

    Returns columns: time, station_name, passenger_increse
    One row per (time, station_name) pair. Sorted by time ascending.
    Used for the passenger flow timeline chart."""
    empty = pd.DataFrame(columns=["time", "station_name", "passenger_increse"])

    if df.empty or "time" not in df.columns or "passenger_increse" not in df.columns:
        return empty

    valid = df[
        df["station_id"].notna()
        & (df["station_id"] != "OCR MISSREAD")
        & df["time"].notna()
        & df["passenger_increse"].notna()
    ].copy()

    if valid.empty:
        return empty

    grouped = (
        valid.groupby(["time", "station_name"])["passenger_increse"]
        .sum()
        .reset_index()
    )

    return grouped.sort_values("time").reset_index(drop=True)


def aggregate_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table for the crowd heatmap: station rows × HH:MM:SS time columns.

    Each cell holds the total passenger_increse at that (station, time) intersection.
    Rows are sorted by total passengers descending (busiest station first), columns
    sorted chronologically. Empty cells are filled with 0."""
    empty = pd.DataFrame()

    if df.empty or "time" not in df.columns or "passenger_increse" not in df.columns:
        return empty

    valid = df[
        df["station_id"].notna()
        & (df["station_id"] != "OCR MISSREAD")
        & df["time"].notna()
        & df["passenger_increse"].notna()
    ].copy()

    if valid.empty:
        return empty

    pivot = valid.pivot_table(
        index="station_name",
        columns="time",
        values="passenger_increse",
        aggfunc="sum",
        fill_value=0,
    )

    row_totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.reindex(row_totals.index)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    return pivot
