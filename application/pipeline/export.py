"""
Serialises the cleaned pipeline DataFrame to CSV text.

to_csv() is the only public function. The Gradio download button and the
REST API both call it, so the two surfaces always return the same file.
"""

import pandas as pd

import application.config as config


def to_csv(df: pd.DataFrame) -> str:
    """Returns the DataFrame as CSV text with a fixed column order.

    passenger_increse is written as an integer: pandas stores the column as
    float once any row is missing a count, which would otherwise export "2.0".
    timestamp is rounded to milliseconds — frame offsets carry float noise."""
    out = df.reindex(columns=config.OUTPUT_COLUMNS).copy()
    out["passenger_increse"] = pd.to_numeric(out["passenger_increse"], errors="coerce").astype("Int64")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").round(3)
    return out.to_csv(index=False)
