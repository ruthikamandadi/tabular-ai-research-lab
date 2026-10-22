"""Datetime parsing helpers shared by profiling and preprocessing."""

import pandas as pd


COMMON_DATETIME_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y",
    "%m/%d/%Y",
)


def parse_datetime(values: pd.Series) -> pd.Series:
    """Parse common uniform formats first, then use pandas mixed-format parsing."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    non_null = values.notna()
    for date_format in COMMON_DATETIME_FORMATS:
        parsed = pd.to_datetime(values, format=date_format, errors="coerce")
        if parsed[non_null].notna().all():
            return parsed

    return pd.to_datetime(values, format="mixed", errors="coerce")
