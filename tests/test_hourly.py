import pandas as pd

from src.data.build_hourly import aggregate_hourly


def _raw(times, values, resolution):
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(times),
        "value": values,
        "resolution": resolution,
    })


def test_quarter_hours_are_averaged():
    raw = _raw(["2025-10-01 00:00", "2025-10-01 00:15", "2025-10-01 00:30", "2025-10-01 00:45"],
               [10, 20, 30, 40], "PT15M")
    out = aggregate_hourly(raw)
    assert len(out) == 1 and out.loc[0, "value"] == 25


def test_incomplete_hour_is_nan():
    raw = _raw(["2025-10-01 00:00", "2025-10-01 00:15", "2025-10-01 00:30"], [10, 20, 30], "PT15M")
    assert pd.isna(aggregate_hourly(raw).loc[0, "value"])


def test_hourly_series_passes_through():
    raw = _raw(["2024-01-01 00:00", "2024-01-01 01:00"], [5, 7], "PT60M")
    assert list(aggregate_hourly(raw)["value"]) == [5, 7]