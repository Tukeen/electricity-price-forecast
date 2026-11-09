import pandas as pd

from src.features.calendar import is_holiday


def test_known_polish_holidays():
    days = pd.to_datetime(["2024-01-01", "2024-01-06", "2024-04-01", "2024-11-01", "2024-12-25", "2024-05-02"])
    assert list(is_holiday(days)) == [1, 1, 1, 1, 1, 0]


def test_wigilia_is_holiday_from_2025():
    # od 2025 Wigilia jest dniem wolnym; jeśli test nie przechodzi: pip install -U holidays
    assert is_holiday(pd.to_datetime(["2025-12-24"]))[0] == 1.0