
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "data" / "raw" / "entsoe_prices" / "prices_2015.xml"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_entsoe_prices import parse_price_document


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def child_text(element, name):
    for child in element:
        if local_name(child.tag) == name:
            return child.text
    return None


# 1. Surowy XML: zakresy okresów z początku historii
root = ET.parse(XML_PATH).getroot()

print("=== OKRESY W SUROWYM XML ===")

for ts_index, ts in enumerate(
    (el for el in root.iter() if local_name(el.tag) == "TimeSeries"),
    start=1,
):
    for period in (
        el for el in ts.iter() if local_name(el.tag) == "Period"
    ):
        interval = next(
            (el for el in period if local_name(el.tag) == "timeInterval"),
            None,
        )
        if interval is None:
            continue

        start = child_text(interval, "start")
        end = child_text(interval, "end")

        if not start or not end:
            continue

        if not ("2015-01-04" <= start[:10] <= "2015-01-07"):
            continue

        resolution = child_text(period, "resolution")
        points = [
            el for el in period.iter()
            if local_name(el.tag) == "Point"
        ]

        print(
            f"TimeSeries #{ts_index}: "
            f"{start} -> {end}, "
            f"resolution={resolution}, "
            f"jawne punkty={len(points)}"
        )

# 2. Porównanie z rekordami utworzonymi przez parser
print("\n=== REKORDY PO PARSOWANIU ===")

records = parse_price_document(XML_PATH.read_text(encoding="utf-8"))
parsed = pd.DataFrame(records)
parsed["timestamp_utc"] = pd.to_datetime(
    parsed["timestamp_utc"], utc=True
)

start = pd.Timestamp("2015-01-04T00:00:00Z")
end = pd.Timestamp("2015-01-08T00:00:00Z")

subset = parsed[
    (parsed["timestamp_utc"] >= start)
    & (parsed["timestamp_utc"] < end)
].sort_values("timestamp_utc")

print("Liczba rekordów:", len(subset))
print(
    subset[
        ["timestamp_utc", "price_eur_mwh", "resolution", "currency"]
    ].to_string(index=False)
)

# 3. Odstępy w CSV w tym samym przedziale
print("\n=== ODSTĘPY W GOTOWYM CSV ===")

csv = pd.read_csv(CSV_PATH, parse_dates=["timestamp_utc"])
csv["timestamp_utc"] = pd.to_datetime(csv["timestamp_utc"], utc=True)

subset_csv = (
    csv[
        (csv["timestamp_utc"] >= start)
        & (csv["timestamp_utc"] < end)
    ]
    .drop_duplicates("timestamp_utc")
    .sort_values("timestamp_utc")
)

subset_csv["nastepny_znacznik"] = subset_csv["timestamp_utc"].shift(-1)
subset_csv["odstep"] = (
    subset_csv["nastepny_znacznik"] - subset_csv["timestamp_utc"]
)

print(
    subset_csv.loc[
        subset_csv["odstep"] > pd.Timedelta("60min"),
        ["timestamp_utc", "nastepny_znacznik", "odstep"],
    ].to_string(index=False)
)