
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "entsoe_prices"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def inspect_xml(path):
    root = ET.parse(path).getroot()

    print(f"\n{'=' * 60}")
    print(path.name)
    print("Typ dokumentu:", local_name(root.tag))

    series_list = [
        el for el in root.iter()
        if local_name(el.tag) == "TimeSeries"
    ]
    print("Liczba TimeSeries:", len(series_list))

    for i, series in enumerate(series_list, start=1):
        values = {}
        for el in series.iter():
            name = local_name(el.tag)
            if name in {
                "currency_Unit.name",
                "price_Measure_Unit.name",
                "businessType",
                "curveType",
            }:
                values.setdefault(name, set()).add(el.text)

        periods = [
            el for el in series
            if local_name(el.tag) == "Period"
        ]

        point_count = sum(
            1 for period in periods
            for el in period.iter()
            if local_name(el.tag) == "Point"
        )

        starts = []
        resolutions = []

        for period in periods:
            for el in period.iter():
                name = local_name(el.tag)
                if name == "start" and el.text:
                    starts.append(el.text)
                elif name == "resolution" and el.text:
                    resolutions.append(el.text)

        print(f"\nTimeSeries #{i}")
        print("  Metadane:", {k: sorted(v) for k, v in values.items()})
        print("  Liczba okresów:", len(periods))
        print("  Liczba punktów:", point_count)
        print("  Początek pierwszego okresu:", min(starts) if starts else None)
        print("  Początek ostatniego okresu:", max(starts) if starts else None)
        print("  Rozdzielczości:", sorted(set(resolutions)))


for year in [2015, 2017, 2018, 2025, 2026]:
    path = RAW_DIR / f"prices_{year}.xml"
    if path.exists():
        inspect_xml(path)

print("\n" + "=" * 60)
print("DUPLIKATY W CSV")

df = pd.read_csv(CSV_PATH)
duplicate_rows = df[
    df.duplicated(subset=["timestamp_utc"], keep=False)
].copy()

if duplicate_rows.empty:
    print("Brak duplikatów.")
else:
    duplicate_summary = (
        duplicate_rows.groupby("timestamp_utc")
        .agg(
            liczba=("price_eur_mwh", "size"),
            rozne_ceny=("price_eur_mwh", "nunique"),
            rozne_waluty=("currency", "nunique"),
            rozne_rozdzielczosci=("resolution", "nunique"),
        )
        .reset_index()
    )

    print("Liczba powtarzających się znaczników czasu:",
          len(duplicate_summary))
    print("Duplikaty z różnymi cenami:",
          (duplicate_summary["rozne_ceny"] > 1).sum())
    print("Duplikaty z różnymi walutami:",
          (duplicate_summary["rozne_waluty"] > 1).sum())
    print("Duplikaty z różnymi rozdzielczościami:",
          (duplicate_summary["rozne_rozdzielczosci"] > 1).sum())

    print("\nPrzykładowe grupy duplikatów:")
    print(duplicate_summary.head(20).to_string(index=False))