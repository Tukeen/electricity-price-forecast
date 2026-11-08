"""Pobieranie danych ENTSO-E i budowa tabeli entsoe_raw w DuckDB.

Uruchomienie:  python -m src.data.ingest [--no-download]
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd
import requests

from src.data.entsoe_client import fetch_xml
from src.data.entsoe_parser import ack_message, parse_document, read_root
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EIC = CFG["market"]["eic"]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "entsoe"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "electricity.duckdb"

DATASETS = {
    "prices": {  # ceny day-ahead
        "params": {"documentType": "A44", "in_Domain": EIC, "out_Domain": EIC},
        "key": "price",
    },
    "load_forecast": {  # prognoza zapotrzebowania na dobę następną
        "params": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": EIC},
        "key": "load_forecast",
    },
    "res_forecast": {  # prognoza wiatru i PV (psrType B16 = PV, B18/B19 = wiatr)
        "params": {"documentType": "A69", "processType": "A01", "in_Domain": EIC},
        "key": "res_forecast",
    },
}


def download_year(session, name, spec, year, today, forward_days):
    path = RAW_DIR / name / f"{name}_{year}.xml"

    # Lata zakończone pobieramy raz; rok bieżący odświeżamy przy każdym uruchomieniu.
    if path.exists() and year < today.year:
        return

    start = date(year, 1, 1)
    end = min(date(year + 1, 1, 1), today + timedelta(days=forward_days))
    if start >= end:
        return

    params = {
        **spec["params"],
        "periodStart": start.strftime("%Y%m%d") + "0000",
        "periodEnd": end.strftime("%Y%m%d") + "0000",
    }
    content = fetch_xml(session, params)
    if content is None:
        print(f"[{name} {year}] nie udało się pobrać")
        return

    root = read_root(content)
    if root.tag == "Acknowledgement_MarketDocument":
        print(f"[{name} {year}] brak danych: {ack_message(root)}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f"[{name} {year}] zapisano {path.name}")


def build_database() -> None:
    rows = []
    for name, spec in DATASETS.items():
        for path in sorted((RAW_DIR / name).glob(f"{name}_*.xml")):
            for record in parse_document(path.read_bytes(), spec["key"]):
                record["dataset"] = name
                rows.append(record)

    df = pd.DataFrame(rows, columns=[
        "dataset", "series_key", "timestamp_utc", "value",
        "resolution", "unit", "currency", "curve_type",
    ])
    # DuckDB przechowuje znaczniki jako czas UTC bez strefy (TIMESTAMP).
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)

    # Duplikaty na granicach dób/lat: zostaje wpis z nowszego pliku.
    key = ["dataset", "series_key", "timestamp_utc"]
    conflicts = int((df.groupby(key)["value"].nunique() > 1).sum())
    if conflicts:
        print(f"UWAGA: {conflicts} znaczników z różnymi wartościami w duplikatach")
    df = df.sort_values(key, kind="stable").drop_duplicates(key, keep="last")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.register("raw_df", df)
    con.execute("CREATE OR REPLACE TABLE entsoe_raw AS SELECT * FROM raw_df")

    print("\n=== entsoe_raw ===")
    print(con.sql("""
        SELECT dataset, series_key, resolution, count(*) AS n,
               min(timestamp_utc) AS first_ts, max(timestamp_utc) AS last_ts
        FROM entsoe_raw GROUP BY ALL ORDER BY 1, 2, 3
    """).df().to_string(index=False))
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true", help="tylko przebuduj bazę z plików XML")
    args = parser.parse_args()

    if not args.no_download:
        today = datetime.now(timezone.utc).date()
        with requests.Session() as session:
            for name, spec in DATASETS.items():
                print(f"== {name} ==")
                for year in range(CFG["ingest"]["from_year"], today.year + 1):
                    download_year(session, name, spec, year, today, CFG["ingest"]["forward_days"])
                    time.sleep(1)

    build_database()


if __name__ == "__main__":
    main()