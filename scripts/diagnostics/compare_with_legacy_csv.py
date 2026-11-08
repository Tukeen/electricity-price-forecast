import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

legacy = pd.read_csv(ROOT / "data" / "processed" / "poland_day_ahead_prices.csv")
legacy["timestamp_utc"] = pd.to_datetime(legacy["timestamp_utc"], utc=True).dt.tz_localize(None)
legacy = legacy.drop_duplicates("timestamp_utc").set_index("timestamp_utc")["price_eur_mwh"]

con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)
new = (con.sql("SELECT timestamp_utc, value FROM entsoe_raw WHERE dataset = 'prices'")
       .df().set_index("timestamp_utc")["value"])

common = legacy.index.intersection(new.index)
print("stary CSV:", len(legacy), "| nowa baza:", len(new), "| wspólne:", len(common))
print("tylko w starym:", len(legacy.index.difference(new.index)))
print("tylko w nowej:", len(new.index.difference(legacy.index)))
print("max |różnica|:", (legacy[common] - new[common]).abs().max())