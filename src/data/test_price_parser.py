
from datetime import datetime, timezone
from pathlib import Path
import sys
import requests
import truststore

truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from download_entsoe_prices import parse_price_document

from dotenv import load_dotenv
import os

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("ENTSOE_API_KEY")

if not API_KEY:
    raise RuntimeError("Brak ENTSOE_API_KEY w pliku .env")

params = {
    "documentType": "A44",
    "in_Domain": "10YPL-AREA-----S",
    "out_Domain": "10YPL-AREA-----S",
    "periodStart": "202401040000",
    "periodEnd": "202401050000",
}

response = requests.get(
    "https://web-api.tp.entsoe.eu/api",
    params=params,
    headers={"SECURITY_TOKEN": API_KEY},
    timeout=60,
)
response.raise_for_status()

records = parse_price_document(response.text)

df = __import__("pandas").DataFrame(records)

print("Liczba rekordów po parsowaniu:", len(df))
print("\nLiczba rekordów według rozdzielczości:")
print(df["resolution"].value_counts())

print("\nRekordy dla pierwszego okresu:")
start = datetime(2024, 1, 3, 23, tzinfo=timezone.utc)
end = datetime(2024, 1, 4, 23, tzinfo=timezone.utc)

first_period = df[
    (df["timestamp_utc"] >= start)
    & (df["timestamp_utc"] < end)
].sort_values("timestamp_utc")

print(first_period[["timestamp_utc", "price_eur_mwh", "currency"]].to_string(index=False))
print("Liczba rekordów w pierwszym okresie:", len(first_period))