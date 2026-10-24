
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

print("\n=== 1. PODSTAWOWE INFORMACJE ===")
print("Liczba wierszy:", len(df))
print("Zakres:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
print("Unikalne znaczniki czasu:", df["timestamp_utc"].nunique())
print("Duplikaty znaczników czasu:", df["timestamp_utc"].duplicated().sum())

print("\n=== 2. DUPLIKATY — PRZYKŁADY ===")
duplicates = df[
    df.duplicated(subset=["timestamp_utc"], keep=False)
].sort_values("timestamp_utc")

print(duplicates.head(30).to_string(index=False))

print("\n=== 3. WALUTA W PODZIALE NA ROK ===")
df["year"] = df["timestamp_utc"].dt.year

currency_by_year = pd.crosstab(
    df["year"], df["currency"], margins=False
)
print(currency_by_year.to_string())

print("\n=== 4. ROZDZIELCZOŚĆ W PODZIALE NA ROK ===")
resolution_by_year = pd.crosstab(
    df["year"], df["resolution"], margins=False
)
print(resolution_by_year.to_string())

print("\n=== 5. LICZBA OBSERWACJI NA DZIEŃ ===")
df["date_utc"] = df["timestamp_utc"].dt.floor("D")

daily = df.groupby("date_utc").size()

print("Dni z co najmniej jedną obserwacją:", len(daily))
print("Dni z mniej niż 24 obserwacjami:", (daily < 24).sum())
print("Dni z ponad 96 obserwacjami:", (daily > 96).sum())

print("\nNajmniej obserwacji w dniu:")
print(daily.sort_values().head(15).to_string())

print("\n=== 6. JEDNOSTKI CEN ===")
print(df.groupby(["currency", "unit"], dropna=False).size().to_string())

print("\n=== KONIEC DIAGNOSTYKI ===")