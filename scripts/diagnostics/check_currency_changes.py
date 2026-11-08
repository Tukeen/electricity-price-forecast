
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

# Na potrzeby diagnostyki usuwamy wyłącznie identyczne znaczniki czasu.
df = (
    df.drop_duplicates("timestamp_utc")
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)

print("=== 1. WALUTA W PODZIALE NA MIESIĄCE ===")

df["month"] = df["timestamp_utc"].dt.strftime("%Y-%m")

monthly = pd.crosstab(df["month"], df["currency"])
print(monthly.to_string())

print("\n=== 2. PIERWSZE I OSTATNIE OBSERWACJE KAŻDEJ WALUTY ===")

for currency in sorted(df["currency"].dropna().unique()):
    subset = df[df["currency"] == currency]
    print(f"\nWaluta: {currency}")
    print("Pierwsza obserwacja:", subset["timestamp_utc"].min())
    print("Ostatnia obserwacja:", subset["timestamp_utc"].max())
    print("Liczba obserwacji:", len(subset))

print("\n=== 3. ZMIANY WALUTY W CHRONOLOGII ===")

df["previous_currency"] = df["currency"].shift()
changes = df[
    df["currency"] != df["previous_currency"]
].copy()

print(
    changes[
        ["timestamp_utc", "previous_currency", "currency", "resolution"]
    ].to_string(index=False)
)

print("\n=== 4. ZMIANY WALUTY W OKOLICACH GRANIC LAT ===")

for year in [2016, 2017, 2018, 2019, 2020]:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    subset = df[
        (df["timestamp_utc"] >= start)
        & (df["timestamp_utc"] < end)
    ]

    print(f"\nRok {year}:")
    print(
        subset.groupby("currency").agg(
            liczba_obserwacji=("timestamp_utc", "size"),
            pierwsza=("timestamp_utc", "min"),
            ostatnia=("timestamp_utc", "max"),
        ).to_string()
    )