
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_utc"])
df = df.sort_values("timestamp_utc").reset_index(drop=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

print("=== 1. DUPLIKATY ZNACZNIKÓW CZASU ===")

duplicate_groups = (
    df.groupby("timestamp_utc")
    .agg(
        liczba=("timestamp_utc", "size"),
        rozne_ceny=("price_eur_mwh", "nunique"),
        rozne_waluty=("currency", "nunique"),
        rozne_rozdzielczosci=("resolution", "nunique"),
    )
)

duplicates = duplicate_groups[duplicate_groups["liczba"] > 1]

print("Liczba grup duplikatów:", len(duplicates))
print("Nadmiarowe wiersze:", int((duplicates["liczba"] - 1).sum()))
print("Grupy z różnymi cenami:", int((duplicates["rozne_ceny"] > 1).sum()))
print("Grupy z różnymi walutami:", int((duplicates["rozne_waluty"] > 1).sum()))
print(
    "Grupy z różnymi rozdzielczościami:",
    int((duplicates["rozne_rozdzielczosci"] > 1).sum()),
)

print("\nPrzykłady duplikatów z różnicami:")
conflicts = duplicates[
    (duplicates["rozne_ceny"] > 1)
    | (duplicates["rozne_waluty"] > 1)
    | (duplicates["rozne_rozdzielczosci"] > 1)
]
print(conflicts.head(20).to_string())

print("\n=== 2. ODSTĘPY MIĘDZY KOLEJNYMI ZNACZNIKAMI CZASU ===")

# Do badania ciągłości wykorzystujemy unikalne znaczniki czasu.
# Nie usuwamy niczego z pliku źródłowego.
unique_times = (
    df.drop_duplicates("timestamp_utc")
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)

unique_times["next_timestamp"] = unique_times["timestamp_utc"].shift(-1)
unique_times["delta"] = (
    unique_times["next_timestamp"] - unique_times["timestamp_utc"]
)

print("\nNajczęstsze odstępy:")
print(unique_times["delta"].value_counts().head(15).to_string())

print("\nOdstępy dłuższe niż 60 minut:")
long_gaps = unique_times[unique_times["delta"] > pd.Timedelta("60min")]

print("Liczba takich luk:", len(long_gaps))
print(
    long_gaps[
        ["timestamp_utc", "next_timestamp", "delta", "resolution"]
    ].head(30).to_string(index=False)
)

print("\n=== 3. KONTROLA ZMIAN ROZDZIELCZOŚCI ===")

resolution_changes = unique_times[
    unique_times["resolution"] != unique_times["resolution"].shift()
]

print("Liczba zmian rozdzielczości:", len(resolution_changes))
print(
    resolution_changes[
        ["timestamp_utc", "resolution", "currency"]
    ].head(30).to_string(index=False)
)

print("\n=== 4. PODSUMOWANIE ===")
print("Unikalne znaczniki czasu:", len(unique_times))
print("Zakres:", unique_times["timestamp_utc"].min(), "->",
      unique_times["timestamp_utc"].max())
print("Liczba luk > 60 minut:", len(long_gaps))