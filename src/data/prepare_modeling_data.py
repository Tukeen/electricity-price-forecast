
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "poland_day_ahead_prices.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_prices_2020_2024_hourly.csv"
)

df = pd.read_csv(INPUT_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

# Wybieramy pełne lata 2020–2024, ceny w EUR i rozdzielczość godzinową.
df = df[
    (df["timestamp_utc"] >= "2020-01-01T00:00:00Z")
    & (df["timestamp_utc"] < "2025-01-01T00:00:00Z")
    & (df["currency"] == "EUR")
    & (df["resolution"] == "PT60M")
].copy()

if df.empty:
    raise ValueError("Po filtrowaniu nie pozostały żadne obserwacje.")

# Przed usunięciem duplikatów sprawdzamy zgodność cen i metadanych.
duplicate_rows = df[df.duplicated("timestamp_utc", keep=False)]

if not duplicate_rows.empty:
    consistency = duplicate_rows.groupby("timestamp_utc").agg(
        prices=("price_eur_mwh", "nunique"),
        currencies=("currency", "nunique"),
        resolutions=("resolution", "nunique"),
        units=("unit", "nunique"),
    )

    conflicts = consistency[
        (consistency["prices"] > 1)
        | (consistency["currencies"] > 1)
        | (consistency["resolutions"] > 1)
        | (consistency["units"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            "Wykryto sprzeczne duplikaty. Przerwano przygotowanie danych:\n"
            f"{conflicts.head(10)}"
        )

# Usuwamy wyłącznie powtórzenia identycznych znaczników czasu
# po sprawdzeniu zgodności kluczowych pól.
df = (
    df.sort_values("timestamp_utc")
    .drop_duplicates("timestamp_utc", keep="first")
    .reset_index(drop=True)
)

# Sprawdzamy kompletność godzin w całym okresie.
expected = pd.date_range(
    start="2020-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    inclusive="left",
    freq="h",
)

actual = pd.DatetimeIndex(df["timestamp_utc"])
missing = expected.difference(actual)
unexpected = actual.difference(expected)

print("=== KONTROLA ZBIORU DO MODELOWANIA ===")
print("Liczba obserwacji:", len(df))
print("Pierwsza obserwacja:", df["timestamp_utc"].min())
print("Ostatnia obserwacja:", df["timestamp_utc"].max())
print("Liczba duplikatów:", df["timestamp_utc"].duplicated().sum())
print("Liczba brakujących godzin:", len(missing))
print("Liczba nieoczekiwanych znaczników:", len(unexpected))
print("Waluty:", df["currency"].value_counts().to_dict())
print("Rozdzielczości:", df["resolution"].value_counts().to_dict())
print("Brakujące ceny:", df["price_eur_mwh"].isna().sum())

if len(df) != len(expected) or len(missing) > 0 or len(unexpected) > 0:
    print("\nPrzykładowe brakujące godziny:", list(missing[:10]))
    print("Przykładowe nieoczekiwane znaczniki:", list(unexpected[:10]))
    raise ValueError(
        "Zbiór nie jest kompletnym szeregiem godzinowym. "
        "Nie zapisano pliku wyjściowego."
    )

if df["price_eur_mwh"].isna().any():
    raise ValueError("Występują brakujące ceny.")

df.to_csv(OUTPUT_PATH, index=False)

print("\nZapisano:", OUTPUT_PATH)
print("Zbiór przeszedł podstawowe kontrole jakości.")