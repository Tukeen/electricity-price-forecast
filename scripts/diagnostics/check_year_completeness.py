
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "poland_day_ahead_prices.csv"

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp_utc"])
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

# Duplikaty identycznych znaczników czasu pomijamy wyłącznie
# na potrzeby diagnostyki ciągłości szeregu.
df = df.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
df["year"] = df["timestamp_utc"].dt.year
df["date_utc"] = df["timestamp_utc"].dt.floor("D")

print("=== 1. OBSERWACJE W PODZIALE NA ROK I ROZDZIELCZOŚĆ ===")
print(pd.crosstab(df["year"], df["resolution"]).to_string())

print("\n=== 2. DNI Z NIEPEŁNYM PROFILEM ===")

daily = (
    df.groupby(["date_utc", "resolution"])
    .size()
    .rename("n")
    .reset_index()
)

# Dla dni z rozdzielczością godzinową oczekujemy zwykle 24 obserwacji.
# Dla dni kwadransowych — zwykle 96.
daily["oczekiwane"] = daily["resolution"].map({
    "PT60M": 24,
    "PT15M": 96,
})
daily["rok"] = daily["date_utc"].dt.year
daily["kompletny_wg_prostej_reguly"] = daily["n"] == daily["oczekiwane"]

print(
    daily.groupby(["rok", "resolution"]).agg(
        liczba_dni=("date_utc", "nunique"),
        dni_kompletne=("kompletny_wg_prostej_reguly", "sum"),
        dni_niekompletne=("kompletny_wg_prostej_reguly",
                          lambda x: (~x).sum()),
        min_obserwacji=("n", "min"),
        mediana_obserwacji=("n", "median"),
        max_obserwacji=("n", "max"),
    ).to_string()
)

print("\n=== 3. NAJWIĘKSZE LUKI W PODZIALE NA ROK ===")

df = df.sort_values("timestamp_utc").copy()
df["next_timestamp"] = df["timestamp_utc"].shift(-1)
df["delta"] = df["next_timestamp"] - df["timestamp_utc"]

gaps = df[df["delta"] > pd.Timedelta("60min")].copy()
gaps["rok"] = gaps["timestamp_utc"].dt.year

print(gaps.groupby("rok").size().rename("liczba_luk").to_string())

print("\nPrzykładowe luki z lat 2020–2026:")
print(
    gaps[gaps["rok"] >= 2020][
        ["timestamp_utc", "next_timestamp", "delta", "resolution"]
    ].head(25).to_string(index=False)
)

print("\n=== 4. WALUTA W PODZIALE NA ROK ===")
print(pd.crosstab(df["year"], df["currency"]).to_string())