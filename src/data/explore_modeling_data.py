
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_prices_2020_2024_hourly.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Ceny dotyczą polskiego rynku, dlatego analizujemy też czas lokalny.
df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("Europe/Warsaw")
df["hour_local"] = df["timestamp_local"].dt.hour
df["weekday_local"] = df["timestamp_local"].dt.day_of_week
df["month_local"] = df["timestamp_local"].dt.month
df["year"] = df["timestamp_utc"].dt.year

price = df["price_eur_mwh"]

print("=== 1. STATYSTYKI OPISOWE CEN ===")
print(price.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

print("\n=== 2. JAKOŚĆ I ZAKRES DANYCH ===")
print("Liczba obserwacji:", len(df))
print("Zakres UTC:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
print("Brakujące ceny:", price.isna().sum())
print("Liczba cen ujemnych:", (price < 0).sum())
print("Udział cen ujemnych (%):", round((price < 0).mean() * 100, 3))
print("Liczba cen równych zero:", (price == 0).sum())

print("\n=== 3. STATYSTYKI ROCZNE ===")
print(
    df.groupby("year")["price_eur_mwh"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
    .to_string()
)

print("\n=== 4. ŚREDNIA CENA WEDŁUG GODZINY LOKALNEJ ===")
print(
    df.groupby("hour_local")["price_eur_mwh"]
    .agg(["mean", "median", "std"])
    .round(2)
    .to_string()
)

print("\n=== 5. ŚREDNIA CENA WEDŁUG DNIA TYGODNIA ===")
weekday_names = {
    0: "Poniedziałek",
    1: "Wtorek",
    2: "Środa",
    3: "Czwartek",
    4: "Piątek",
    5: "Sobota",
    6: "Niedziela",
}

weekday_stats = (
    df.groupby("weekday_local")["price_eur_mwh"]
    .agg(["mean", "median", "std"])
    .rename(index=weekday_names)
)

print(weekday_stats.round(2).to_string())

print("\n=== 6. ŚREDNIE MIESIĘCZNE ===")
monthly = (
    df.set_index("timestamp_utc")["price_eur_mwh"]
    .resample("MS")
    .agg(["mean", "median"])
)

print(monthly.round(2).to_string())

# Wykres 1: pełny szereg cen
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["timestamp_utc"], price, linewidth=0.5)
ax.set_title("Godzinowe ceny energii elektrycznej w Polsce (2020–2024)")
ax.set_xlabel("Data (UTC)")
ax.set_ylabel("Cena (EUR/MWh)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(REPORTS_DIR / "eda_price_series.png", dpi=150)
plt.close(fig)

# Wykres 2: rozkład cen
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(price, bins=100)
ax.set_title("Rozkład godzinowych cen energii")
ax.set_xlabel("Cena (EUR/MWh)")
ax.set_ylabel("Liczba obserwacji")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(REPORTS_DIR / "eda_price_distribution.png", dpi=150)
plt.close(fig)

# Wykres 3: średnia cena według godziny lokalnej
hourly = df.groupby("hour_local")["price_eur_mwh"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hourly.index, hourly.values, marker="o")
ax.set_xticks(range(24))
ax.set_title("Średnia cena według godziny lokalnej w Polsce")
ax.set_xlabel("Godzina lokalna")
ax.set_ylabel("Średnia cena (EUR/MWh)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(REPORTS_DIR / "eda_hourly_pattern.png", dpi=150)
plt.close(fig)

# Wykres 4: średnia cena miesiąc po miesiącu
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly.index, monthly["mean"], marker="o", markersize=3)
ax.set_title("Średnia miesięczna cena energii (2020–2024)")
ax.set_xlabel("Miesiąc")
ax.set_ylabel("Średnia cena (EUR/MWh)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(REPORTS_DIR / "eda_monthly_prices.png", dpi=150)
plt.close(fig)

print("\nZapisano wykresy w katalogu:", REPORTS_DIR)