
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_prices_2020_2024_hourly.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "reports" / "baseline_results.csv"

df = pd.read_csv(INPUT_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

if not df.index.is_unique:
    raise ValueError("W danych występują powtarzające się znaczniki czasu.")

if df["price_eur_mwh"].isna().any():
    raise ValueError("W danych występują brakujące ceny.")

# Zbiór testowy: pełny rok 2024.
test_start = pd.Timestamp("2024-01-01T00:00:00Z")
test_end = pd.Timestamp("2025-01-01T00:00:00Z")

df["actual"] = df["price_eur_mwh"]

# Sezonowość dobowa i tygodniowa w szeregu godzinowym UTC.
df["forecast_previous_day"] = df["price_eur_mwh"].shift(24)
df["forecast_previous_week"] = df["price_eur_mwh"].shift(168)

test = df.loc[(df.index >= test_start) & (df.index < test_end)].copy()

models = {
    "Poprzednia doba (lag 24h)": "forecast_previous_day",
    "Poprzedni tydzień (lag 168h)": "forecast_previous_week",
}

results = []

for model_name, prediction_column in models.items():
    valid = test[["actual", prediction_column]].dropna()

    mae = mean_absolute_error(valid["actual"], valid[prediction_column])
    rmse = mean_squared_error(
        valid["actual"], valid[prediction_column]
    ) ** 0.5

    results.append({
        "model": model_name,
        "test_start_utc": test_start,
        "test_end_exclusive_utc": test_end,
        "n_observations": len(valid),
        "MAE_EUR_MWh": mae,
        "RMSE_EUR_MWh": rmse,
    })

results_df = pd.DataFrame(results).sort_values("MAE_EUR_MWh")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("=== WYNIKI MODELI REFERENCYJNYCH ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nZapisano wyniki:", OUTPUT_PATH)