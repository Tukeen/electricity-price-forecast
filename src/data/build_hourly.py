"""Buduje tabelę hourly (UTC + czas lokalny) z entsoe_raw.

Uruchomienie:  python -m src.data.build_hourly
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
TZ = CFG["market"]["timezone"]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "electricity.duckdb"

SERIES = {  # (dataset, series_key) -> kolumna w tabeli hourly
    ("prices", "price"): "price_eur_mwh",
    ("load_forecast", "load_forecast"): "load_fc_mw",
    ("res_forecast", "B16"): "pv_fc_mw",
    ("res_forecast", "B19"): "wind_onshore_fc_mw",
    ("res_forecast", "B18"): "wind_offshore_fc_mw",
}
PERIODS_PER_HOUR = {"PT60M": 1, "PT30M": 2, "PT15M": 4}


def aggregate_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Średnia z okresów rynkowych w obrębie godziny UTC.

    Godzina jest kompletna tylko wtedy, gdy ma wszystkie okresy
    (1 dla PT60M, 4 dla PT15M); w przeciwnym razie wartość to NaN.
    """
    df = raw.copy()
    df["hour"] = df["timestamp_utc"].dt.floor("h")
    df["expected"] = df["resolution"].map(PERIODS_PER_HOUR)
    g = df.groupby("hour").agg(
        value=("value", "mean"), n=("value", "size"), expected=("expected", "max")
    )
    g.loc[g["n"] < g["expected"], "value"] = float("nan")
    return g.rename_axis("timestamp_utc").reset_index()[["timestamp_utc", "value", "n", "expected"]]


def build() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = con.sql(
        "SELECT dataset, series_key, timestamp_utc, value, resolution, currency FROM entsoe_raw"
    ).df()
    con.close()

    # Początek danych = północ czasu lokalnego w dniu start_date.
    start = pd.Timestamp(str(CFG["data"]["start_date"])).tz_localize(TZ).tz_convert("UTC").tz_localize(None)

    parts = {}
    for (dataset, key), col in SERIES.items():
        s = raw[(raw["dataset"] == dataset) & (raw["series_key"] == key)]
        if dataset == "prices":
            dropped = s[s["currency"] != "EUR"]
            if len(dropped):
                print(f"UWAGA: pominięto {len(dropped)} rekordów cen nie-EUR "
                      f"({dropped['resolution'].value_counts().to_dict()})")
            s = s[s["currency"] == "EUR"]
        s = s[s["timestamp_utc"] >= start]
        parts[col] = aggregate_hourly(s).set_index("timestamp_utc")

    idx = pd.date_range(start, parts["price_eur_mwh"].index.max(), freq="h")
    local = idx.tz_localize("UTC").tz_convert(TZ)

    df = pd.DataFrame({"timestamp_utc": idx})
    df["timestamp_local"] = local.tz_localize(None)
    df["delivery_date"] = local.date
    df["hour_local"] = local.hour
    df["utc_offset_h"] = ((local.tz_localize(None) - idx) / pd.Timedelta(hours=1)).astype(int)
    for col, part in parts.items():
        df[col] = part["value"].reindex(idx).to_numpy()
    return df


def report(df: pd.DataFrame) -> None:
    cols = list(SERIES.values())

    year = pd.DatetimeIndex(df["timestamp_local"]).year
    rep = df.groupby(year).agg(
        hours=("timestamp_utc", "size"),
        **{f"miss_{c}": (c, lambda s: int(s.isna().sum())) for c in cols},
    )
    print("\n=== brakujące godziny według roku (czas lokalny) ===")
    print(rep.to_string())

    print("\n=== luki wewnątrz zakresu każdej serii ===")
    s_all = df.set_index("timestamp_utc")
    for c in cols:
        s = s_all[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            print(f"{c}: brak danych")
            continue
        inner = s.loc[first:last]
        missing = inner[inner.isna()]
        print(f"{c}: {first} -> {last}, luk: {len(missing)}")
        if len(missing):
            print("   ", [str(t) for t in missing.index[:10]])


def main() -> None:
    df = build()
    con = duckdb.connect(str(DB_PATH))
    con.register("hourly_df", df)
    con.execute("CREATE OR REPLACE TABLE hourly AS SELECT * FROM hourly_df")
    con.close()
    print(f"hourly: {len(df)} godzin, {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()} (UTC)")
    report(df)


if __name__ == "__main__":
    main()