"""Krok 4: analiza eksploracyjna. Uruchomienie: python -m src.analysis.eda

Testy i ACF tylko na danych <= validation.end (sejf 2026); 2026 tylko opisowo.
"""
from __future__ import annotations

import warnings

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils.config import PROJECT_ROOT, load_config

warnings.filterwarnings("ignore")
CFG = load_config()
DB_PATH = PROJECT_ROOT / "data" / "processed" / "electricity.duckdb"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
DEV_END = pd.Timestamp(CFG["evaluation"]["validation"]["end"])
REGIMES = CFG["regimes"]


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.sql("SELECT * FROM hourly ORDER BY timestamp_utc").df()
    con.close()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    df["price"] = df["price_eur_mwh"]
    df["year"] = df["delivery_date"].dt.year
    df["month"] = df["delivery_date"].dt.to_period("M")
    df["is_neg"] = df["price"] < 0
    df["regime"] = "poza_regimem"
    for name, r in REGIMES.items():
        mask = (df["delivery_date"] >= pd.Timestamp(r["start"])) & (df["delivery_date"] <= pd.Timestamp(r["end"]))
        df.loc[mask, "regime"] = name
    return df


def describe(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)["price"]
    return pd.DataFrame({
        "hours": g.size(), "mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max(),
        "skew": g.skew(), "excess_kurt": g.apply(pd.Series.kurt),
        "neg_hours": df.groupby(by)["is_neg"].sum(),
    })


def stationarity(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in series.items():
        s = s.dropna()
        rows.append({
            "seria": name, "n": len(s),
            "ADF_p (H0: pierwiastek jednostkowy)": adfuller(s, autolag="AIC")[1],
            "KPSS_p (H0: stacjonarność)": kpss(s, regression="c", nlags="auto")[1],
        })
    return pd.DataFrame(rows)


def asinh_transform(s: pd.Series, a: float, b: float) -> pd.Series:
    return np.arcsinh((s - a) / b)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["residual_load_mw"] = d["load_fc_mw"] - d["wind_onshore_fc_mw"] - d["pv_fc_mw"]
    cols = ["load_fc_mw", "wind_onshore_fc_mw", "pv_fc_mw", "residual_load_mw"]
    rows = {y: g[["price"] + cols].corr(method="spearman")["price"][cols] for y, g in d.groupby("year")}
    return pd.DataFrame(rows).T


def shade_regimes(axes) -> None:
    for ax in axes:
        for name, r in REGIMES.items():
            ax.axvspan(pd.Timestamp(r["start"]), pd.Timestamp(r["end"]), alpha=0.07)
        ax.axvline(pd.Timestamp(CFG["evaluation"]["test"]["start"]), color="grey", ls="--", lw=1)


def plot_monthly(df: pd.DataFrame) -> None:
    m = df.groupby("month")["price"].agg(["mean", "std"])
    neg = df.groupby("month")["is_neg"].sum()
    x = m.index.to_timestamp()
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(x, m["mean"]); ax[0].set_ylabel("średnia [EUR/MWh]")
    ax[1].plot(x, m["std"]); ax[1].set_ylabel("odch. std. [EUR/MWh]")
    ax[2].bar(x, neg.values, width=20); ax[2].set_ylabel("godziny z ceną < 0")
    shade_regimes(ax)
    fig.suptitle("Ceny RDN w Polsce: miesięczne statystyki (cieniowanie = robocze reżimy, linia = początek 2026)")
    fig.tight_layout(); fig.savefig(FIGURES / "01_monthly.png", dpi=150); plt.close(fig)


def plot_hourly_profiles(df: pd.DataFrame) -> None:
    prof = df.pivot_table(index="hour_local", columns="year", values="price", aggfunc="mean")
    ax = prof.plot(figsize=(10, 5), marker="o", ms=3)
    ax.set_xlabel("godzina doby (czas lokalny)"); ax.set_ylabel("średnia cena [EUR/MWh]")
    ax.set_title("Średni profil dobowy według roku")
    ax.figure.tight_layout(); ax.figure.savefig(FIGURES / "02_hourly_profiles.png", dpi=150); plt.close(ax.figure)


def plot_acf_pacf(dev: pd.DataFrame) -> None:
    s = dev["price"].dropna()
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    plot_acf(s, lags=192, ax=ax[0], title="ACF ceny godzinowej (192 opóźnienia = 8 dób)")
    plot_pacf(s, lags=48, method="ywm", ax=ax[1], title="PACF ceny godzinowej (48 opóźnień)")
    fig.tight_layout(); fig.savefig(FIGURES / "03_acf_pacf.png", dpi=150); plt.close(fig)


def plot_distribution(dev: pd.DataFrame, a: float, b: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dev["price"].dropna(), bins=120); ax[0].set_title("Cena [EUR/MWh]")
    ax[1].hist(asinh_transform(dev["price"].dropna(), a, b), bins=120); ax[1].set_title("asinh((cena - a) / b)")
    fig.tight_layout(); fig.savefig(FIGURES / "04_distribution.png", dpi=150); plt.close(fig)


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.round(3).to_string())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load()
    dev = df[df["delivery_date"] <= DEV_END]

    for name, table in {
        "stats_by_year": describe(df, "year"),
        "stats_by_regime": describe(df, "regime"),
        "correlations_by_year": correlations(df),
    }.items():
        table.to_csv(TABLES / f"{name}.csv", encoding="utf-8-sig")
        show(name, table)

    neg = df[df["is_neg"]]
    print("\nPierwsza godzina z ceną < 0:", neg["timestamp_local"].min() if len(neg) else "brak")
    print("Godziny z ceną < 0 według roku:", neg.groupby("year").size().to_dict())

    # parametry transformacji z danych rozwojowych (bez 2026)
    a = float(dev["price"].median())
    b = float(1.4826 * (dev["price"] - a).abs().median())
    print(f"\nasinh: a (mediana) = {a:.2f}, b (1.4826*MAD) = {b:.2f}")
    t = asinh_transform(dev["price"], a, b)
    print(f"skośność / kurtoza nadmiarowa: cena {dev['price'].skew():.2f} / {dev['price'].kurt():.2f}; "
          f"asinh {t.skew():.2f} / {t.kurt():.2f}")

    series = {
        "cena godzinowa 2020-2025": dev["price"],
        "średnia dobowa 2020-2025": dev.groupby("delivery_date")["price"].mean(),
        "asinh cena godzinowa 2020-2025": t,
    }
    for year, g in dev.groupby("year"):
        series[f"cena godzinowa {year}"] = g["price"]
    stat = stationarity(series)
    stat.to_csv(TABLES / "stationarity.csv", index=False, encoding="utf-8-sig")
    show("testy stacjonarności (KPSS: p ograniczone do 0.01-0.10)", stat.set_index("seria"))

    plot_monthly(df)
    plot_hourly_profiles(df)
    plot_acf_pacf(dev)
    plot_distribution(dev, a, b)
    print(f"\nTabele: {TABLES}\nWykresy: {FIGURES}")


if __name__ == "__main__":
    main()