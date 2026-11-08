"""Panele dobowe: wiersz = doba dostawy (czas lokalny), kolumny = godziny 0-23."""
from __future__ import annotations

import duckdb
import pandas as pd

from src.utils.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "processed" / "electricity.duckdb"
COLUMNS = {
    "price": "price_eur_mwh",
    "load": "load_fc_mw",
    "wind": "wind_onshore_fc_mw",
    "pv": "pv_fc_mw",
}


def _to_panel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Zdublowana godzina (jesień) jest uśredniana, brakująca (wiosna) interpolowana."""
    panel = df.pivot_table(index="delivery_date", columns="hour_local", values=col, aggfunc="mean")
    return panel.reindex(columns=range(24)).interpolate(axis=1, limit_area="inside")


def load_panels() -> dict[str, pd.DataFrame]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.sql("SELECT * FROM hourly ORDER BY timestamp_utc").df()
    con.close()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])

    panels = {name: _to_panel(df, col) for name, col in COLUMNS.items()}
    price = panels["price"].dropna()
    if not (price.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Luki w ciągłości dób w panelu cen")
    out = {k: v.reindex(price.index) for k, v in panels.items()}
    for v in out.values():
        v.index.name = "delivery_date"
    return out


def info_set(panels: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Informacje dostępne przy prognozowaniu doby D: ceny do doby D-1, prognozy egzogeniczne do doby D."""
    out = {"price": panels["price"].loc[: day - pd.Timedelta(days=1)]}
    for key, panel in panels.items():
        if key != "price":
            out[key] = panel.loc[:day]
    return out