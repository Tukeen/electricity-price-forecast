"""Panele dobowe do prognozy operacyjnej: dodatkowa doba D bez znanych cen."""
from __future__ import annotations
 
import pandas as pd
 
from src.features.panels import COLUMNS, _to_panel
 
EXOG_KEYS = ("load", "wind", "pv")
 
 
def build_forecast_panels(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """Panele z jedną dodatkową dobą D (następna po ostatniej dobie ze znanymi cenami).
 
    Ceny doby D to NaN; zmienne egzogeniczne doby D są wypełnione, o ile ENTSO-E je już opublikowało.
    """
    panels = {name: _to_panel(df, col) for name, col in COLUMNS.items()}
    known = panels["price"].dropna()
    if not (known.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Luki w ciągłości dób z cenami")
    day = known.index.max() + pd.Timedelta(days=1)
    idx = known.index.append(pd.DatetimeIndex([day], name=known.index.name))
    return {k: v.reindex(idx) for k, v in panels.items()}, day
 
 
def exog_complete(panels: dict, day: pd.Timestamp) -> bool:
    """Czy dla doby D są wszystkie 24 godziny prognoz zapotrzebowania, wiatru i PV."""
    return all(k in panels and bool(panels[k].loc[day].notna().all()) for k in EXOG_KEYS)