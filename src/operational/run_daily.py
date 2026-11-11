"""Codzienna prognoza operacyjna na najbliższą dobę bez znanych cen (poza zamrożonym kodem testu 2026).

python -m src.operational.run_daily
python -m src.operational.run_daily --models naive_mixed lear_x
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import yaml

from src.evaluation.protocol import rolling_forecast
from src.evaluation.registry import build_model
from src.features.panels import DB_PATH
from src.models.naive import Naive
from src.operational.panels import build_forecast_panels, exog_complete
from src.utils.config import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_DIR = PROJECT_ROOT / "app_data"
LOG_COLS = ["run_utc", "delivery_date", "model", "hour_local", "y_pred"]


def needs_exog(key: str) -> bool:
    return key.endswith("_x") or key == "sarimax"


def make_model(key: str):
    if key.startswith("naive_"):
        return Naive(key.split("_", 1)[1])
    return build_model(key)  # konfiguracja wybrana po strojeniu (config/model_params.yaml)


def forecast_model(key: str, panels: dict, day: pd.Timestamp) -> np.ndarray:
    model = make_model(key)
    if hasattr(model, "forecast_block"):          # SARIMA(X): estymacja na oknie, prognoza jednej doby
        pred = model.forecast_block(panels, pd.DatetimeIndex([day]))[0]
    else:
        window = getattr(model, "window_days", 364)
        pred = rolling_forecast(model, panels, [day], window, n_jobs=1).iloc[0].to_numpy()
    return np.asarray(pred, dtype=float)


def merge_log(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Dopisuje tylko prognozy (doba, model), których jeszcze nie ma: obowiązuje pierwsza prognoza."""
    if existing.empty:
        return new[LOG_COLS]
    keys = set(zip(existing["delivery_date"], existing["model"]))
    keep = [(d, m) not in keys for d, m in zip(new["delivery_date"], new["model"])]
    return pd.concat([existing, new.loc[keep]], ignore_index=True)[LOG_COLS]


def export_history(start: str) -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        """SELECT delivery_date, hour_local, price_eur_mwh, load_fc_mw, wind_onshore_fc_mw, pv_fc_mw
           FROM hourly WHERE delivery_date >= ? ORDER BY timestamp_utc""", [start]).df()
    con.close()
    df.round({"price_eur_mwh": 2, "load_fc_mw": 1, "wind_onshore_fc_mw": 1, "pv_fc_mw": 1}).to_csv(
        APP_DIR / "prices_hourly.csv", index=False)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--only-missing", action="store_true",
                    help="tryb popołudniowy: licz tylko modele, których jeszcze nie ma w logu dla tej doby")
    ap.add_argument("--budget-min", type=float, default=40.0, help="limit czasu; kolejne modele są pomijane")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))["operational"]
    models = args.models or cfg["models"]

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.sql("SELECT * FROM hourly ORDER BY timestamp_utc").df()
    con.close()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])

    panels, day = build_forecast_panels(df)
    has_exog = exog_complete(panels, day)
    print(f"Doba prognozowana: {day.date()} | zmienne egzogeniczne dostępne: {has_exog}", flush=True)

    run_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_start = time.time()
    log_path = APP_DIR / "forecast_log.csv"
    existing = (pd.read_csv(log_path, dtype={"delivery_date": str}) if log_path.exists()
                else pd.DataFrame(columns=LOG_COLS))
    done = set(existing.loc[existing["delivery_date"] == str(day.date()), "model"])
    status_path = APP_DIR / "status.json"
    prev = {}
    if args.only_missing and status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("forecast_day") == str(day.date()):
            prev = old.get("models", {})
    results, rows = dict(prev), []
    for key in models:
        if args.only_missing and key in done:
            continue                                   # już policzony we wcześniejszym biegu (pierwsza prognoza obowiązuje)
        if needs_exog(key) and not has_exog:
            results[key] = {"status": f"pominięty: brak prognoz zmiennych egzogenicznych na {day.date()}"}
            continue
        if (time.time() - t_start) / 60 > args.budget_min:
            results[key] = {"status": "pominięty: limit czasu"}
            continue
        t0 = time.time()
        try:
            pred = forecast_model(key, panels, day)
            if pred.shape != (24,) or not np.isfinite(pred).all():
                raise ValueError("prognoza ma braki lub złą długość")
        except Exception as exc:
            results[key] = {"status": f"błąd: {type(exc).__name__}: {exc}"[:200]}
            print(f"{key}: {results[key]['status']}", flush=True)
            continue
        results[key] = {"status": "ok", "seconds": round(time.time() - t0)}
        print(f"{key}: ok ({results[key]['seconds']} s)", flush=True)
        rows.append(pd.DataFrame({"run_utc": run_utc, "delivery_date": str(day.date()), "model": key,
                                  "hour_local": range(24), "y_pred": pred}))

    APP_DIR.mkdir(exist_ok=True)
    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=LOG_COLS)
    merge_log(existing, new).sort_values(["delivery_date", "model", "hour_local"], kind="stable").to_csv(
        log_path, index=False, float_format="%.2f")
    n_hist = export_history(cfg["history_start"])

    status = {"run_utc": run_utc, "forecast_day": str(day.date()),
              "last_price_day": str((day - pd.Timedelta(days=1)).date()), "exog_available": has_exog,
              "code_version": os.environ.get("GITHUB_SHA", "lokalnie")[:7], "models": results}
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Zapisano: {len(new)} wierszy prognoz, {n_hist} wierszy historii")
    if not rows and not args.only_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()