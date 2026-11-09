"""Modele liniowe (ARX, LEAR) z i bez zmiennych egzogenicznych, ocena 2021-2025.

Uruchomienie:  python -m src.evaluation.run_linear [--models arx arx_x lear lear_x] [--jobs 10]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.evaluation.metrics import errors
from src.evaluation.protocol import load_forecasts, rolling_forecast, save_forecasts
from src.evaluation.report import common_days, dm_table, summarize
from src.features.panels import DB_PATH, load_panels
from src.models.linear import LinearEPF
from src.utils.config import PROJECT_ROOT, load_config
from src.evaluation.registry import build_model

CFG = load_config()
EV = CFG["evaluation"]
SPECS = {"arx": ("arx", False), "arx_x": ("arx", True), "lear": ("lear", False), "lear_x": ("lear", True)}
PAIRS = [("arx", "naive_mixed"), ("arx_x", "naive_mixed"), ("lear", "naive_mixed"), ("lear_x", "naive_mixed"),
         ("arx_x", "arx"), ("lear_x", "lear"), ("lear", "arx"), ("lear_x", "arx_x")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    days = price.loc[pd.Timestamp(EV["regime_analysis"]["start"]): pd.Timestamp(EV["validation"]["end"])].index

    for key in args.models:
        model = build_model(key)
        print(f"== {model.name} ==")
        t0 = time.time()
        pred = rolling_forecast(model, panels, days, EV["window_days"], n_jobs=args.jobs)
        save_forecasts(DB_PATH, model.name, pred, price)
        print(f"   {time.time() - t0:.0f} s")

    errs = {}
    for name in ["naive_daily", "naive_mixed", *SPECS]:
        pred = load_forecasts(DB_PATH, name)
        if not pred.empty:
            errs[name] = errors(pred.reindex(days), price)
    errs = common_days(errs)

    summary = summarize(errs, "naive_mixed", CFG["regimes"])
    dm = dm_table(errs, PAIRS, CFG["regimes"])
    out = PROJECT_ROOT / "reports" / "tables"
    summary.to_csv(out / "linear_results.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(out / "linear_dm.csv", index=False, encoding="utf-8-sig")

    order = list(summary["grupa"].unique())
    for metric in ("MAE", "rMAE"):
        print(f"\n=== {metric} (rMAE względem naive_mixed) ===")
        print(summary.pivot(index="grupa", columns="model", values=metric).loc[order].round(3).to_string())
    print("\n=== Diebold-Mariano (DM < 0: model A lepszy od B) ===")
    print(dm[dm["grupa"].isin(["całość", *CFG["regimes"]])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()