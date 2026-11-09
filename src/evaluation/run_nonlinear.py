"""Diagnostyka nieliniowości (2022-2025): czy LEAR_X z cechami progowymi zbliża się do XGB_X?

python -m src.evaluation.run_nonlinear --jobs 10          # liczy prognozy do tabeli forecasts_diag
python -m src.evaluation.run_nonlinear --report-only      # sam raport
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time

import pandas as pd

from src.evaluation.metrics import errors
from src.evaluation.protocol import load_forecasts, rolling_forecast, save_forecasts
from src.evaluation.registry import build_model
from src.evaluation.report import common_days, dm_table, summarize
from src.features.panels import DB_PATH, load_panels
from src.models.linear_nl import LearNL
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EV = CFG["evaluation"]
TABLE = "forecasts_diag"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    days = price.loc[pd.Timestamp("2022-01-01"): pd.Timestamp(EV["validation"]["end"])].index
    window = build_model("lear_x").window_days
    model = LearNL(window)

    if not args.report_only:
        t0 = time.time()
        pred = rolling_forecast(model, panels, days, window, n_jobs=args.jobs)
        save_forecasts(DB_PATH, model.name, pred, price, table=TABLE)
        print(f"{model.name}: {time.time() - t0:.0f} s")

    errs = {"lear_nl": errors(load_forecasts(DB_PATH, "lear_nl", TABLE).reindex(days), price)}
    for m in ("lear_x", "xgb_x", "sarimax"):
        errs[m] = errors(load_forecasts(DB_PATH, m).reindex(days), price)
    errs = common_days(errs)

    summary = summarize(errs, "lear_x", CFG["regimes"])
    pairs = [("lear_nl", "lear_x"), ("xgb_x", "lear_nl"), ("xgb_x", "lear_x"), ("sarimax", "lear_nl")]
    dm = dm_table(errs, pairs, CFG["regimes"])
    out = PROJECT_ROOT / "reports" / "tables"
    summary.to_csv(out / "nonlinear_results.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(out / "nonlinear_dm.csv", index=False, encoding="utf-8-sig")

    order = list(summary["grupa"].unique())
    mae_tab = summary.pivot(index="grupa", columns="model", values="MAE").loc[order]
    print("\n=== MAE ===")
    print(mae_tab.round(3).to_string())
    gap = mae_tab["lear_x"] - mae_tab["xgb_x"]
    closed = 100 * ((mae_tab["lear_x"] - mae_tab["lear_nl"]) / gap).where(gap > 0.3)
    print("\n=== Jaka część luki LEAR_X - XGB_X zamyka LEAR_NL, % (tylko grupy z luką > 0,3 EUR/MWh) ===")
    print(closed.dropna().round(0).to_string())
    print("\n=== Diebold-Mariano (DM < 0: A lepszy od B) ===")
    print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()