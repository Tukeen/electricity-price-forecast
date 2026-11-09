"""H4 (część 1): jakość prognoz przedziałowych (ocena 2021-2025). python -m src.evaluation.run_intervals"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.evaluation.intervals import LEVELS, daily_interval_metrics, dm_scores, empirical_intervals
from src.evaluation.protocol import load_forecasts
from src.evaluation.report import groups
from src.features.panels import DB_PATH, load_panels
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EV = CFG["evaluation"]
MODELS = ["naive_mixed", "arx", "arx_x", "lear", "lear_x", "sarima", "sarimax", "var1",
          "xgb", "xgb_x", "mlp", "mlp_x", "prophet", "prophet_x"]
PAIRS = [("xgb_x", "lear_x"), ("mlp_x", "lear_x"), ("xgb_x", "sarimax"), ("lear_x", "sarimax"),
         ("xgb", "lear"), ("mlp", "lear")]
LABELS = {"coverage": "pokrycie (cel = poziom nominalny)", "width": "średnia szerokość [EUR/MWh]",
          "score": "wynik przedziałowy [EUR/MWh] (mniejszy = lepszy)"}


def main() -> None:
    price = load_panels()["price"]
    days = price.loc[pd.Timestamp(EV["regime_analysis"]["start"]): pd.Timestamp(EV["validation"]["end"])].index
    y = price.reindex(days)

    daily = {}
    for m in MODELS:
        pred = load_forecasts(DB_PATH, m).reindex(days)
        for level in LEVELS:
            lo, hi = empirical_intervals(pred, y, level)
            daily[(m, level)] = daily_interval_metrics(lo, hi, y, level)

    valid = pd.concat([d["score"].notna() for d in daily.values()], axis=1).all(axis=1)
    print(f"Doby z przedziałami dla wszystkich modeli: {int(valid.sum())} (od {valid[valid].index[0].date()})")

    grp = groups(days, CFG["regimes"])
    show = ["całość", "2021-2024", *CFG["regimes"]]
    frames = []
    for level in LEVELS:
        for metric in ("coverage", "width", "score"):
            rows = {}
            for g, mask in grp.items():
                m = (mask & valid).to_numpy()
                if m.any():
                    rows[g] = {mod: daily[(mod, level)].loc[m, metric].mean() for mod in MODELS}
            t = pd.DataFrame(rows).T
            frames.append(t.reset_index().rename(columns={"index": "grupa"}).assign(poziom=level, miara=metric))
            print(f"\n=== poziom {level:.0%}: {LABELS[metric]} ===")
            print(t.loc[show].round(3).to_string())

    rows = []
    for g in show:
        m = (grp[g] & valid).to_numpy()
        for a, b in PAIRS:
            stat, p = dm_scores(daily[(a, 0.9)].loc[m, "score"], daily[(b, 0.9)].loc[m, "score"])
            rows.append({"grupa": g, "A": a, "B": b, "dni": int(m.sum()), "DM": stat, "p": p})
    dm = pd.DataFrame(rows)

    out = PROJECT_ROOT / "reports" / "tables"
    pd.concat(frames).to_csv(out / "interval_results.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(out / "interval_dm.csv", index=False, encoding="utf-8-sig")
    print("\n=== Diebold-Mariano na wyniku przedziałowym 90% (DM < 0: A lepszy od B) ===")
    print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()