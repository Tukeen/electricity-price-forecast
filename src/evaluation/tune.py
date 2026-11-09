"""Strojenie hiperparametrów na walidacji (rok 2025). Rok 2026 nie jest używany.

Każdy wariant modelu (bez / ze zmiennymi egzogenicznymi) jest strojony osobno.
Wybór: najprostsza konfiguracja (pierwsza w siatce) w granicach 1% MAE od najlepszej.

python -m src.evaluation.tune --family linear xgb prophet --jobs 10
python -m src.evaluation.tune --family mlp --smoke        # pierwsza konfiguracja każdego modelu, 28 dób
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import itertools
import time

import pandas as pd

from src.evaluation.metrics import errors, mae, rmse
from src.evaluation.protocol import rolling_forecast, rolling_forecast_blocks
from src.features.panels import load_panels
from src.models.linear import LinearEPF
from src.models.mlp import MLPEPF
from src.models.prophet_epf import ProphetEPF
from src.models.sarima import SarimaEPF
from src.models.xgb import XGBoostEPF
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EV = CFG["evaluation"]
OUT = PROJECT_ROOT / "reports" / "tables" / "tuning_results.csv"
FAMILIES = ["linear", "xgb", "prophet", "mlp", "sarima"]
MIN_GAIN = 0.01


def specs(family: str):
    """(klucz modelu, etykieta, fabryka, tryb). Pierwsza konfiguracja każdego modelu = obecna domyślna."""
    out = []
    for use_x in (False, True):
        sfx = "_x" if use_x else ""
        if family == "linear":
            for kind, win in itertools.product(("arx", "lear"), (364, 182, 728)):
                out.append((f"{kind}{sfx}", f"w{win}",
                            lambda k=kind, x=use_x, w=win: LinearEPF(k, x, w), "daily"))
        elif family == "xgb":
            for win, depth, (n, lr) in itertools.product((364, 728), (5, 3, 7), ((300, 0.05), (600, 0.03))):
                out.append((f"xgb{sfx}", f"w{win}_d{depth}_n{n}",
                            lambda x=use_x, w=win, d=depth, n=n, lr=lr: XGBoostEPF(
                                x, w, dict(max_depth=d, n_estimators=n, learning_rate=lr)), "daily"))
        elif family == "prophet":
            for cps, win in ((0.05, 364), (0.01, 364), (0.2, 364), (0.05, 728)):
                label = f"cps{cps:g}" + ("" if win == 364 else f"_w{win}")
                out.append((f"prophet{sfx}", label,
                            lambda x=use_x, c=cps, w=win: ProphetEPF(x, w, c), "daily"))
        elif family == "mlp":
            for win, hidden, alpha in ((364, (64, 32), 1e-3), (364, (32, 16), 1e-3), (364, (64, 32), 1e-1),
                                       (364, (128, 64), 1e-2), (728, (64, 32), 1e-3)):
                out.append((f"mlp{sfx}", f"w{win}_h{'-'.join(map(str, hidden))}_a{alpha:g}",
                            lambda x=use_x, w=win, h=hidden, a=alpha: MLPEPF(
                                x, w, params=dict(hidden_layer_sizes=h, alpha=a)), "daily"))
        elif family == "sarima":
            for order, win in (((2, 0, 1), 364), ((1, 0, 1), 364), ((3, 0, 1), 364), ((2, 0, 1), 728)):
                label = "o" + "".join(map(str, order)) + "_s101" + ("" if win == 364 else f"_w{win}")
                out.append((f"sarima{'x' if use_x else ''}", label,
                            lambda x=use_x, o=order, w=win: SarimaEPF(x, w, o, (1, 0, 1, 24)), "blocks"))
    return out


def evaluate(builder, mode: str, panels: dict, days, jobs: int) -> dict:
    model = builder()
    t0 = time.time()
    if mode == "blocks":
        pred = rolling_forecast_blocks(model, panels, days, 28, jobs)
    else:
        pred = rolling_forecast(model, panels, days, EV["window_days"], n_jobs=jobs)
    e = errors(pred, panels["price"])
    ok = e.notna().all(axis=1)
    return {"mae": mae(e[ok]), "rmse": rmse(e[ok]), "nan_days": int((~ok).sum()), "seconds": time.time() - t0}


def choose(g: pd.DataFrame) -> pd.Series:
    """Konfiguracja domyślna (pierwsza w siatce), chyba że najlepsza poprawia MAE o więcej niż MIN_GAIN."""
    g = g.reset_index(drop=True)
    best = g.loc[g["mae"].idxmin()]
    if pd.isna(g.loc[0, "mae"]) or best["mae"] < (1 - MIN_GAIN) * g.loc[0, "mae"]:
        return best
    return g.loc[0]


def summarize(done: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in done.groupby("model", sort=False):
        g = g.reset_index(drop=True)
        c = choose(g)
        rows.append({"model": model, "konfiguracji": len(g),
                     "domyślna": g.loc[0, "label"], "MAE domyślna": g.loc[0, "mae"],
                     "najlepsza": g.loc[g["mae"].idxmin(), "label"], "MAE najlepsza": g["mae"].min(),
                     "wybrana": c["label"], "MAE wybrana": c["mae"],
                     "zmiana vs domyślna %": 100 * (c["mae"] / g.loc[0, "mae"] - 1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=["linear", "xgb"], choices=FAMILIES)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    days = price.loc[pd.Timestamp(EV["validation"]["start"]): pd.Timestamp(EV["validation"]["end"])].index
    if args.smoke:
        days = days[:28]

    cols = ["model", "label", "mae", "rmse", "nan_days", "seconds"]
    done = pd.read_csv(OUT) if (OUT.exists() and not args.smoke) else pd.DataFrame(columns=cols)
    seen = set()
    for family in args.family:
        for model, label, builder, mode in specs(family):
            if args.smoke:
                if model in seen:
                    continue
                seen.add(model)
            elif ((done["model"] == model) & (done["label"] == label)).any():
                continue
            res = evaluate(builder, mode, panels, days, args.jobs)
            print(f"{model:10s} {label:22s} MAE {res['mae']:7.3f}  RMSE {res['rmse']:7.3f}  "
                  f"dób z NaN: {res['nan_days']}  {res['seconds']:.0f} s", flush=True)
            if not args.smoke:
                done = pd.concat([done, pd.DataFrame([{"model": model, "label": label, **res}])],
                                 ignore_index=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                done.to_csv(OUT, index=False)

    if not args.smoke and len(done):
        print("\n=== podsumowanie (walidacja 2025) ===")
        print(summarize(done).round(3).to_string(index=False))


if __name__ == "__main__":
    main()