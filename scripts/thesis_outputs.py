"""Tabele i wykresy do rozdziału z wynikami (CSV + PNG) z prognoz zapisanych w DuckDB.

Okres rozwojowy (2021-2025):   python scripts/thesis_outputs.py --period dev  --spec main
Test końcowy 2026 (styczeń):   python scripts/thesis_outputs.py --period test --spec main
                               python scripts/thesis_outputs.py --period test --spec sens
Opcje: --pl (CSV z przecinkiem dziesiętnym i średnikiem, do Excela/Worda), --boot 2000, --out katalog, --db plik

Skrypt tylko czyta prognozy (tabele forecasts*), niczego nie liczy od nowa i nie zmienia zamrożonego kodu badania.
Plik leży w scripts/ (poza ścieżkami zamrożonymi: nie wpływa na `freeze --check`).
Wyniki trafiają do reports/thesis/<okres>_<specyfikacja>/ (tabele t*.csv, wykresy f*.png).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # korzeń repozytorium (import pakietu src)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.evaluation.inference import gain_difference, holm
from src.evaluation.metrics import diebold_mariano, mae
from src.evaluation.report import common_days, groups, holdout_groups, summarize
from src.features.panels import DB_PATH
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
REGIMES = CFG["regimes"]
PERIODS = {"dev": (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
           "test": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))}
TABLES = {("dev", "main"): ["forecasts", "forecasts_diag"],
          ("dev", "sens"): ["forecasts_v2", "forecasts", "forecasts_diag"],
          ("test", "main"): ["forecasts_test"],
          ("test", "sens"): ["forecasts_test_v2", "forecasts_test"]}
MODELS = ["naive_mixed", "arx", "arx_x", "lear", "lear_x", "sarima", "sarimax", "var1",
          "xgb", "xgb_x", "mlp", "mlp_x", "prophet", "prophet_x", "lear_nl"]
KEY = ["naive_mixed", "arx", "lear", "sarima", "xgb", "mlp", "arx_x", "lear_x", "sarimax", "xgb_x", "mlp_x"]
SHOW = ["naive_mixed", "lear", "lear_x", "sarimax", "xgb", "xgb_x", "mlp_x"]
PAIRS_X = [("arx", "arx_x"), ("lear", "lear_x"), ("sarima", "sarimax"), ("xgb", "xgb_x"),
           ("mlp", "mlp_x"), ("prophet", "prophet_x")]
ML_X = {"xgb": "xgb_x", "mlp": "mlp_x"}
ECON_X = {"arx": "arx_x", "lear": "lear_x", "sarima": "sarimax"}

NAMES = {"naive_mixed": "Naiwny (D-1/D-7)", "arx": "ARX", "arx_x": "ARX+X", "lear": "LEAR", "lear_x": "LEAR+X",
         "sarima": "SARIMA", "sarimax": "SARIMAX", "var1": "VAR(1)", "xgb": "XGBoost", "xgb_x": "XGBoost+X",
         "mlp": "MLP", "mlp_x": "MLP+X", "prophet": "Prophet", "prophet_x": "Prophet+X",
         "lear_nl": "LEAR-NL (eksploracyjny)", "naive_daily": "Naiwny (D-1)"}
CLASS = {"naive_mixed": "benchmark", "naive_daily": "benchmark", "arx": "ekonometria", "arx_x": "ekonometria",
         "lear": "ekonometria", "lear_x": "ekonometria", "sarima": "ekonometria", "sarimax": "ekonometria",
         "var1": "ekonometria", "lear_nl": "ekonometria", "xgb": "ML", "xgb_x": "ML", "mlp": "ML", "mlp_x": "ML",
         "prophet": "inne", "prophet_x": "inne"}
COLORS = {"benchmark": "#7F7F7F", "ekonometria": "#0072B2", "ML": "#D55E00", "inne": "#009E73"}


def nm(m: str) -> str:
    return NAMES.get(m, m)


def load(db: Path, tables: list[str], models: list[str], days: pd.DatetimeIndex):
    """Zwraca (prognozy, ceny rzeczywiste) jako panele doby x 24 h; model bierze pierwszą tabelę, w której występuje."""
    con = duckdb.connect(str(db), read_only=True)
    preds, true = {}, None
    for m in models:
        for t in tables:
            try:
                df = con.execute(f"SELECT delivery_date, hour_local, y_pred, y_true FROM {t} WHERE model = ?", [m]).df()
            except Exception:
                continue
            if df.empty:
                continue
            df["delivery_date"] = pd.to_datetime(df["delivery_date"])
            preds[m] = df.pivot(index="delivery_date", columns="hour_local", values="y_pred").sort_index().reindex(days)
            if true is None:
                true = df.pivot(index="delivery_date", columns="hour_local", values="y_true").sort_index().reindex(days)
            break
    con.close()
    return preds, true


def save_csv(df: pd.DataFrame, path: Path, pl: bool, index: bool = False) -> None:
    if pl:
        df.to_csv(path, index=index, sep=";", decimal=",", encoding="utf-8-sig", float_format="%.4f")
    else:
        df.to_csv(path, index=index, encoding="utf-8-sig")


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fig_save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "test"], default="dev")
    ap.add_argument("--spec", choices=["main", "sens"], default="main")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pl", action="store_true", help="CSV: średnik i przecinek dziesiętny")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    start, end = PERIODS[args.period]
    out = args.out or PROJECT_ROOT / "reports" / "thesis" / f"{args.period}_{args.spec}"
    out.mkdir(parents=True, exist_ok=True)
    days_all = pd.date_range(start, end)
    preds, true = load(args.db, TABLES[(args.period, args.spec)], MODELS, days_all)
    if not preds or "naive_mixed" not in preds:
        sys.exit("Brak prognoz (lub benchmarku naive_mixed) w bazie dla wybranego okresu.")
    errs = {m: (p - true) for m, p in preds.items()}
    errs = {m: e.loc[e.index.isin(true.dropna(how="all").index)] for m, e in errs.items()}
    # modele z niepełnym pokryciem (np. lear_nl w okresie rozwojowym liczono tylko dla 2022-2025) nie mogą zawężać
    # wspólnego zbioru dób dla pozostałych; porównujemy je osobno (tabela t8)
    n_full = {m: int(e.notna().all(axis=1).sum()) for m, e in errs.items()}
    partial = [m for m, n in n_full.items() if n < 0.9 * n_full["naive_mixed"]]
    errs_partial = {m: errs.pop(m) for m in partial}
    for m in partial:
        print(f"Uwaga: {m} ma prognozy tylko dla {n_full[m]} dób; wyłączony ze wspólnego zestawienia (zob. t8).")
    errs = common_days(errs)
    days = errs["naive_mixed"].index
    price = true.loc[days]
    models = [m for m in MODELS if m in errs]
    key = [m for m in KEY if m in errs]
    daily = {m: errs[m].abs().mean(axis=1) for m in models}
    is_dev = args.period == "dev"
    grp = groups(days, REGIMES) if is_dev else holdout_groups(days)
    tag = f"{'rozwój 2021-2025' if is_dev else 'test 2026'}, specyfikacja {args.spec}"
    print(f"{tag}: {len(days)} dób ({days[0].date()} - {days[-1].date()}), modele: {', '.join(models)}")

    # ---------------- tabele ----------------
    summ = summarize(errs, "naive_mixed", REGIMES, group_fn=(None if is_dev else holdout_groups))
    summ["klasa"] = summ["model"].map(CLASS)
    t1 = summ[summ["grupa"] == "całość"].sort_values("MAE").reset_index(drop=True)
    save_csv(t1, out / "t1_metryki_calosc.csv", args.pl)
    save_csv(summ, out / "t2_metryki_wg_grup.csv", args.pl)

    rows = []
    for a, b in itertools.combinations(key, 2):
        stat, p = diebold_mariano(errs[a], errs[b])
        rows.append({"A": a, "B": b, "dni": len(days), "DM": stat, "p": p,
                     "MAE A/B - 1 [%]": 100 * (mae(errs[a]) / mae(errs[b]) - 1)})
    t3 = pd.DataFrame(rows)
    t3["p_holm"] = holm(t3["p"]).to_numpy()
    save_csv(t3, out / "t3_dm_wszystkie_pary.csv", args.pl)

    rows = []
    for g, mask in grp.items():
        m_ = mask.to_numpy()
        if not m_.any():
            continue
        for b, x in PAIRS_X:
            if b in errs and x in errs:
                rows.append({"grupa": g, "model": f"{b} -> {x}",
                             "zmiana MAE [%]": 100 * (mae(errs[x].loc[m_]) / mae(errs[b].loc[m_]) - 1)})
    t4 = pd.DataFrame(rows)
    save_csv(t4, out / "t4_zysk_zm_zewnetrznych.csv", args.pl)

    neg = (price.min(axis=1) < 0).to_numpy()
    rows = []
    for label_, mask in (("doby z ujemną ceną", neg), ("pozostałe doby", ~neg)):
        if mask.sum():
            rows.append({"podzbiór": label_, "dni": int(mask.sum()), **{m: mae(errs[m].loc[mask]) for m in models}})
    t5 = pd.DataFrame(rows)
    save_csv(t5, out / "t5_ujemne_ceny_MAE.csv", args.pl)

    t6 = pd.DataFrame({m: errs[m].abs().mean(axis=0) for m in models})
    t6.index.name = "godzina"
    save_csv(t6, out / "t6_MAE_wg_godzin.csv", args.pl, index=True)

    rows = []
    for g in (["całość", "2021-2024", *REGIMES] if is_dev else ["całość"]):
        mask = grp[g].to_numpy()
        for mb, mx in ML_X.items():
            for eb, ex in ECON_X.items():
                if all(k in daily for k in (mb, mx, eb, ex)):
                    d, lo, hi, prob = gain_difference(*(daily[k].to_numpy()[mask] for k in (mb, mx, eb, ex)),
                                                      B=args.boot)
                    rows.append({"grupa": g, "ML": mx, "ECON": ex, "Delta [p.p.]": 100 * d, "CI dolna": 100 * lo,
                                 "CI górna": 100 * hi, "P(Delta<0)": prob})
    save_csv(pd.DataFrame(rows), out / "t7_H3_bootstrap.csv", args.pl)

    rows = []
    for m, e in errs_partial.items():
        d = e.index[e.notna().all(axis=1) & e.index.isin(days)]
        if not len(d):
            continue
        for k in [m] + [r for r in ("lear_x", "xgb_x", "mlp_x", "sarimax") if r in errs]:
            ek = errs_partial[k] if k == m else errs[k]
            row = {"model": k, "dni": len(d), "od": d[0].date(), "do": d[-1].date(), "MAE": mae(ek.loc[d])}
            if k != m and "lear_x" in errs:
                row["DM (model częściowy vs ten model)"], row["p"] = diebold_mariano(e.loc[d], ek.loc[d])
            rows.append(row)
    if rows:
        save_csv(pd.DataFrame(rows), out / "t8_modele_czesciowe.csv", args.pl)

    # ---------------- wykresy ----------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

    # F1: MAE modeli (cały okres)
    d1 = t1.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(d1) + 1.2))
    ax.barh([nm(m) for m in d1["model"]], d1["MAE"], color=[COLORS[CLASS[m]] for m in d1["model"]])
    for i, v in enumerate(d1["MAE"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("MAE [EUR/MWh]")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], labels=list(COLORS), loc="upper right")
    fig_save(fig, out / "f1_MAE_modele.png")

    # F2: MAE w czasie (lata lub miesiące)
    keys = [str(y) for y in sorted(set(days.year))] if is_dev else [f"{m:02d}" for m in range(1, 13)]
    pivot = summ.pivot(index="grupa", columns="model", values="MAE")
    keys = [k for k in keys if k in pivot.index]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in [s for s in SHOW if s in pivot.columns]:
        ax.plot(keys, pivot.loc[keys, m], marker="o", ms=3, label=nm(m), color=COLORS[CLASS[m]],
                ls="-" if m.endswith("_x") or m == "sarimax" else "--" if CLASS[m] != "benchmark" else ":")
    ax.set_ylabel("MAE [EUR/MWh]")
    ax.set_xlabel("rok" if is_dev else "miesiąc 2026")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig_save(fig, out / "f2_MAE_w_czasie.png")

    # F3: mapa rMAE
    gcols = ["całość", "stabilny", "kryzys", "oze_ujemne", *keys] if is_dev else ["całość", "Q1", "Q2", "Q3", "Q4"]
    rm = summ.pivot(index="model", columns="grupa", values="rMAE")
    rm = rm.loc[[m for m in models if m != "naive_mixed"], [g for g in gcols if g in rm.columns]]
    fig, ax = plt.subplots(figsize=(0.75 * rm.shape[1] + 2.5, 0.38 * rm.shape[0] + 1.2))
    lim = max(abs(rm.to_numpy() - 1).max(), 0.05)
    im = ax.imshow(rm.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim), aspect="auto")
    ax.set_xticks(range(rm.shape[1]), rm.columns, rotation=30, ha="right")
    ax.set_yticks(range(rm.shape[0]), [nm(m) for m in rm.index])
    for i, j in itertools.product(range(rm.shape[0]), range(rm.shape[1])):
        ax.text(j, i, f"{rm.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="rMAE (1 = naiwny; <1 lepszy)")
    fig_save(fig, out / "f3_rMAE_mapa.png")

    # F4: macierz DM
    order = list(t1[t1["model"].isin(key)]["model"])
    n = len(order)
    M, S = np.full((n, n), np.nan), [[""] * n for _ in range(n)]
    ph = {(r.A, r.B): r for r in t3.itertuples()}
    for i, j in itertools.combinations(range(n), 2):
        a, b = order[i], order[j]
        r = ph.get((a, b))
        if r is not None:
            M[i, j], S[i][j] = r.DM, f"{r.DM:+.1f}{stars(r.p_holm)}"
        else:
            r = ph.get((b, a))
            if r is not None:
                M[i, j], S[i][j] = -r.DM, f"{-r.DM:+.1f}{stars(r.p_holm)}"
    fig, ax = plt.subplots(figsize=(0.8 * n + 2, 0.7 * n + 1))
    lim = max(np.nanmax(np.abs(M)), 1)
    ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(n), [nm(m) for m in order], rotation=40, ha="right")
    ax.set_yticks(range(n), [nm(m) for m in order])
    for i, j in itertools.product(range(n), range(n)):
        if S[i][j]:
            ax.text(j, i, S[i][j], ha="center", va="center", fontsize=7)
    ax.grid(False)
    ax.set_xlim(0.5, n - 0.5)
    ax.set_ylim(n - 1.5, -0.5)
    ax.set_xlabel("B (kolumna)")
    ax.set_ylabel("A (wiersz); DM < 0 (niebieski): A lepszy od B")
    fig_save(fig, out / "f4_DM_macierz.png")

    # F5: przykładowy tydzień wokół najniższej ceny okresu
    day0 = price.min(axis=1).idxmin()
    wk = pd.date_range(day0 - pd.Timedelta(days=3), periods=7)
    wk = wk[(wk >= days[0]) & (wk <= days[-1])]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = np.arange(len(wk) * 24)
    ax.plot(xs, price.loc[wk].to_numpy().ravel(), color="black", lw=1.6, label="cena rzeczywista")
    for m in [s for s in ("lear_x", "xgb_x", "sarimax") if s in preds]:
        ax.plot(xs, preds[m].reindex(wk).to_numpy().ravel(), lw=1, color=COLORS[CLASS[m]],
                ls="-" if m != "sarimax" else "--", label=nm(m))
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(np.arange(0, len(wk) * 24, 24), [d.strftime("%d.%m") for d in wk])
    ax.set_ylabel("EUR/MWh")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig_save(fig, out / "f5_przyklad_tygodnia.png")

    # F6: MAE według godzin
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m in [s for s in SHOW if s in t6.columns]:
        ax.plot(t6.index, t6[m], label=nm(m), color=COLORS[CLASS[m]],
                ls="-" if m.endswith("_x") or m == "sarimax" else "--" if CLASS[m] != "benchmark" else ":")
    ax.set_xlabel("godzina doby dostawy (czas lokalny)")
    ax.set_ylabel("MAE [EUR/MWh]")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig_save(fig, out / "f6_MAE_wg_godzin.png")

    # F7: skumulowana różnica strat (rośnie, gdy XGBoost+X jest lepszy od konkurenta)
    if "xgb_x" in daily:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for other in [s for s in ("lear_x", "sarimax") if s in daily]:
            ax.plot(days, (daily[other] - daily["xgb_x"]).cumsum(), label=f"{nm(other)} - {nm('xgb_x')}",
                    color=COLORS[CLASS[other]], ls="-" if other == "lear_x" else "--")
        if is_dev:
            for name, r in REGIMES.items():
                ax.axvline(pd.Timestamp(r["start"]), color="grey", lw=0.6, ls=":")
                ax.text(max(pd.Timestamp(r["start"]), days[0]), ax.get_ylim()[1], f" {name}", va="top", fontsize=7, color="grey")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("skumulowana różnica MAE dobowego [EUR/MWh]")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig_save(fig, out / "f7_skumulowana_roznica.png")

    # F8: zmiana MAE po dodaniu zmiennych zewnętrznych
    g8 = ["całość", "stabilny", "kryzys", "oze_ujemne"] if is_dev else ["całość", "Q1", "Q2", "Q3", "Q4"]
    p8 = t4[t4["grupa"].isin(g8)].pivot(index="grupa", columns="model", values="zmiana MAE [%]").reindex(g8)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.8 / max(p8.shape[1], 1)
    for k, col in enumerate(p8.columns):
        ax.bar(np.arange(len(g8)) + k * w, p8[col], w, label=col.replace(" -> ", " → "),
               color=["#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00", "#CC79A7"][k % 6])
    ax.set_xticks(np.arange(len(g8)) + 0.4 - w / 2, g8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("zmiana MAE po dodaniu zm. zewnętrznych [%]")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig_save(fig, out / "f8_zysk_zm_zewnetrznych.png")

    print(f"\nZapisano w {out}:")
    for f in sorted(out.iterdir()):
        print("  ", f.name)
    print("\nMAE (cały okres):")
    print(t1[["model", "MAE", "RMSE", "rMAE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()