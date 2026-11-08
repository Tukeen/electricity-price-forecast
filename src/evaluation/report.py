"""Tabele zbiorcze: metryki i testy Diebolda-Mariano w grupach (całość, reżimy, lata)."""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import diebold_mariano, mae, me, rmse


def common_days(errs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Zostawia doby, dla których każdy model ma komplet 24 prognoz."""
    idx = None
    for e in errs.values():
        ok = e.index[e.notna().all(axis=1)]
        idx = ok if idx is None else idx.intersection(ok)
    dropped = len(next(iter(errs.values()))) - len(idx)
    if dropped:
        print(f"Uwaga: pominięto {dropped} dób bez prognozy któregoś z modeli")
    return {k: v.loc[idx] for k, v in errs.items()}


def groups(days: pd.DatetimeIndex, regimes: dict) -> dict[str, pd.Series]:
    out = {"całość": pd.Series(True, index=days)}
    for name, r in regimes.items():
        out[name] = pd.Series((days >= pd.Timestamp(r["start"])) & (days <= pd.Timestamp(r["end"])), index=days)
    for y in sorted(set(days.year)):
        out[str(y)] = pd.Series(days.year == y, index=days)
    return out


def summarize(errs: dict[str, pd.DataFrame], ref: str, regimes: dict) -> pd.DataFrame:
    days = errs[ref].index
    rows = []
    for g, mask in groups(days, regimes).items():
        m = mask.to_numpy()
        if not m.any():
            continue
        ref_mae = mae(errs[ref].loc[m])
        for name, e in errs.items():
            sel = e.loc[m]
            rows.append({"grupa": g, "model": name, "dni": len(sel), "MAE": mae(sel), "RMSE": rmse(sel),
                         "ME": me(sel), "rMAE": mae(sel) / ref_mae})
    return pd.DataFrame(rows)


def dm_table(errs: dict[str, pd.DataFrame], pairs: list[tuple[str, str]], regimes: dict) -> pd.DataFrame:
    """stat < 0: pierwszy model z pary ma mniejszą stratę (jest lepszy)."""
    days = next(iter(errs.values())).index
    rows = []
    for g, mask in groups(days, regimes).items():
        m = mask.to_numpy()
        if m.sum() < 30:
            continue
        for a, b in pairs:
            if a in errs and b in errs:
                stat, p = diebold_mariano(errs[a].loc[m], errs[b].loc[m])
                rows.append({"grupa": g, "A": a, "B": b, "dni": int(m.sum()), "DM": stat, "p": p})
    return pd.DataFrame(rows)