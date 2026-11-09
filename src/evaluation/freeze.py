"""Zamrożenie kodu (tag git) i danych (odcisk paneli) przed testem 2026.

python -m src.evaluation.freeze --check
python -m src.evaluation.freeze --snapshot        # w styczniu, po pobraniu pełnego 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

from src.features.panels import DB_PATH, load_panels
from src.utils.config import PROJECT_ROOT, load_config

TAG = "pre-test-2026"
# Zamrożone: kod modeli, cech, ewaluacji i ustawień wpływających na wyniki. Poza zamrożeniem: src/data i
# src/analysis (dane zabezpiecza odcisk), aplikacja, automatyzacja i testy.
FROZEN_PATHS = ["src/models", "src/features", "src/evaluation", "src/utils",
                "config/settings.yaml", "config/model_params.yaml"]
SNAPSHOT = PROJECT_ROOT / "docs" / "frozen_data.json"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def data_fingerprint(panels: dict, end: pd.Timestamp) -> str:
    """SHA-256 paneli (ceny i zmienne egzogeniczne) do doby `end` włącznie."""
    h = hashlib.sha256()
    for key in sorted(panels):
        p = panels[key].loc[:end]
        h.update(key.encode())
        h.update(pd.util.hash_pandas_object(p, index=True).to_numpy().tobytes())
        h.update(str(p.index.max().date()).encode())
    return h.hexdigest()


def check_frozen(panels: dict, end: pd.Timestamp) -> list[str]:
    problems = []
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode != 0:
        problems.append(f"brak taga git '{TAG}'")
    elif _git("diff", "--quiet", TAG, "--", *FROZEN_PATHS).returncode != 0:
        problems.append(f"zamrożony kod ({', '.join(FROZEN_PATHS)}) różni się od taga '{TAG}'")
    if _git("status", "--porcelain", "--", *FROZEN_PATHS).stdout.strip():
        problems.append("niezatwierdzone zmiany lub nowe pliki w zamrożonych ścieżkach")
    if not SNAPSHOT.exists():
        problems.append("brak docs/frozen_data.json (uruchom: python -m src.evaluation.freeze --snapshot)")
    elif json.loads(SNAPSHOT.read_text(encoding="utf-8"))["fingerprint"] != data_fingerprint(panels, end):
        problems.append("dane różnią się od snapshotu")
    return problems


def snapshot(panels: dict, end: pd.Timestamp) -> None:
    price = panels["price"].loc[:end]
    info = {"fingerprint": data_fingerprint(panels, end), "test_end": str(end.date()),
            "last_price_day": str(price.index.max().date()), "price_days": int(len(price)),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    SNAPSHOT.write_text(json.dumps(info, indent=2), encoding="utf-8")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    target = (FROZEN_DIR / "hourly_frozen.parquet").as_posix()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"COPY (SELECT * FROM hourly WHERE delivery_date <= '{end.date()}') TO '{target}' (FORMAT PARQUET)")
    con.close()
    print(f"Zapisano {SNAPSHOT} oraz kopię {target}\n{info}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    end = pd.Timestamp(load_config()["evaluation"]["test"]["end"])
    panels = load_panels()
    if args.snapshot:
        last = panels["price"].loc[:end].index.max()
        if last < end and not args.allow_partial:
            print(f"Dane cen kończą się {last.date()}, a test ma się kończyć {end.date()}. Najpierw pobierz dane.")
            sys.exit(1)
        snapshot(panels, end)
        return
    problems = check_frozen(panels, end)
    print("OK: kod i dane zamrożone." if not problems else "Problemy:\n- " + "\n- ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()