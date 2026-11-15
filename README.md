# Electricity Price Forecast

Zautomatyzowany potok danych do prognozowania cen energii elektrycznej na polskim rynku dnia następnego
(SDAC, strefa PL) oraz porównanie metod ekonometrycznych i uczenia maszynowego. Część praktyczna pracy
magisterskiej (AGH, Informatyka i Ekonometria).

Aplikacja: https://electricity-price-forecast.streamlit.app

## Zakres

- Dane: ENTSO-E Transparency Platform – ceny RDN, prognoza zapotrzebowania, prognozy wiatru i PV (od 2020 r., EUR/MWh).
- Modele: naiwny (D−1/D−7), ARX, LEAR, VAR(1), SARIMA/SARIMAX, XGBoost, MLP, Prophet; warianty bez i ze zmiennymi
  zewnętrznymi.
- Ocena: prognoza krocząca doba po dobie (24 h), MAE/RMSE/ME/rMAE, test Diebolda–Mariano z korektą Holma,
  bootstrap blokowy, przedziały empiryczne.
- Okresy: rozwojowy 2021–2025, testowy 2026 (zamrożony, uruchamiany jednokrotnie).

## Struktura

| Katalog | Zawartość |
|---|---|
| `src/data` | klient API, parser XML, budowa bazy DuckDB (`entsoe_raw`, `hourly`) |
| `src/features` | panele dobowe, zbiór informacji, kalendarz świąt |
| `src/models` | implementacje modeli (wspólny interfejs: `predict(info, day)` → 24 ceny) |
| `src/evaluation` | protokół, metryki, strojenie, testy hipotez, test 2026, zamrożenie |
| `src/operational` | codzienna prognoza do aplikacji |
| `app` | aplikacja Streamlit (czyta wyłącznie `app_data/`) |
| `scripts` | tabele i wykresy do pracy, eksport wyników do aplikacji, diagnostyka |
| `config` | `settings.yaml` (dane, okresy, reżimy), `model_params.yaml` (konfiguracje), `app.yaml` |
| `docs` | `decisions.md` (decyzje projektowe), `test_protocol_2026.md` (protokół testu) |
| `tests` | testy automatyczne (pytest) |

## Uruchomienie lokalne

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt
# klucz API w pliku .env (ENTSOE_API_KEY=...), plik nie trafia do repozytorium
python -m pytest -q
python -m src.data.ingest          # pobranie danych i budowa tabeli entsoe_raw
python -m src.data.build_hourly    # tabela godzinowa w czasie lokalnym
python -m src.operational.run_daily
streamlit run app/streamlit_app.py
```

Tabele i wykresy do rozdziału 4: `python scripts/thesis_outputs.py --period dev|test --spec main|sens`.

## Automatyzacja

Workflow `.github/workflows/daily_forecast.yml` uruchamia się codziennie po publikacji wyników aukcji (12:10 UTC,
zapasowo 16:10 UTC): testy, pobranie danych, prognoza, zapis do `app_data/` i commit. Prognozowana jest doba
rozstrzygana w najbliższej aukcji; pierwsza zapisana prognoza doby nie jest później zmieniana. Klucz API jest
przechowywany jako sekret repozytorium `ENTSOE_API_KEY`.

## Zamrożenie i test 2026

Kod modeli, cech, oceny i konfiguracja badania są zamrożone tagiem `pre-test-2026`
(`python -m src.evaluation.freeze --check`). Test 2026 uruchamiany jest jednokrotnie po zakończeniu roku
zgodnie z `docs/test_protocol_2026.md`.

## Bezpieczeństwo

Do repozytorium nie trafiają klucze API, pliki `.env`, dane surowe ani bazy danych.