# Protokół testu końcowego 2026 (zamrożony przed testem)

1. **Zakres.** Test out-of-sample na dobach 2026-01-01 do 2026-12-31. Dane ENTSO-E zamrożone odciskiem
   (docs/frozen_data.json), kod tagiem git `pre-test-2026`. Test uruchamiany jednorazowo.
2. **Modele.** Główna specyfikacja (config/model_params.yaml), specyfikacja wrażliwości (konfiguracje domyślne,
   okno 364; modele o niezmienionej konfiguracji współdzielą prognozy). Model eksploracyjny: lear_nl (powstał po
   analizie 8d, bez strojenia).
3. **Miary.** MAE (główna), RMSE, ME, rMAE względem naive_mixed; według całości, kwartałów i miesięcy.
4. **Testy potwierdzające.** DM na dobowych stratach bezwzględnych (HAC, 7 opóźnień), cały rok, para
   xgb_x–lear_x, xgb_x–sarimax, xgb–lear, xgb–sarima; poprawka Holma (4 testy), alfa = 0,05.
   Wynik uznajemy za odporny, jeśli ma ten sam kierunek i p_holm < 0,05 w obu specyfikacjach.
5. **Testy drugorzędne i opisowe.** DM dla pozostałych par i kwartałów (bez korekty, eksploracyjnie);
   doby z ujemną ceną i pozostałe (opisowo); H3: bootstrap blokowy (7 dób, 2000 prób, ziarno 42);
   H4: pokrycie i wynik przedziałowy 90% (empiryczne kwantyle z 90 dób, historia z 2025), według miesięcy.
   Zadeklarowany wcześniej test drugorzędny: DM xgb_x vs lear_x na dobach z ujemną ceną (obie specyfikacje), p bez korekty.” W punkcie 7 dodaj oczekiwanie (e): „XGB_X ma niższe MAE niż LEAR_X na dobach z ujemną ceną (w 2024–2025 o 11–14% w obu specyfikacjach, przewaga ma charakter opisowy: podział po zrealizowanej cenie). Na pozostałych dobach różnica jest niestabilna (od −11% do +4% w latach 2023–2025).
6. **Zasady.** Po uruchomieniu testu nie zmieniamy modeli, hiperparametrów ani protokołu. Błędy techniczne
   harnessu poprawiamy tylko z zapisem w decisions.md. Raportujemy wszystkie wyniki niezależnie od kierunku.
7. **Oczekiwania z okresu rozwojowego (2021-2025).** (a) Modele z egzogenicznymi lepsze od bez nich;
   (b) XGB_X ma najniższe MAE, lecz różnica do LEAR_X jest mała lub nieistotna; (c) bez egzogenicznych LEAR
   nie jest gorszy od XGBoost; (d) pokrycie przedziałów jesienią 2026 niższe niż w pozostałych miesiącach.
8. **Ograniczenia.** SARIMA(X): estymacja co 28 dób; prognozy egzogeniczne ENTSO-E w wersji końcowej;
   przedziały niedoszacowane; strojenie na jednym roku (2025); wnioski z okresu rozwojowego są wstępne.