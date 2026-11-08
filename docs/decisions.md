agregacja 15→60 min średnią z 4 kwadransów;
doba dostawy w czasie lokalnym;
waluta EUR, start danych 2020-01-01;
schemat walidacji (tabela z poprzedniej wiadomości);
reguła sejfu dla 2026;
transformacja asinh.
pierwsza godzina 2020 bez danych egzogenicznych (start modeli od 2020-01-02), wiatr morski wyłączony, PV od 2020-04-10
EDA i testy na danych do 31.12.2025 (2026 tylko opisowo) oraz transformacja asinh z parametrami z danych rozwojowych
Granice reżimów wiążemy z obserwowalnymi zdarzeniami: stabilny 2020-01-01 → 2021-08-31, kryzys 2021-09-01 → 2023-06-10, OZE z cenami ujemnymi od 2023-06-11. Ostatnią granicę dałoby się uzasadnić statystycznie testem punktów zmiany (opcjonalnie w 4.2).
Okno kalibracji 364 dni jako główne. Przy oknie 2-letnim reżim stabilny (do VIII 2021) wypadał całkowicie z oceny, a to jedno z pytań badawczych. Przy 364 dniach ocena zaczyna się 2021-01-02 i obejmuje 8 miesięcy stabilności. Okno 728 dni będzie analizą wrażliwości od 2022.
Parametry asinh liczymy w każdym oknie osobno. Parametry z całych danych (a=94,6, b=48,3) przeciekałyby informację z przyszłości.
Doba = 24 wartości (wiersz na dobę, kolumny to godziny lokalne). Zdublowaną godzinę jesienią uśredniamy, brakującą wiosną interpolujemy.
Cechy budujemy w krokach 6–7, osobno dla każdej klasy modeli, bo ich format się różni. Krok 5 to panele danych, protokół i benchmarki.
prognozując dobę D, widzimy ceny do doby D−1 i prognozy zapotrzebowania/OZE na dobę D
asinh z parametrami (mediana, 1,4826·MAD) liczonymi osobno w każdym oknie kalibracji;
powrót do EUR przez odwrotną transformację (to w przybliżeniu mediana, nie średnia);
LEAR używa prognoz egzogenicznych tylko z doby D (24 godziny), co jest uproszczeniem względem wersji z literatury;
wiersze z brakującym PV (przed 2020-04-10) są pomijane przy estymacji, więc wczesny 2021 ma mniej obserwacji;
jeśli okno ma za mało wierszy (< liczba cech + 10), doba jest pomijana i raport liczy tylko doby wspólne dla wszystkich modeli.