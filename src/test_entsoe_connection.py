
import truststore
truststore.inject_into_ssl()
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# Wczytanie klucza z pliku .env w katalogu głównym projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

api_key = os.getenv("ENTSOE_API_KEY")

if not api_key:
    raise ValueError(
        "Nie znaleziono ENTSOE_API_KEY. Sprawdź plik .env."
    )

# Testowy zakres: jedna doba w przeszłości, czas UTC
params = {
    "securityToken": api_key,
    "documentType": "A44",
    "in_Domain": "10YPL-AREA-----S",
    "out_Domain": "10YPL-AREA-----S",
    "periodStart": "202501150000",
    "periodEnd": "202501160000",
}

url = "https://web-api.tp.entsoe.eu/api"

response = requests.get(url, params=params, timeout=60)

print("Kod HTTP:", response.status_code)
print("Typ odpowiedzi:", response.headers.get("Content-Type"))

response.raise_for_status()

# Nie wypisujemy całej odpowiedzi ani adresu zapytania,
# ponieważ adres może zawierać klucz API.
content = response.text

if "Publication_MarketDocument" in content:
    print("Sukces: otrzymano dokument z danymi rynkowymi.")
elif "Acknowledgement_MarketDocument" in content:
    print("API zwróciło komunikat o błędzie lub braku danych.")
    print(content[:1500])
else:
    print("Otrzymano odpowiedź, której nie rozpoznano.")
    print(content[:1000])