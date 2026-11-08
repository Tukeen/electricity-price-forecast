"""Klient HTTP dla ENTSO-E Transparency Platform."""
from __future__ import annotations

import os
import time

import requests
import truststore
from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT

truststore.inject_into_ssl()
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://web-api.tp.entsoe.eu/api"


def fetch_xml(session: requests.Session, params: dict, retries: int = 3) -> bytes | None:
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Brak ENTSOE_API_KEY. Sprawdź plik .env.")

    # Token w nagłówku, żeby nie trafił do adresu URL ani logów.
    headers = {"SECURITY_TOKEN": api_key}

    for attempt in range(1, retries + 1):
        try:
            response = session.get(API_URL, params=params, headers=headers, timeout=120)
        except requests.RequestException as exc:
            print(f"  próba {attempt}/{retries}: {type(exc).__name__}")
        else:
            if response.status_code == 200:
                return response.content
            print(f"  HTTP {response.status_code}")
            if response.status_code in (400, 401, 403):
                return None  # błąd zapytania lub autoryzacji, ponawianie nie pomoże
        time.sleep(3 * attempt)
    return None