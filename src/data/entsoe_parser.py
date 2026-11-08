"""Parser dokumentów ENTSO-E (Publication_MarketDocument i GL_MarketDocument)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

STEPS = {
    "PT15M": timedelta(minutes=15),
    "PT30M": timedelta(minutes=30),
    "PT60M": timedelta(hours=1),
}
VALUE_TAGS = ("price.amount", "quantity")


class NoDataError(Exception):
    """API zwróciło komunikat zamiast danych."""


def read_root(content: bytes) -> ET.Element:
    root = ET.fromstring(content)
    for el in root.iter():  # usuwamy przestrzenie nazw
        el.tag = el.tag.rsplit("}", 1)[-1]
    return root


def ack_message(root: ET.Element) -> str:
    for el in root.iter("text"):
        if el.text:
            return el.text
    return "brak komunikatu"


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_document(content: bytes, default_key: str) -> list[dict]:
    """Zamienia dokument XML na rekordy (jeden na okres rynkowy).

    Dla krzywej A03 pominięte pozycje przejmują poprzednią wartość.
    """
    root = read_root(content)
    if root.tag == "Acknowledgement_MarketDocument":
        raise NoDataError(ack_message(root))

    records = []
    for ts in root.iter("TimeSeries"):
        curve = ts.findtext("curveType")
        currency = ts.findtext("currency_Unit.name")
        unit = ts.findtext("price_Measure_Unit.name") or ts.findtext("quantity_Measure_Unit.name")
        key = ts.findtext("MktPSRType/psrType") or default_key

        for period in ts.findall("Period"):
            start = _ts(period.findtext("timeInterval/start"))
            end = _ts(period.findtext("timeInterval/end"))
            resolution = period.findtext("resolution")
            step = STEPS.get(resolution)
            if step is None:
                print(f"  nieobsługiwana rozdzielczość: {resolution}")
                continue

            values = {}
            for point in period.findall("Point"):
                raw = next((point.findtext(t) for t in VALUE_TAGS if point.findtext(t) is not None), None)
                if raw is not None:
                    values[int(point.findtext("position"))] = float(raw)

            previous = None
            for pos in range(1, int((end - start) / step) + 1):
                if pos in values:
                    previous = values[pos]
                elif not (curve == "A03" and previous is not None):
                    continue
                records.append({
                    "timestamp_utc": start + (pos - 1) * step,
                    "series_key": key,
                    "value": previous,
                    "resolution": resolution,
                    "unit": unit,
                    "currency": currency,
                    "curve_type": curve,
                })
    return records