import pytest

from src.data.entsoe_parser import NoDataError, parse_document

PRICE_XML = """<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <currency_Unit.name>EUR</currency_Unit.name>
    <price_Measure_Unit.name>MWH</price_Measure_Unit.name>
    <curveType>A03</curveType>
    <Period>
      <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T04:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>10.0</price.amount></Point>
      <Point><position>3</position><price.amount>30.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""

GL_XML = """<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <curveType>A01</curveType>
    <MktPSRType><psrType>B16</psrType></MktPSRType>
    <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>
    <Period>
      <timeInterval><start>2024-06-01T00:00Z</start><end>2024-06-01T00:30Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>5</quantity></Point>
      <Point><position>2</position><quantity>7</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

ACK_XML = """<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason><code>999</code><text>No matching data found</text></Reason>
</Acknowledgement_MarketDocument>"""


def test_a03_fills_skipped_positions():
    records = parse_document(PRICE_XML.encode(), "price")
    assert [r["value"] for r in records] == [10.0, 10.0, 30.0, 30.0]
    assert records[0]["currency"] == "EUR"


def test_gl_document_quarter_hours_and_psr_type():
    records = parse_document(GL_XML.encode(), "res_forecast")
    assert [r["series_key"] for r in records] == ["B16", "B16"]
    assert [r["value"] for r in records] == [5.0, 7.0]
    assert (records[1]["timestamp_utc"] - records[0]["timestamp_utc"]).seconds == 900
    assert records[0]["unit"] == "MAW"


def test_acknowledgement_raises():
    with pytest.raises(NoDataError):
        parse_document(ACK_XML.encode(), "price")