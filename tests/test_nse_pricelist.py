"""Reading real NSE history from the scanned daily price lists.

The fixtures are the OCR output of the 23 September 2026 list, as read on the
server: text and left edge of each box, including the garbles (KCB's price as
"006"; NCBA's high, low and price as "0006", "0968", "00°06"). Every session's
price is printed twice, as VWAP in its own list and PREVIOUS in the next, and
the checks lean on that rather than on any single OCR reading.
"""
import csv
import json
import sys
from datetime import date

import pytest

import src.connectors.nse_scraper as nse_scraper
from src.agent import history_warmstart
from src.connectors import nse_pricelist as pl
from src.connectors.nse_pricelist import Box

WIDTH = 1700


def _line(y, cells):
    return [Box(float(x), float(y), 22.0, text) for text, x in cells]


HEADINGS = (_line(120, [("52WKHIGH", 197), ("52WKLOW", 276), ("ISIN", 692), ("HIGH", 877),
                        ("LOW", 981), ("VWAP", 1081), ("PREVIOUS", 1195)])
            + _line(150, [("AGRICULTURAL", 455), ("CODE", 693), ("STATUS", 763), ("PRICE", 1208),
                          ("VOLUME", 1333)]))

PAGE1_ROWS = [
    _line(600, [("36.50", 198), ("15.00ABSA Bank Kenya Plc Ord 0.50", 302), ("KE0000000067", 675),
                ("xd", 761), ("33.30", 911), ("32.50", 1008), ("33.00", 1117), ("33.05", 1258),
                ("213,157", 1371)]),
    _line(690, [("109.00", 198), ("41.20", 302), ("Equity Group Holdings Plc Ord 0.50", 338),
                ("KE0000000554", 676), ("107.50", 906), ("105.50", 1004), ("107.00", 1114),
                ("105.00", 1255), ("1,135,601", 1362)]),
    _line(810, [("101.00", 199), ("35.00", 301), ("KCB Group Plc Ord 1.00", 339),
                ("KE0000000315", 675), ("93.25", 911), ("92.50", 1007), ("006", 1117),
                ("93.00", 1258), ("743,476", 1366)]),
    _line(840, [("100.00", 199), ("48.00", 302), ("NCBA Group Plc Ord 5.00", 338),
                ("KE0000000406", 676), ("xd", 759), ("0006", 911), ("0968", 1008), ("00°06", 1117),
                ("90.00", 1259), ("1,167,538", 1361)]),
    _line(900, [("370.00", 198), ("260.00", 294), ("Standard Chartered Bank Kenya Ltd Ord 5.00", 340),
                ("KE0000000448", 676), ("xd", 760), ("340.00", 905), ("328.50", 1001),
                ("330.50", 1111), ("332.00", 1251), ("25,219", 1375)]),
    _line(930, [("39.95", 198), ("13.80", 303), ("The Co-openative Bank of Kenya Ltd Ord 1.00", 340),
                ("KE1000001568", 676), ("37.45", 911), ("36.05", 1008), ("37.10", 1118),
                ("37.35", 1258), ("707,829", 1366)]),
]
PAGE1 = HEADINGS + [b for row in PAGE1_ROWS for b in row]

PAGE2 = [b for row in [
    _line(120, [("629.00", 198), ("345.00British American Tobacco Kenya Plc Ord 10.00", 294),
                ("KE0000000075", 674), ("xd", 760), ("568.00", 905), ("555.00", 1001),
                ("561.00", 1112), ("561.00", 1252), ("3,011", 1382)]),
    _line(180, [("351.00", 198), ("167.00East African Breweries Plc Ord 2.00", 295),
                ("KE0000000216", 676), ("cd", 758), ("285.00", 906), ("281.00", 1004),
                ("281.75", 1112), ("282.00", 1251), ("22,038", 1373)]),
    _line(390, [("0966", 198), ("17.00Safaricom Plc Ord 0.05", 303), ("KE1000001402", 677),
                ("37.00", 911), ("36.00", 1008), ("36.20", 1117), ("36.60", 1258),
                ("2,501,236", 1359)]),
    # A dividend note: quotes KCB's ISIN inside running text, next to numbers.
    _line(1020, [("KCB Group Plc Ord 1.00: KE0000000315;announced an Interim Dividend of Kes.3.00 "
                  "on 13-Aug-2026:; Books Closure; 02-Sep-2026", 199), ("13,544", 1259),
                 ("12,699", 1381)]),
] for b in row]


# ------------------------------------------------------------------ reading

def test_list_names_zero_pad_the_day():
    """1-JUL-26.pdf is a 404; 01-JUL-26.pdf is the list."""
    assert pl.pricelist_name(date(2026, 7, 1)) == "01-JUL-26"
    assert pl.pricelist_name(date(2026, 9, 23)) == "23-SEP-26"


def test_the_column_headings_are_found():
    anchors = pl.find_anchors(pl.group_lines(HEADINGS))
    assert anchors == {"high": 877, "low": 981, "vwap": 1081, "previous": 1195, "volume": 1333}


def test_a_clean_row_reads_every_column():
    rows, _ = pl.read_page(PAGE1, WIDTH)
    assert rows["EQTY"] == {"high": 107.5, "low": 105.5, "vwap": 107.0, "previous": 105.0,
                            "volume": 1135601}
    # The xd status cell sits left of HIGH and is not a price.
    assert rows["ABSA"]["high"] == 33.3 and rows["ABSA"]["vwap"] == 33.0


def test_a_garbled_cell_reads_as_unreadable_not_as_a_number():
    rows, _ = pl.read_page(PAGE1, WIDTH)
    assert rows["KCB"]["vwap"] is None and rows["KCB"]["previous"] == 93.0
    assert rows["NCBA"] == {"high": None, "low": None, "vwap": None, "previous": 90.0,
                            "volume": 1167538}


@pytest.mark.parametrize("text", ["006", "0968", "00°06", "00000", "36.2", "3620", "36,20", "93.00x"])
def test_only_a_well_formed_price_is_a_price(text):
    assert pl._parse(text, "vwap") is None


@pytest.mark.parametrize("text,value", [("36.20", 36.2), ("5450.00", 5450.0), ("1,135.50", 1135.5),
                                        ("0.40", 0.4)])
def test_well_formed_prices_parse(text, value):
    assert pl._parse(text, "vwap") == value


def test_a_page_without_headings_uses_the_previous_pages_columns():
    _, anchors = pl.read_page(PAGE1, WIDTH)
    rows, _ = pl.read_page(PAGE2, WIDTH, anchors)
    assert rows["SCOM"] == {"high": 37.0, "low": 36.0, "vwap": 36.2, "previous": 36.6,
                            "volume": 2501236}
    assert rows["EABL"]["vwap"] == 281.75


def test_the_default_columns_read_the_observed_layout():
    rows, _ = pl.read_page(PAGE2, WIDTH)
    assert rows["SCOM"]["vwap"] == 36.2 and rows["BAT"]["previous"] == 561.0


def test_a_dividend_note_quoting_an_isin_is_not_a_price_row():
    rows, _ = pl.read_page(PAGE2, WIDTH)
    assert "KCB" not in rows


def test_two_rows_claiming_one_stock_are_both_distrusted():
    dup = PAGE1 + [Box(b.x, b.y + 900, b.height, b.text) for b in PAGE1_ROWS[1]]
    rows, _ = pl.read_page(dup, WIDTH)
    assert rows["EQTY"] == {c: None for c in pl.COLUMNS}


def test_ocr_boxes_are_grouped_into_printed_lines():
    boxes = [Box(10, 100, 20, "a"), Box(300, 104, 20, "b"), Box(10, 130, 20, "c")]
    assert [[b.text for b in line] for line in pl.group_lines(boxes)] == [["a", "b"], ["c"]]


def test_rapidocr_results_become_boxes():
    result = [[[[745.0, 66.9], [1231.9, 65.7], [1231.9, 93.9], [745.0, 95.1]], "36.20", 0.99]]
    (b,) = pl.boxes_from_ocr(result)
    assert (b.x, b.text) == (745.0, "36.20") and b.y == pytest.approx(80.4, abs=0.1)


# --------------------------------------------------------------- validation

def _r(high=None, low=None, vwap=None, previous=None, volume=1000):
    return {"high": high, "low": low, "vwap": vwap, "previous": previous, "volume": volume}


D1, D2, D3 = date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)


def _one(readings, sym="SCOM"):
    return {v.date: v for v in pl.validate(readings, [sym])}


def test_a_price_both_lists_agree_on_is_confirmed():
    v = _one({D2: {"SCOM": _r(37.0, 36.4, 36.6, 36.5)},
              D3: {"SCOM": _r(37.0, 36.0, 36.2, 36.6)}})[D2]
    assert v.how == "confirmed"
    assert (v.bar.close, v.bar.open, v.bar.high, v.bar.low, v.bar.source) == (36.6, 36.5, 37.0, 36.4, "nse_pricelist")
    assert v.bar.change_pct == pytest.approx(0.27, abs=0.01)


def test_two_readings_that_disagree_are_rejected():
    v = _one({D2: {"SCOM": _r(37.0, 36.0, 36.8, 36.5)},
              D3: {"SCOM": _r(37.0, 36.0, 36.2, 36.6)}})[D2]
    assert v.bar is None and "36.8 here but 36.6" in v.how


def test_an_unreadable_price_is_recovered_from_the_next_list_within_its_range():
    """KCB on 23 Sep: VWAP read as "006", range 92.50-93.25 readable."""
    rows, _ = pl.read_page(PAGE1, WIDTH)
    v = _one({D3: {"KCB": rows["KCB"]}, date(2026, 9, 24): {"KCB": _r(93.5, 92.75, 93.0, 93.0)}},
             "KCB")[D3]
    assert v.how == "range-checked" and v.bar.close == 93.0


def test_a_single_reading_with_nothing_to_check_it_against_is_rejected():
    """NCBA on 23 Sep: high, low and VWAP all garbled."""
    rows, _ = pl.read_page(PAGE1, WIDTH)
    v = _one({D3: {"NCBA": rows["NCBA"]}, date(2026, 9, 24): {"NCBA": _r(91.0, 89.0, 90.5, 89.6)}},
             "NCBA")[D3]
    assert v.bar is None and "no high/low that confirms it" in v.how


def test_the_newest_list_is_accepted_only_inside_its_own_range():
    ok = _one({D3: {"SCOM": _r(37.0, 36.0, 36.2, 36.6)}})[D3]
    assert ok.how == "range-checked" and ok.bar.close == 36.2
    bad = _one({D3: {"SCOM": _r(37.0, 36.0, 38.2, 36.6)}})[D3]
    assert bad.bar is None


def test_a_price_unreadable_everywhere_is_rejected():
    v = _one({D2: {"SCOM": _r(37.0, 36.0)}, D3: {"SCOM": _r(37.0, 36.0, 36.2)}})[D2]
    assert v.bar is None and "unreadable in both" in v.how


def test_a_missing_row_is_reported():
    v = _one({D3: {"EQTY": _r(1, 1, 1, 1)}})[D3]
    assert v.bar is None and "row not found" in v.how


def test_a_move_beyond_a_sessions_limit_is_rejected():
    """Both readings agree on 3.66 (a dropped digit read twice alike)."""
    v = _one({D2: {"SCOM": _r(None, None, 3.66, 36.5)}, D3: {"SCOM": _r(37, 36, 36.2, 3.66)}})[D2]
    assert v.bar is None and "beyond a session's limit" in v.how


def test_an_implausible_high_or_low_does_not_shape_the_bar():
    v = _one({D2: {"SCOM": _r(3.72, 36.4, 36.6, 36.5)}, D3: {"SCOM": _r(37, 36, 36.2, 36.6)}})[D2]
    assert v.how == "confirmed" and (v.bar.high, v.bar.low) == (36.6, 36.5)


def test_a_gap_in_the_lists_cannot_confirm_across_it():
    """With 22 Sep missing, 23 Sep's PREVIOUS is 22 Sep's price, not 21 Sep's."""
    v = _one({D1: {"SCOM": _r(None, None, 36.5, 36.4)}, D3: {"SCOM": _r(37, 36, 36.2, 36.6)}})[D1]
    assert v.bar is None


# ------------------------------------------------------------------ storage

def _csv(folder, symbol, rows):
    fields = ["date", "symbol", "open", "high", "low", "close", "volume", "change_pct", "source"]
    with open(folder / f"{symbol}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"symbol": symbol, **r})


def _rows(folder, symbol):
    with open(folder / f"{symbol}.csv", newline="") as f:
        return {r["date"]: r for r in csv.DictReader(f)}


def test_the_backfill_replaces_seed_data_but_never_a_live_price(tmp_path, monkeypatch):
    monkeypatch.setattr(nse_scraper, "DATA_DIR", tmp_path)
    _csv(tmp_path, "SCOM", [{"date": "2026-09-22", "close": 99, "source": "synthetic"},
                            {"date": "2026-09-23", "close": 36.25, "source": "nse_ticker"}])
    bars = [v.bar for v in pl.validate({D2: {"SCOM": _r(37, 36.4, 36.6, 36.5)},
                                        D3: {"SCOM": _r(37, 36, 36.2, 36.6)}}, ["SCOM"])]
    assert pl.merge_into_csv("SCOM", bars) == (1, 1)
    rows = _rows(tmp_path, "SCOM")
    assert (rows["2026-09-22"]["close"], rows["2026-09-22"]["source"]) == ("36.6", "nse_pricelist")
    assert (rows["2026-09-23"]["close"], rows["2026-09-23"]["source"]) == ("36.25", "nse_ticker")


def test_backfilled_bars_count_as_real_history():
    assert "nse_pricelist" in nse_scraper.REAL_NSE_SOURCES
    assert history_warmstart.REAL_NSE_SOURCES is nse_scraper.REAL_NSE_SOURCES


# ------------------------------------------------------------------- script

def _row(y, isin, high, low, vwap, prev, vol):
    return _line(y, [(isin, 676), (high, 906), (low, 1004), (vwap, 1114), (prev, 1255), (vol, 1362)])


def _cache(folder, d, *pages):
    folder.mkdir(exist_ok=True)
    entry = {"url": "x", "status": 200,
             "pages": [{"width": WIDTH, "boxes": [list(b) for b in boxes]} for boxes in pages]}
    (folder / f"{d.isoformat()}.json").write_text(json.dumps(entry))


def test_the_script_checks_writes_and_reports_from_its_cache(tmp_path, monkeypatch, capsys):
    """A cached run needs no OCR engine and downloads nothing."""
    from scripts import backfill_nse_pricelists as script
    cache, prices = tmp_path / "cache", tmp_path / "nse_historical"
    prices.mkdir()
    monkeypatch.setattr(script, "CACHE_DIR", cache)
    monkeypatch.setattr(nse_scraper, "DATA_DIR", prices)
    monkeypatch.setattr(script.Reader, "engine", lambda self: pytest.fail("OCR engine loaded"))
    _cache(cache, D2, HEADINGS + _row(600, "KE1000001402", "37.00", "36.40", "36.60", "36.50", "1,000"))
    _cache(cache, D3, PAGE1, PAGE2)
    monkeypatch.setattr(sys, "argv", ["x", "--end", "2026-09-23", "--days", "2", "--write"])
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "SCOM" in out and "2026-09-23 36.20" in out
    rows = _rows(prices, "SCOM")
    assert rows["2026-09-22"]["close"] == "36.6" and rows["2026-09-23"]["close"] == "36.2"
    assert (prices / "EQTY.csv").exists()


def test_without_write_the_script_stores_nothing(tmp_path, monkeypatch, capsys):
    from scripts import backfill_nse_pricelists as script
    cache, prices = tmp_path / "cache", tmp_path / "nse_historical"
    prices.mkdir()
    monkeypatch.setattr(script, "CACHE_DIR", cache)
    monkeypatch.setattr(nse_scraper, "DATA_DIR", prices)
    _cache(cache, D3, PAGE1, PAGE2)
    monkeypatch.setattr(sys, "argv", ["x", "--end", "2026-09-23", "--days", "1"])
    assert script.main() == 0
    assert "Nothing written" in capsys.readouterr().out
    assert list(prices.iterdir()) == []


def test_sessions_are_weekdays_ending_at_the_given_day():
    from scripts.backfill_nse_pricelists import sessions
    assert sessions(date(2026, 9, 28), 3) == [date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)]
