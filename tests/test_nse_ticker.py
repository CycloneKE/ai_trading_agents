"""The NSE's own ticker feed as the live NSE price source.

Both earlier sources died: the www.nse.co.ke market-statistics pages return
404 since the site was rebuilt, and afx.kwayisi.org is unreachable from the
server. Every NSE price was therefore synthetic seed data and the agent
rightly refused to evaluate any of them. The ticker on nse.co.ke's home page
reads a JSON feed; these tests pin the reply shape it was observed to return
and the account handshake it needs.
"""
import csv
import json
import logging
from datetime import date

import pytest

import src.connectors.nse_scraper as nse_scraper
from src.agent import history_warmstart
from src.connectors.nse_scraper import parse_ticker_reply

THURSDAY = date(2026, 9, 24)

# Captured from the feed on 24 September 2026 (SCOM and KCB rows verbatim).
# SCOM's today_open (37.2) is above its today_high (37): the feed's own
# open/high/low are not always consistent with each other.
REPLY = {"message": [
    {"snapshot": [
        {"issuer": "SCOM", "price": 36.2, "ltp": 36.2, "prev_price": 36.6, "today_open": 37.2,
         "today_high": 37, "today_low": 36, "turnover": 90592026.15, "volume": 2501236,
         "change": -1.09, "today_close": 36.2},
        {"issuer": "KCB", "price": 93, "ltp": 93, "prev_price": 93, "today_open": 93.5,
         "today_high": 93.25, "today_low": 92.5, "turnover": 69121078.5, "volume": 743476,
         "change": 0, "today_close": 93},
        {"issuer": "EQTY", "price": 107, "ltp": 107, "prev_price": 105, "today_open": 105.5,
         "today_high": 108, "today_low": 105, "volume": 410000, "change": 1.9},
        {"issuer": "HAFR", "price": 0.4, "prev_price": 0.4},
    ]},
    {"updated_at": {"date": "24/09/2026", "time": "00:00 AM GMT+3", "market_status": "open"}},
]}


def _reply(date_str="24/09/2026", rows=None):
    r = json.loads(json.dumps(REPLY))
    r["message"][1]["updated_at"]["date"] = date_str
    if rows is not None:
        r["message"][0]["snapshot"] = rows
    return r


# ------------------------------------------------------------------ parsing

def test_the_observed_reply_becomes_one_real_bar_per_watched_symbol():
    bars = parse_ticker_reply(REPLY, ["SCOM", "EQTY", "KCB", "COOP"], THURSDAY)
    assert sorted(bars) == ["EQTY", "KCB", "SCOM"]  # COOP absent from this reply; HAFR unwatched
    scom = bars["SCOM"]
    assert (scom.date, scom.close, scom.volume, scom.source) == ("2026-09-24", 36.2, 2501236, "nse_ticker")
    assert scom.change_pct == pytest.approx(-1.09, abs=0.01)


def test_an_inconsistent_open_high_low_still_makes_a_valid_bar():
    scom = parse_ticker_reply(REPLY, ["SCOM"], THURSDAY)["SCOM"]
    assert scom.low <= min(scom.open, scom.close)
    assert scom.high >= max(scom.open, scom.close)
    assert (scom.open, scom.high, scom.low) == (37.2, 37.2, 36.0)


def test_a_price_beyond_a_sessions_reach_is_not_stored(caplog):
    rows = [{"issuer": "SCOM", "price": 362, "prev_price": 36.6, "volume": 10}]
    with caplog.at_level(logging.WARNING):
        assert parse_ticker_reply(_reply(rows=rows), ["SCOM"], THURSDAY) == {}
    assert any("previous close" in r.getMessage() for r in caplog.records)


def test_a_misread_open_is_replaced_by_the_previous_close():
    rows = [{"issuer": "SCOM", "price": 36.2, "prev_price": 36.6, "today_open": 3.72,
             "today_high": 37, "today_low": 36}]
    scom = parse_ticker_reply(_reply(rows=rows), ["SCOM"], THURSDAY)["SCOM"]
    assert scom.open == 36.6


def test_a_zero_or_missing_price_is_skipped():
    rows = [{"issuer": "SCOM", "price": 0, "ltp": None, "prev_price": 36.6},
            {"issuer": "KCB", "prev_price": 93}]
    assert parse_ticker_reply(_reply(rows=rows), ["SCOM", "KCB"], THURSDAY) == {}


def test_the_last_traded_price_is_used_when_price_is_absent():
    rows = [{"issuer": "SCOM", "ltp": "36.20", "prev_price": "36.60"}]
    assert parse_ticker_reply(_reply(rows=rows), ["SCOM"], THURSDAY)["SCOM"].close == 36.2


def test_a_weekend_dated_reply_stores_nothing():
    """A Saturday bar would be a flat copy of Friday counted as a session."""
    assert parse_ticker_reply(_reply("26/09/2026"), ["SCOM"], date(2026, 9, 26)) == {}


def test_the_bar_takes_the_feeds_session_date_not_the_clock():
    bars = parse_ticker_reply(_reply("25/09/2026"), ["SCOM"], date(2026, 9, 27))
    assert bars["SCOM"].date == "2026-09-25"


def test_a_feed_stuck_on_an_old_date_stores_nothing(caplog):
    """Storing under the stuck date would overwrite a past session's bar
    with today's price."""
    with caplog.at_level(logging.WARNING):
        assert parse_ticker_reply(_reply("01/07/2026"), ["SCOM"], THURSDAY) == {}
    assert any("may be stuck" in r.getMessage() for r in caplog.records)


def test_a_long_holiday_weekend_is_not_mistaken_for_a_stuck_feed():
    # Easter 2027: Thursday 25 March is the last session before Tuesday 30 March.
    bars = parse_ticker_reply(_reply("25/03/2027"), ["SCOM"], date(2027, 3, 30))
    assert bars["SCOM"].date == "2027-03-25"


@pytest.mark.parametrize("raw", ["30/09/2026", "not a date", None])
def test_a_future_or_unreadable_date_falls_back_to_today(raw):
    bars = parse_ticker_reply(_reply(raw), ["SCOM"], THURSDAY)
    assert bars["SCOM"].date == "2026-09-24"


@pytest.mark.parametrize("payload", [{}, {"message": "isinno is required!"},
                                     {"message": [{"updated_at": {}}]}, [], None])
def test_a_reply_without_a_snapshot_is_flagged(payload, caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_ticker_reply(payload, ["SCOM"], THURSDAY) == {}
    assert any("format may have changed" in r.getMessage() for r in caplog.records)


def test_a_snapshot_naming_none_of_the_watched_symbols_is_flagged(caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_ticker_reply(REPLY, ["ZZZZ"], THURSDAY) == {}
    assert any("none of the watched symbols" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------ the account handshake

HOME = ('<div data-background="#303d4a" data-account="KE3000009674" '
        'data-issuer="SCOM" id="nseticker3"></div>')


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class _FakeNet:
    """The NSE home page and ticker feed, recording what was asked."""

    def __init__(self, home=HOME, valid="KE3000009674"):
        self.home, self.valid = home, valid
        self.home_reads, self.posts = 0, []

    def get(self, url, **kw):
        assert url == nse_scraper.NSE_HOME_URL
        self.home_reads += 1
        return _Resp(200, self.home)

    def post(self, url, headers=None, data=None, **kw):
        assert url == nse_scraper.NSE_TICKER_URL
        assert headers["Content-Type"] == "application/json"
        body = json.loads(data)
        self.posts.append(body)
        if body.get("isinno") != self.valid:
            return _Resp(400, {"message": "Invalid account"})
        return _Resp(200, REPLY)


@pytest.fixture
def net(monkeypatch):
    fake = _FakeNet()
    monkeypatch.setattr(nse_scraper, "_ticker_account", {})
    monkeypatch.setattr(nse_scraper.requests, "get", fake.get)
    monkeypatch.setattr(nse_scraper.requests, "post", fake.post)
    monkeypatch.setattr(nse_scraper, "_eat_now",
                        lambda: nse_scraper.datetime(2026, 9, 24, 11, 0, tzinfo=nse_scraper.EAT))
    return fake


def test_the_feed_is_asked_the_way_the_nse_ticker_asks_it(net):
    bars = nse_scraper.scrape_nse_ticker(["SCOM", "KCB"])
    assert sorted(bars) == ["KCB", "SCOM"]
    assert net.posts == [{"nopage": "true", "isinno": "KE3000009674"}]


def test_the_account_is_read_from_the_home_page_once_a_day_not_every_cycle(net):
    nse_scraper.scrape_nse_ticker(["SCOM"])
    nse_scraper.scrape_nse_ticker(["SCOM"])
    assert net.home_reads == 1 and len(net.posts) == 2


def test_a_changed_account_is_picked_up_from_the_home_page(net):
    nse_scraper.scrape_nse_ticker(["SCOM"])
    net.valid = "KE9999999999"
    net.home = HOME.replace("KE3000009674", "KE9999999999")
    bars = nse_scraper.scrape_nse_ticker(["SCOM"])
    assert bars["SCOM"].close == 36.2
    assert [p["isinno"] for p in net.posts] == ["KE3000009674", "KE3000009674", "KE9999999999"]


def test_a_home_page_without_the_ticker_is_flagged(net, caplog):
    net.home = "<html>redesigned</html>"
    with caplog.at_level(logging.WARNING):
        assert nse_scraper.scrape_nse_ticker(["SCOM"]) == {}
    assert net.posts == []
    assert any("no ticker data-account" in r.getMessage() for r in caplog.records)


def test_a_rejected_request_is_logged_with_the_feeds_reason(net, caplog):
    net.valid = "never"
    with caplog.at_level(logging.WARNING):
        assert nse_scraper.scrape_nse_ticker(["SCOM"]) == {}
    assert any("HTTP 400" in r.getMessage() and "Invalid account" in r.getMessage()
               for r in caplog.records)


def test_a_scrape_cycle_stores_ticker_prices_as_real(net, monkeypatch, tmp_path):
    monkeypatch.setattr(nse_scraper, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nse_scraper, "scrape_afx_kwayisi", lambda syms: {})
    s = nse_scraper.NSEPeriodicScraper(symbols=["SCOM", "KCB"])
    assert s.run_once() == {"SCOM": 1, "KCB": 1}
    assert s.get_status()["last_cycle_real_prices"] == 2
    with open(tmp_path / "SCOM.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert (row["date"], row["close"], row["source"]) == ("2026-09-24", "36.2", "nse_ticker")


def test_afx_is_asked_only_for_symbols_the_ticker_lacked(net, monkeypatch, tmp_path):
    monkeypatch.setattr(nse_scraper, "DATA_DIR", tmp_path)
    asked = []
    monkeypatch.setattr(nse_scraper, "scrape_afx_kwayisi", lambda syms: asked.append(syms) or {})
    nse_scraper.NSEPeriodicScraper(symbols=["SCOM", "COOP"]).run_once()
    assert asked == [["COOP"]]


# ------------------------------------------------ one list of real sources

def test_every_consumer_judges_nse_prices_by_the_same_list():
    from src.api import api_server
    assert "nse_ticker" in nse_scraper.REAL_NSE_SOURCES
    assert api_server.REAL_NSE_SOURCES is nse_scraper.REAL_NSE_SOURCES
    assert history_warmstart.REAL_NSE_SOURCES is nse_scraper.REAL_NSE_SOURCES


def test_ticker_bars_count_as_history_for_the_warm_start(tmp_path):
    fields = ["date", "symbol", "open", "high", "low", "close", "volume", "change_pct", "source"]
    with open(tmp_path / "SCOM.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"date": "2026-09-22", "symbol": "SCOM", "close": 99, "source": "synthetic"})
        w.writerow({"date": "2026-09-23", "symbol": "SCOM", "open": 36.6, "high": 36.9,
                    "low": 36.1, "close": 36.6, "volume": 1, "change_pct": 0, "source": "nse_ticker"})
    hist = history_warmstart.read_nse_history(["SCOM"], tmp_path, today=THURSDAY)
    assert hist["SCOM"]["close"] == [36.6]
