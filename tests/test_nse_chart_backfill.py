"""NSE chart backfill: afx's chart endpoint carries ~10y of real daily closes,
used to give strategies their 50-bar lookback instead of leaving them blind.
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connectors import nse_scraper as ns
from src.connectors.nse_scraper import (
    DailyBar, build_chart_bars, parse_afx_chart, save_bars_csv, load_csv,
    backfill_afx_chart_history,
)


def _chart_js(points, name="SCOM"):
    data = ",".join(f'[d("{d}"),{c}]' for d, c in points)
    return ("$(function(){var d=function(d){return Date.parse(d)};"
            f"$('.h2:first+*').highcharts('StockChart',{{series:[{{name:'{name}',"
            f"tooltip:{{valueDecimals:2}},data:[{data},]}}]}});}})")


def _weekdays(end, n):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _points(end, n, start_price=20.0):
    return [(d.isoformat(), round(start_price + i * 0.1, 2))
            for i, d in enumerate(_weekdays(end, n))]


def test_parse_afx_chart_extracts_date_close_pairs():
    pts = [("2026-10-06", 36.3), ("2026-10-07", 36.05), ("2026-10-08", 35.85)]
    assert parse_afx_chart(_chart_js(pts)) == pts


def test_parse_afx_chart_garbage_returns_empty():
    assert parse_afx_chart("<html>not a chart</html>") == []


def test_build_chart_bars_keeps_last_n_with_change_pct():
    today = date(2026, 10, 8)
    pts = _points(today, 200)
    bars = build_chart_bars("SCOM", pts, days=120, today=today)
    assert len(bars) == 120
    assert bars[-1].date == pts[-1][0]
    assert all(b.source == "afx_chart" and b.symbol == "SCOM" for b in bars)
    b = bars[-1]
    assert b.open == b.high == b.low == b.close == pts[-1][1]
    prev = pts[-2][1]
    assert b.change_pct == round((pts[-1][1] - prev) / prev * 100, 2)


def test_build_chart_bars_skips_stale_series():
    today = date(2026, 10, 8)
    stale = _points(date(2025, 5, 20), 200)  # BAMB: stops mid-2025
    assert build_chart_bars("BAMB", stale, days=120, today=today) == []


def test_build_chart_bars_drops_nonpositive_closes():
    today = date(2026, 10, 8)
    pts = _points(today, 10)
    pts[3] = (pts[3][0], 0.0)
    bars = build_chart_bars("SCOM", pts, days=120, today=today)
    assert len(bars) == 9


def _bar(d, close, source, volume=0):
    return DailyBar(date=d, symbol="SCOM", open=close, high=close, low=close,
                    close=close, volume=volume, change_pct=0.0, source=source)


def test_save_without_overwrite_keeps_existing_real_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "DATA_DIR", tmp_path)
    save_bars_csv("SCOM", [_bar("2026-10-07", 36.0, "afx_history", volume=500)])
    save_bars_csv("SCOM", [_bar("2026-10-07", 99.0, "afx_chart"),
                           _bar("2026-10-06", 35.0, "afx_chart")], overwrite=False)
    rows = {r["date"]: r for r in load_csv("SCOM")}
    assert rows["2026-10-07"]["close"] == "36.0" and rows["2026-10-07"]["volume"] == "500"
    assert rows["2026-10-06"]["source"] == "afx_chart"


def test_save_without_overwrite_replaces_synthetic_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "DATA_DIR", tmp_path)
    save_bars_csv("SCOM", [_bar("2026-10-07", 12.0, "synthetic")])
    save_bars_csv("SCOM", [_bar("2026-10-07", 36.0, "afx_chart")], overwrite=False)
    assert load_csv("SCOM")[0]["close"] == "36.0"


def test_backfill_gives_strategy_50_real_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "DATA_DIR", tmp_path)
    today = date.today()
    # Pre-existing: only the latest 18 real bars, as in production.
    existing = [DailyBar(date=d, symbol="SCOM", open=1, high=1, low=1, close=c,
                         volume=1, change_pct=0.0, source="afx_history")
                for d, c in _points(today, 18)]
    save_bars_csv("SCOM", existing)

    class Resp:
        status_code = 200
        text = _chart_js(_points(today, 300))

    monkeypatch.setattr(ns.requests, "get", lambda *a, **k: Resp())
    added = backfill_afx_chart_history(["SCOM"], days=120)

    rows = load_csv("SCOM")
    assert added == {"SCOM": 120 - 18}
    assert len(rows) == 120
    assert len([r for r in rows if r["close"] and float(r["close"]) > 0]) >= 50


def test_backfill_survives_fetch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "DATA_DIR", tmp_path)

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(ns.requests, "get", boom)
    assert backfill_afx_chart_history(["SCOM"], days=120) == {}
