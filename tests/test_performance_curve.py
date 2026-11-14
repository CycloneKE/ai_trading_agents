"""The account-value curve and the ratios measured on it
(performance_analytics.py, chart_data.thin_curve, /api/performance).

Before: the agent values the account every minute, and
- volatility, Sharpe, Sortino and the annualised return treated each
  minute's change as a day's (a Sharpe of -4.9 or +41 on an ordinary month),
  and Sortino multiplied by the square root of 252 twice;
- the history was capped at 365 x 24 readings, about six days at one a
  minute, so the chart's 1M, 1Y and ALL could never show more;
- /api/performance sent every reading (3.5 MB for 60 days, every 20 s).
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.agent import performance_analytics as pa_mod
from src.agent.chart_data import thin_curve
from src.agent.performance_analytics import PerformanceAnalytics

START = datetime(2026, 6, 1, 0, 0)


def _analytics(tmp_path, monkeypatch):
    monkeypatch.setattr(pa_mod, 'DATA_DIR', tmp_path)
    return PerformanceAnalytics({'risk_free_rate': 0.0})


def _record_minutes(a, daily_values, per_day=24 * 60 // 30):
    """Readings every 30 minutes, drifting within each day to its close."""
    prev = daily_values[0]
    for d, close in enumerate(daily_values):
        for i in range(per_day):
            t = START + timedelta(days=d, minutes=30 * i)
            a.portfolio_values.append({'timestamp': t, 'value': prev + (close - prev) * (i + 1) / per_day})
            if len(a.portfolio_values) > 1:
                p = a.portfolio_values[-2]['value']
                a.returns_history.append({'timestamp': t, 'return': a.portfolio_values[-1]['value'] / p - 1})
        prev = close


def test_ratios_are_measured_on_daily_returns(tmp_path, monkeypatch):
    a = _analytics(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    closes = list(100000 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)))
    _record_minutes(a, closes)
    m = a.calculate_performance_metrics()
    daily = np.diff(closes) / closes[:-1]
    per_year = 365                                   # readings on weekends too
    assert m.volatility == pytest.approx(np.std(daily, ddof=1) * math.sqrt(per_year), rel=1e-6)
    assert m.sharpe_ratio == pytest.approx(np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(per_year), rel=1e-6)
    assert abs(m.sharpe_ratio) < 10                  # not a per-minute figure
    assert m.total_return == pytest.approx(closes[-1] / closes[0] - 1, rel=1e-6)
    assert a.days_measured() == 59


def test_too_few_days_give_no_ratios(tmp_path, monkeypatch):
    a = _analytics(tmp_path, monkeypatch)
    _record_minutes(a, [100000, 101000, 99000, 102000])
    m = a.calculate_performance_metrics()
    assert (m.sharpe_ratio, m.sortino_ratio, m.annualized_return, m.volatility) == (0, 0, 0, 0)
    assert m.total_return == pytest.approx(0.02)
    report = a.generate_performance_report('ALL')
    assert report['metrics']['days_measured'] == 3 and report['metrics']['min_days_for_ratios'] == 20


def test_history_is_kept_by_time_not_by_count(tmp_path, monkeypatch):
    a = _analytics(tmp_path, monkeypatch)
    a.portfolio_values = [{'timestamp': START + timedelta(minutes=i), 'value': 100000.0}
                          for i in range(20 * 1440)]   # 20 days, one reading a minute
    a.record_portfolio_value(100000, START + timedelta(days=20))
    assert a.portfolio_values[0]['timestamp'] == START          # the old cap kept ~6 days
    a.config['max_history_days'] = 7
    a.record_portfolio_value(100000, START + timedelta(days=20))
    assert a.portfolio_values[0]['timestamp'] >= START + timedelta(days=13)


def test_the_curve_sent_to_the_chart_is_thinned_by_age():
    now = datetime(2026, 9, 1, 12, 0)
    points = [{'timestamp': (now - timedelta(minutes=m)).isoformat(), 'value': 100000 + m}
              for m in range(90 * 1440, -1, -1)]               # 90 days a minute apart
    thin = thin_curve(points)
    assert len(points) > 129000 and 1000 < len(thin) < 2000
    assert thin[0] is points[0] and thin[-1] is points[-1]   # first and latest kept
    stamps = [datetime.fromisoformat(p['timestamp']) for p in thin]
    assert stamps == sorted(stamps)
    gaps = {d: [] for d in ('day', 'hour', 'recent')}
    for a, b in zip(stamps, stamps[1:]):
        age = now - b
        gaps['recent' if age <= timedelta(days=2) else 'hour' if age <= timedelta(days=30) else 'day'].append(b - a)
    assert max(gaps['recent']) <= timedelta(minutes=5)
    assert max(gaps['hour'][1:]) <= timedelta(hours=1)
    assert max(gaps['day'][1:]) <= timedelta(days=1)
    assert thin_curve(points[:2]) == points[:2] and thin_curve([]) == []
