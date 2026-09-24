"""Live prices must build daily bars, because that is what the strategies,
the regime detector and the ATR tracker were written and backtested on.

In production the loop called them once a minute and every call appended a
bar, so "the last 50 bars" meant fifty minutes. Momentum needed a 2% move
inside ten minutes and mean reversion a close 2% from its twenty-minute
average; the agent produced no trade signal in 36 hours. Overnight, frozen
US quotes were appended hundreds of times, flattening every indicator.
"""
import json
import logging
import math
import random
from datetime import date

import pytest

from src.agent.daily_bars import APPEND, REPLACE, SKIP, classify, utc_bar_date
from src.agent.regime import RegimeDetector
from src.agent.strategy_manager import StrategyManager
from src.agent.technical_strategy import TechnicalStrategy
from src.agent.volatility import VolatilityTracker


# ------------------------------------------------------------------ rule

def test_the_rule():
    assert classify(None, None, 100.0, 100.0) == APPEND        # no date: one bar per call
    assert classify('2026-09-23', '2026-09-23', 100.0, 101.0) == REPLACE
    assert classify('2026-09-23', '2026-09-24', 100.0, 101.0) == APPEND
    assert classify('2026-09-23', '2026-09-24', 100.0, 100.0) == SKIP   # frozen quote
    assert classify(None, '2026-09-24', 100.0, 101.0) == APPEND  # first live bar after seeding
    assert classify(None, '2026-09-24', 100.0, 100.0) == SKIP    # weekend after seeding


def test_utc_bar_date_format():
    assert len(utc_bar_date()) == 10 and utc_bar_date()[4] == '-'


# ---------------------------------------------------------- strategies

def _strategy():
    return TechnicalStrategy('momentum', {'lookback_period': 50, 'threshold': 0.02})


def test_a_thousand_quotes_in_one_day_are_one_bar():
    s = _strategy()
    s.seed_history('BTC-USD', [100.0 + i * 0.1 for i in range(60)])
    before = len(s.historical_data['BTC-USD'])
    for i in range(1000):
        s.generate_signals({'symbol': 'BTC-USD', 'price': 110.0 + i * 0.001,
                            'bar_date': '2026-09-24'})
    # 50 kept after the first trim; the day's thousand quotes are one bar,
    # holding the latest price, not a thousand bars.
    assert len(s.historical_data['BTC-USD']) == min(before + 1, 50)
    assert s.historical_data['BTC-USD'][-1] == pytest.approx(110.999)


def test_a_frozen_price_on_a_new_date_adds_no_bar():
    s = _strategy()
    s.seed_history('AAPL', [100.0] * 49 + [190.0])
    for d in ('2026-09-26', '2026-09-27'):          # the weekend
        s.generate_signals({'symbol': 'AAPL', 'price': 190.0, 'bar_date': d})
    assert len(s.historical_data['AAPL']) == 50 and s.historical_data['AAPL'][-2] == 100.0


def test_without_a_date_every_call_is_a_bar_as_before():
    """Backtests feed one bar per call and must be unchanged."""
    s = _strategy()
    for p in range(10):
        s.generate_signals({'symbol': 'X', 'price': 100.0 + p})
    assert len(s.historical_data['X']) == 10


# -------------------------------------------------------- regime and ATR

def test_regime_detector_builds_daily_bars():
    rd = RegimeDetector(fast_period=2, slow_period=3)
    for p in (100, 101, 102):
        rd.update('X', p, bar_date='2026-09-24')
    assert list(rd._closes['X']) == [102.0]


@pytest.mark.parametrize('with_range', [False, True])
def test_minute_quotes_give_daily_atr_not_minute_atr(with_range):
    """Fed per minute, Bitcoin measured 0.06% volatility instead of ~2% a
    day, pinning every stop to the 2% floor. Minute quotes with a bar_date
    must give exactly the ATR of the equivalent daily bars, whether the feed
    carries a high/low or only a price."""
    rnd = random.Random(7)
    per_min = 0.55 / math.sqrt(525600)
    daily = VolatilityTracker(length=14)
    minute = VolatilityTracker(length=14)
    px = 63000.0
    for day in range(40):
        d = f'2026-08-{day + 1:02d}' if day < 31 else f'2026-09-{day - 30:02d}'
        closes = []
        for _ in range(1440):
            px *= math.exp(rnd.gauss(0, per_min))
            closes.append(px)
            if with_range:
                minute.update('BTC', px, px, px, bar_date=d)
            else:
                minute.update('BTC', px, bar_date=d)
        if with_range:
            daily.update('BTC', closes[-1], max(closes), min(closes))
        else:
            daily.update('BTC', closes[-1])
    assert minute.atr_pct('BTC') == pytest.approx(daily.atr_pct('BTC'), rel=1e-9)
    assert minute.atr_pct('BTC') > 0.01      # a daily figure, not ~0.0006


def test_warm_starts_replace_rather_than_append():
    """A retried warm-start must not land after live bars."""
    v = VolatilityTracker(length=3)
    v.update('X', 5.0, bar_date='2026-09-24')
    v.warm_start('X', [1.0, 2.0, 3.0])
    assert list(v._close['X']) == [1.0, 2.0, 3.0]
    rd = RegimeDetector(fast_period=2, slow_period=3)
    rd.update('X', 5.0, bar_date='2026-09-24')
    rd.warm_start('X', [1.0, 2.0, 3.0])
    assert list(rd._closes['X']) == [1.0, 2.0, 3.0]


# ------------------------------------------------------ the whole thing

def _cfg():
    from src.utils.config_validator import load_config
    return load_config('config/config.json')


def test_production_feed_now_signals_the_way_the_backtest_did():
    """End to end with the real config: warm-start 60 daily closes, then a
    day of minute quotes with a bar_date. A 6% up-day on a quiet series is a
    textbook momentum break and must produce a trade signal."""
    sm = StrategyManager(_cfg())
    rnd = random.Random(11)
    hist, px = [], 100.0
    for _ in range(60):
        px *= math.exp(rnd.gauss(0, 0.004)); hist.append(px)
    sm.warm_start({'SPY': hist})
    actions = set()
    for m in range(390):
        price = hist[-1] * (1 + 0.06 * (m + 1) / 390)
        sig = sm.generate_signals({'symbol': 'SPY', 'price': price, 'close': price,
                                   'bar_date': '2026-09-24'})
        actions.add(sig.get('action'))
    assert 'buy' in actions


def test_the_old_production_feed_could_not_signal():
    """The failure itself: same strategies, a minute feed with no bar_date
    and no warm-start, a day of ordinary prices. No trade signal at all."""
    sm = StrategyManager(_cfg())
    rnd = random.Random(1)
    px = 63000.0
    per_min = 0.55 / math.sqrt(525600)
    actions = set()
    for _ in range(1440):
        px *= math.exp(rnd.gauss(0, per_min))
        actions.add(sm.generate_signals({'symbol': 'BTC-USD', 'price': px, 'close': px}).get('action'))
    assert actions == {'hold'}
