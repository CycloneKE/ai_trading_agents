"""The pullback sleeve: the RSI(2) rule, and its daily decision near the US
close in the agent."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.agent import pullback_sleeve as pb
from src.agent.main import TradingAgent
from tests.test_momentum_sleeve import _Broker, _agent

CFG = pb.PullbackConfig(enabled=True)
NY = ZoneInfo('America/New_York')
CLOSE_WINDOW = datetime(2026, 12, 8, 15, 50, tzinfo=NY)        # a Tuesday, 15:50 New York
MORNING = datetime(2026, 12, 8, 11, 0, tzinfo=NY)


def _uptrend_then_dip(n=260):
    """A steady rise, then three sharp down days: above the 200-day average,
    RSI(2) near zero."""
    up = [100 * (1.001 ** i) for i in range(n)]
    return up + [up[-1] * 0.985, up[-1] * 0.97, up[-1] * 0.955]


# ------------------------------------------------------------------ the rule

def test_rsi_matches_the_pandas_formula_the_research_used():
    rng = np.random.default_rng(1)
    px = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    s = pd.Series(px)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    expected = float((100 - 100 / (1 + up / down)).iloc[-1])
    assert pb.rsi(px, 2) == pytest.approx(expected, rel=1e-9)


def test_a_dip_in_an_uptrend_is_a_buy_and_the_bounce_is_a_sell():
    closes = _uptrend_then_dip()
    d = pb.decide(closes, holding=False, cfg=CFG)
    assert d['action'] == 'buy' and d['rsi'] < 10 and d['close'] > d['sma_trend']
    bounced = closes + [closes[-1] * 1.04]
    assert pb.decide(bounced, holding=True, cfg=CFG)['action'] == 'sell'
    assert pb.decide(closes, holding=True, cfg=CFG)['action'] is None      # still below the 5-day average


def test_no_buy_in_a_downtrend_however_sharp_the_dip():
    down = [100 * (0.999 ** i) for i in range(260)]
    down += [down[-1] * 0.97, down[-1] * 0.94]
    assert pb.decide(down, holding=False, cfg=CFG)['action'] is None


def test_too_little_history_decides_nothing():
    assert pb.decide([100.0] * 50, holding=False, cfg=CFG)['action'] is None


def test_the_window_is_the_last_minutes_of_a_weekday_session():
    assert pb.in_window(CLOSE_WINDOW, CFG)
    assert not pb.in_window(MORNING, CFG)
    assert not pb.in_window(datetime(2026, 12, 8, 16, 0, tzinfo=NY), CFG)
    assert not pb.in_window(datetime(2026, 12, 12, 15, 50, tzinfo=NY), CFG)    # Saturday


def test_the_sleeve_stays_off_if_another_book_trades_its_fund():
    assert not pb.config_from_dict({'enabled': True, 'symbol': 'SPY'}, ['SPY', 'AAPL']).enabled
    assert pb.config_from_dict({'enabled': True}, ['SPY', 'VOO']).symbol == 'IVV'


def test_the_shipped_config_runs_it_on_a_fund_no_other_book_holds():
    with open('config/config.json') as f:
        cfg = json.load(f)
    dm = cfg['data_manager']
    taken = set(dm['symbols']) | set(cfg['core_satellite']['core_symbols']) | set(cfg['momentum_sleeve']['universe'])
    sleeve = pb.config_from_dict(cfg['pullback_sleeve'], taken)
    assert sleeve.enabled and sleeve.symbol == 'IVV' and sleeve.symbol not in taken


# ------------------------------------------------------------------ in the agent

def _history(closes):
    return lambda symbols, bars: {s: {'close': closes} for s in symbols}


def _run(agent, now, closes, live):
    agent._pullback_last_check = 0.0
    return TradingAgent._run_pullback_cycle(agent, now=now, fetch_history=_history(closes),
                                            price_for=lambda s: live)


def test_near_the_close_a_dip_buys_the_sleeves_share_once(tmp_path, monkeypatch):
    broker = _Broker()
    agent = _agent(tmp_path, monkeypatch, broker)
    closes = _uptrend_then_dip()
    live = closes.pop()                                   # today's price is the live one
    placed = _run(agent, CLOSE_WINDOW, closes, live)
    assert [(o['symbol'], o['side']) for o in placed] == [('IVV', 'buy')]
    # 100,000 equity x 0.55 active x 0.15 sleeve = 8,250.
    assert placed[0]['notional'] == pytest.approx(8250, rel=0.01)
    assert all('-rsi2_pullback-' in o.client_order_id for o in broker.orders)
    saved = json.loads((tmp_path / 'pullback_sleeve.json').read_text())
    assert saved['last_decision'] == '2026-12-08' and saved['decision']['action'] == 'buy'
    assert _run(agent, CLOSE_WINDOW.replace(minute=55), closes, live) == []      # once a day


def test_outside_the_window_and_while_halted_it_does_nothing(tmp_path, monkeypatch):
    broker = _Broker()
    agent = _agent(tmp_path, monkeypatch, broker)
    closes = _uptrend_then_dip()
    live = closes.pop()
    assert _run(agent, MORNING, closes, live) == []
    agent.trading_halted = True
    assert _run(agent, CLOSE_WINDOW, closes, live) == []
    assert broker.orders == []


def test_a_bounce_sells_everything_even_past_a_loss_limit(tmp_path, monkeypatch):
    closes = _uptrend_then_dip() + [0]
    closes[-1] = closes[-2] * 1.04
    held = [SimpleNamespace(symbol='IVV', quantity=20.0, current_price=closes[-1],
                            market_value=20.0 * closes[-1])]
    broker = _Broker(positions=held)
    agent = _agent(tmp_path, monkeypatch, broker, limits=SimpleNamespace(blocks_buys=lambda: True))
    live = closes.pop()
    placed = _run(agent, CLOSE_WINDOW, closes, live)
    assert [(o['symbol'], o['side'], o['quantity']) for o in placed] == [('IVV', 'sell', 20.0)]


def test_a_loss_limit_holds_back_a_buy(tmp_path, monkeypatch):
    broker = _Broker()
    agent = _agent(tmp_path, monkeypatch, broker, limits=SimpleNamespace(blocks_buys=lambda: True))
    closes = _uptrend_then_dip()
    live = closes.pop()
    assert _run(agent, CLOSE_WINDOW, closes, live) == []
    saved = json.loads((tmp_path / 'pullback_sleeve.json').read_text())
    assert saved['decision']['held_back'] == 'loss_limit'


def test_its_fund_carries_no_daily_stop_and_shrinks_the_technical_share(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, _Broker())
    assert 'IVV' in TradingAgent._sleeve_symbols(agent)
    agent._pullback_value = 1234.0
    agent._sleeve_value = 0.0
    assert TradingAgent._sleeve_value_now(agent) == pytest.approx(1234.0)
