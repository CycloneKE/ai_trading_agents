"""Crypto: the AI never blocks an exit, and a coin is capped by its
volatility and trimmed back to the cap once a week."""
import json
import math
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent import crypto_risk as cr
from src.agent.main import TradingAgent
from tests.test_momentum_sleeve import _Broker, _agent

CFG = cr.CryptoRiskConfig(enabled=True)
MONDAY = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)


def _closes(daily_vol, n=200, start=100.0):
    """Closes alternating up and down by `daily_vol`: that daily volatility."""
    out = [start]
    for i in range(n):
        out.append(out[-1] * (1 + (daily_vol if i % 2 == 0 else -daily_vol)))
    return out


# ------------------------------------------------------------------ the rule

def test_volatility_is_measured_over_the_window_and_yearly():
    vol = cr.annual_vol(_closes(0.02), 60)
    assert vol == pytest.approx(0.02 * math.sqrt(365), rel=0.05)
    assert cr.annual_vol([100.0] * 30, 60) is None


def test_a_coin_swinging_more_than_the_target_gets_less_room():
    assert cr.cap_value(1000.0, 0.25, CFG) == pytest.approx(1000.0)     # calmer than 50%: the maximum
    assert cr.cap_value(1000.0, 1.00, CFG) == pytest.approx(500.0)      # twice the target: half
    assert cr.cap_value(1000.0, None, CFG) == pytest.approx(1000.0)


def test_a_holding_is_trimmed_only_beyond_the_buffer_and_only_down_to_the_cap():
    assert cr.trim_quantity(1150.0, 10.0, 1000.0, CFG) == 0.0           # within 20%
    assert cr.trim_quantity(1500.0, 10.0, 1000.0, CFG) == pytest.approx(50.0)
    assert cr.trim_quantity(1210.0, 10.0, 1000.0, cr.CryptoRiskConfig(min_trade_usd=500)) == 0.0


def test_trims_are_due_once_per_iso_week():
    assert cr.due(None, date(2026, 11, 23))
    assert not cr.due('2026-11-23', date(2026, 11, 29))
    assert cr.due('2026-11-29', date(2026, 11, 30))


# ------------------------------------------------------------------ the AI veto

def _veto_agent(veto_enabled):
    return SimpleNamespace(config={'llm_veto_enabled': veto_enabled,
                                   'data_manager': {'crypto_symbols': ['BTC-USD']}})


def test_the_ai_can_never_block_a_crypto_exit():
    for enabled in (True, False):
        agent = _veto_agent(enabled)
        assert TradingAgent._veto_is_advisory(agent, 'BTC-USD', {'action': 'hold'}, 'sell')
        assert TradingAgent._veto_is_advisory(agent, 'BTC-USD', {'consensus_verified': False}, 'sell')


def test_a_crypto_buy_follows_the_same_veto_setting_as_every_other_market():
    assert TradingAgent._veto_is_advisory(_veto_agent(False), 'BTC-USD', {}, 'buy')
    assert not TradingAgent._veto_is_advisory(_veto_agent(True), 'BTC-USD', {}, 'buy')
    assert not TradingAgent._veto_is_advisory(_veto_agent(False), 'BTC-USD', {'consensus_verified': False}, 'buy')
    assert TradingAgent._veto_is_advisory(_veto_agent(False), 'AAPL', {}, 'sell')
    assert not TradingAgent._veto_is_advisory(_veto_agent(True), 'AAPL', {}, 'sell')


# ------------------------------------------------------------------ the weekly trim in the agent

def _btc(qty, price):
    return SimpleNamespace(symbol='BTC-USD', quantity=qty, current_price=price, market_value=qty * price)


def _run(agent, now=MONDAY, closes=None):
    agent._crypto_trim_last_check = 0.0
    closes = closes if closes is not None else _closes(0.05)     # about 95% a year
    return TradingAgent._run_crypto_trim_cycle(agent, now=now, closes_for=lambda sym: closes)


def _cap_for(agent, closes):
    max_pct = 0.05
    return TradingAgent._crypto_cap(agent, 'BTC-USD', TradingAgent._technical_equity(agent, 100000.0),
                                    max_pct, closes=closes)


def test_a_coin_grown_past_its_cap_is_sold_back_to_it_once_a_week(tmp_path, monkeypatch):
    broker = _Broker(positions=[_btc(0.1, 60000.0)])               # worth 6,000
    agent = _agent(tmp_path, monkeypatch, broker)
    cap = _cap_for(agent, _closes(0.05))
    assert cap < 6000 / 1.2
    placed = _run(agent)
    assert [(o['symbol'], o['side']) for o in placed] == [('BTC-USD', 'sell')]
    assert placed[0]['quantity'] == pytest.approx((6000.0 - cap) / 60000.0, rel=1e-6)
    assert all('-crypto_trim-' in o.client_order_id for o in broker.orders)
    saved = json.loads((tmp_path / 'crypto_risk.json').read_text())
    assert saved['last_trim'] == '2026-11-23' and saved['review']['BTC-USD']['trim_qty'] > 0
    assert _run(agent, now=MONDAY.replace(day=25)) == []            # the same week: nothing


def test_a_coin_within_its_buffer_and_other_holdings_are_left_alone(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, _Broker())
    cap = _cap_for(agent, _closes(0.05))
    held = [_btc(cap * 1.1 / 60000.0, 60000.0),
            SimpleNamespace(symbol='AAPL', quantity=500, current_price=200.0, market_value=100000.0)]
    broker = _Broker(positions=held)
    agent = _agent(tmp_path, monkeypatch, broker)
    assert _run(agent) == [] and broker.orders == []


def test_no_trim_while_halted_or_without_enough_history(tmp_path, monkeypatch):
    broker = _Broker(positions=[_btc(0.1, 60000.0)])
    agent = _agent(tmp_path, monkeypatch, broker)
    agent.trading_halted = True
    assert _run(agent) == []
    agent.trading_halted = False
    assert _run(agent, closes=[60000.0] * 30) == []
    assert broker.orders == []


def test_a_crypto_buy_is_never_larger_than_its_cap(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, _Broker())
    calm, wild = _closes(0.01), _closes(0.08)
    assert TradingAgent._crypto_cap(agent, 'BTC-USD', 30000.0, 0.05, closes=calm) == pytest.approx(1500.0)
    assert TradingAgent._crypto_cap(agent, 'BTC-USD', 30000.0, 0.05, closes=wild) < 1500.0 * 0.5
    agent.config['crypto_risk']['enabled'] = False
    agent._crypto_risk_cfg = None
    assert TradingAgent._crypto_cap(agent, 'BTC-USD', 30000.0, 0.05, closes=wild) == pytest.approx(1500.0)
