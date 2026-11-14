"""Account loss limits (src/agent/loss_limits.py) and stale-quote refusal.

Before: risk_management.max_daily_loss was never read, the risk manager
reported drawdown as a hard-coded 0.0, a breached risk limit dropped sells as
well as buys, and a vendor's last batch was traded on however old it was.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agent.loss_limits import LossLimits, limits_from_config
from src.agent.main import TradingAgent
from src.utils.timeutil import quote_age_seconds

CFG = {'risk_management': {'max_daily_loss': 0.03}, 'risk_limits': {'max_drawdown': 0.10}}
DAY1 = datetime(2026, 11, 16, 14, 0, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


@pytest.fixture
def limits(tmp_path):
    return LossLimits(CFG, path=str(tmp_path / 'loss_limits.json'))


# ------------------------------------------------------------- the limits

def test_the_limits_come_from_the_settings():
    assert limits_from_config(CFG) == {'max_daily_loss': 0.03, 'max_drawdown': 0.10}
    assert limits_from_config({}) == {'max_daily_loss': 0.03, 'max_drawdown': 0.10}
    assert limits_from_config({'risk_management': {'max_drawdown': 0.2}})['max_drawdown'] == 0.2


def test_a_three_percent_fall_in_a_day_stops_new_buys(limits):
    limits.update(100000, DAY1)
    assert limits.update(98000, DAY1)['breach'] is None
    status = limits.update(96900, DAY1)
    assert status['breach'] == 'daily_loss' and status['buys_blocked']
    assert status['daily_loss'] == pytest.approx(0.031)


def test_the_daily_limit_clears_on_the_next_day(limits):
    limits.update(100000, DAY1)
    limits.update(96000, DAY1)
    assert limits.blocks_buys()
    status = limits.update(96000, DAY2)            # new day: 96,000 is its opening
    assert status['breach'] is None and status['day_start_equity'] == 96000


def test_a_ten_percent_drawdown_stops_new_buys_until_it_recovers(limits):
    for day, equity in enumerate([100000, 98000, 96000, 94000, 92000, 90000]):
        status = limits.update(equity, DAY1 + timedelta(days=day))
    assert status['breach'] == 'drawdown' and status['drawdown'] == pytest.approx(0.10)
    assert limits.update(91000, DAY1 + timedelta(days=6))['breach'] is None


def test_an_operator_can_reset_the_drawdown_reference(limits):
    for day, equity in enumerate([100000, 97000, 94000, 91000, 89000]):
        limits.update(equity, DAY1 + timedelta(days=day))
    assert limits.breach() == 'drawdown'
    status = limits.reset_peak(by='operator')
    assert status['peak_equity'] == 89000 and status['breach'] is None


def test_a_bad_reading_is_ignored(limits):
    limits.update(100000, DAY1)
    for bad in (0, -5, None, 'n/a'):
        assert limits.update(bad, DAY1)['breach'] is None
    assert limits.equity == 100000


def test_the_peak_and_the_days_opening_survive_a_restart(tmp_path):
    path = str(tmp_path / 'loss_limits.json')
    first = LossLimits(CFG, path=path)
    first.update(100000, DAY1)
    first.update(110000, DAY1)
    again = LossLimits(CFG, path=path)                # restart mid-day
    status = again.update(106000, DAY1)
    assert status['peak_equity'] == 110000 and status['day_start_equity'] == 100000
    assert status['daily_loss'] == 0.0                # up on the day, not a loss


def test_a_limit_of_zero_is_off(tmp_path):
    off = LossLimits({'risk_management': {'max_daily_loss': 0}, 'risk_limits': {'max_drawdown': 0}},
                     path=str(tmp_path / 'l.json'))
    off.update(100000, DAY1)
    assert off.update(50000, DAY1)['breach'] is None


# ------------------------------------------------------------- in the agent

def _agent(tmp_path, breached=True):
    limits = LossLimits(CFG, path=str(tmp_path / 'loss_limits.json'))
    limits.update(100000, DAY1)
    if breached:
        limits.update(96000, DAY1)
    sent = []
    agent = SimpleNamespace(
        config={'data_manager': {'forex_symbols': ['EUR_USD'], 'crypto_symbols': ['BTC-USD']}},
        loss_limits=limits, components={}, order_journal=None, _cycle_decisions={},
        alerter=SimpleNamespace(send=lambda *a, **k: sent.append(a)))
    return agent, sent


def test_a_breach_refuses_buys_but_never_sells(tmp_path):
    agent, _ = _agent(tmp_path)
    assert TradingAgent._loss_limit_blocks_buy(agent, 'AAPL', 'buy') is True
    assert TradingAgent._loss_limit_blocks_buy(agent, 'AAPL', 'sell') is False
    assert TradingAgent._loss_limit_blocks_buy(agent, 'EUR_USD', 'buy') is False   # its own book
    calm, _ = _agent(tmp_path / 'calm', breached=False)
    assert TradingAgent._loss_limit_blocks_buy(calm, 'AAPL', 'buy') is False


def test_a_buy_the_limit_would_refuse_costs_no_ai_review(tmp_path):
    agent, _ = _agent(tmp_path)
    assert TradingAgent._held_back_before_review(agent, 'BTC-USD', 'buy', 50000.0) == 'loss_limit'


def test_the_operator_is_alerted_once_when_a_limit_is_reached(tmp_path):
    agent, sent = _agent(tmp_path, breached=False)
    TradingAgent._update_loss_limits(agent, 99000)
    assert sent == []
    TradingAgent._update_loss_limits(agent, 96500)
    TradingAgent._update_loss_limits(agent, 96000)
    assert len(sent) == 1
    key, subject, _body, severity = sent[0]
    assert key == 'risk:daily_loss' and severity == 'critical' and 'buys' in subject


def test_a_breached_risk_limit_still_lets_sells_through(tmp_path):
    agent, sent = _agent(tmp_path)
    agent._cycle_decisions = {'AAPL': {}, 'MSFT': {}}
    signals = {'AAPL': {'action': 'buy', 'position_size': 0.03},
               'MSFT': {'action': 'sell', 'position_size': 0.03}}
    agent.config['risk_limits'] = {'max_drawdown': 0.10}
    kept = TradingAgent._apply_risk_gate(agent, signals, {'portfolio_var': 0, 'current_drawdown': 0.12,
                                                          'position_sizes': {}})
    assert list(kept) == ['MSFT']
    assert agent._cycle_decisions['AAPL']['skip_reason'] == 'risk_limits'
    assert sent and sent[0][0] == 'heal:risk_limits'
    ok = TradingAgent._apply_risk_gate(agent, signals, {'portfolio_var': 0, 'current_drawdown': 0.0,
                                                        'position_sizes': {}})
    assert ok == signals


# ------------------------------------------------------------- stale quotes

def _iso(seconds_ago):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).replace(tzinfo=None).isoformat()


def _quote_agent(live=None, **dm):
    return SimpleNamespace(config={'data_manager': dm}, components={},
                           _live_quote=lambda symbol, broker=None: live)


def _market(**batches):
    return {'market_data': {src: {'data': {'AAPL': {'close': price}}, 'timestamp': ts}
                            for src, (price, ts) in batches.items()}}


def test_quote_ages_are_read_from_iso_timestamps():
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    assert quote_age_seconds('2026-11-16T11:59:00', now) == 60
    assert quote_age_seconds('2026-11-16T11:58:00Z', now) == 120
    assert quote_age_seconds(None, now) is None and quote_age_seconds('soon', now) is None


def test_a_fresh_batch_is_used():
    agent = _quote_agent()
    got = TradingAgent._extract_symbol_data(agent, _market(real_data=(190.0, _iso(30))), 'AAPL')
    assert got['source'] == 'real_data' and got['close'] == 190.0


def test_an_hours_old_batch_is_never_taken_as_current():
    """The vendor stopped answering an hour ago; only its old batch and the
    synthetic fallback remain. The old batch used to win."""
    agent = _quote_agent()
    market = _market(real_data=(190.0, _iso(3600)), fallback=(201.0, _iso(10)))
    got = TradingAgent._extract_symbol_data(agent, market, 'AAPL')
    assert got['source'] == 'stale' and got['stale_source'] == 'real_data'
    assert got['age_seconds'] >= 3600


def test_a_stale_batch_gives_way_to_a_fresh_broker_quote():
    live = {'symbol': 'AAPL', 'price': 191.5, 'source': 'broker'}
    agent = _quote_agent(live=live)
    got = TradingAgent._extract_symbol_data(agent, _market(real_data=(190.0, _iso(3600))), 'AAPL')
    assert got is live


def test_the_age_limit_follows_the_settings():
    assert TradingAgent._max_quote_age(_quote_agent()) == 180
    assert TradingAgent._max_quote_age(_quote_agent(market_data_interval=30)) == 90
    off = _quote_agent(max_quote_age_seconds=0)
    got = TradingAgent._extract_symbol_data(off, _market(real_data=(190.0, _iso(3600))), 'AAPL')
    assert got['source'] == 'real_data'


def test_a_batch_without_a_timestamp_is_not_refused():
    agent = _quote_agent()
    got = TradingAgent._extract_symbol_data(agent, _market(real_data=(190.0, None)), 'AAPL')
    assert got['source'] == 'real_data'


# ------------------------------------------------------------- the API

def test_the_api_shows_the_limits_and_lets_an_operator_reset_the_peak(tmp_path, monkeypatch):
    import json
    import bcrypt
    import src.api.auth as auth
    from src.api.api_server import TradingAPI
    from src.api.auth import create_token
    users = tmp_path / 'users.json'
    pw = bcrypt.hashpw(b'x', bcrypt.gensalt()).decode()
    users.write_text(json.dumps({'operator': {'password': pw, 'role': 'operator'},
                                 'viewer': {'password': pw, 'role': 'viewer'}}))
    monkeypatch.setattr(auth, 'USERS_FILE', str(users))
    limits = LossLimits(CFG, path=str(tmp_path / 'loss_limits.json'))
    for day, equity in enumerate([100000, 95000, 89000]):
        limits.update(equity, DAY1 + timedelta(days=day))
    client = TradingAPI(SimpleNamespace(components={}, config={'test_mode': True}, loss_limits=limits),
                        {}).app.test_client()
    as_ = lambda who: {'Authorization': f"Bearer {create_token(who, who)}"}

    status = client.get('/api/risk/loss-limits', headers=as_('viewer')).get_json()
    assert status['enabled'] and status['breach'] == 'drawdown' and status['buys_blocked']
    assert client.post('/api/risk/loss-limits/reset', json={'confirm': True},
                       headers=as_('viewer')).status_code == 403
    assert client.post('/api/risk/loss-limits/reset', json={},
                       headers=as_('operator')).status_code == 400
    reset = client.post('/api/risk/loss-limits/reset', json={'confirm': True},
                        headers=as_('operator')).get_json()
    assert reset['breach'] is None and reset['peak_equity'] == 89000
    assert limits.state['peak_reset_by'] == 'operator'
