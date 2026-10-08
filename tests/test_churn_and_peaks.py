"""The buy, stop, sell, buy-again loop, and the guards against it.

In the paper run a trailing stop closed a crypto position; the strategy still
said buy, so the agent bought it back at nearly the same price. The new
position inherited the old position's trailing-stop peak, was stopped out
within minutes, and the cycle repeated all day: hundreds of round trips, 15%
of them winners, each paying the spread and fees. The fixes:

1. a position the agent closes no longer leaves a trailing-stop peak behind;
2. a symbol sold by a strategy or a stop is not bought back for 24 hours;
3. a symbol bought and sold again within the hour is visible (scorecard, alert,
   scripts/churn_report.py), whatever the cause.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import src.utils.paths as paths
from src.agent import churn, scorecard
from src.agent.alerts import AlertLog, EmailAlerter
from src.agent.main import TradingAgent
from src.agent.order_journal import OrderJournal
from src.agent.realtime_risk_manager import RealTimeRiskManager
from src.agent.round_trips import closed_round_trips
from src.agent.self_healing import RunState, SelfHealing
from src.connectors.base_broker import Position
from src.utils.config_validator import load_config

CONFIG = load_config('config/config.json')


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'DATA_DIR', tmp_path)
    return tmp_path


def btc(price, avg, qty=0.03):
    return Position(symbol='BTC-USD', quantity=qty, avg_entry_price=avg, current_price=price,
                    market_value=qty * price, unrealized_pl=qty * (price - avg),
                    unrealized_pl_percent=(price - avg) / avg, cost_basis=qty * avg)


def seen(price, avg):
    return SimpleNamespace(symbol='BTC-USD', quantity=0.03, avg_entry_price=avg, current_price=price,
                           market_value=0.03 * price)


# ---------------------------------------------------- the peak of a closed position

def test_a_reentry_at_the_same_price_inherits_a_stale_peak_unless_the_close_is_remembered(data_dir):
    """Why the agent must say when it closes a position: the entry-price check
    alone cannot tell a re-entry at the same price from the same holding."""
    risk = RealTimeRiskManager({}, None)
    for price in (83134, 85000, 87400):
        risk.sync_broker_positions([seen(price, 83134)])
    risk.sync_broker_positions([seen(83000, 83134)])
    assert [s['symbol'] for s in risk.check_trailing_stops(0.05)] == ['BTC-USD']        # a real stop
    risk.sync_broker_positions([])
    risk.sync_broker_positions([seen(83085, 83085)])                                   # bought back
    assert risk.positions['BTC-USD']['high_watermark'] == 87400.0                      # the old peak, on a new position

    risk.forget_peak('BTC-USD')
    risk.sync_broker_positions([])
    risk.sync_broker_positions([seen(83085, 83085)])
    assert risk.positions['BTC-USD']['high_watermark'] == 83085.0                     # its own price
    risk.sync_broker_positions([seen(83000, 83085)])
    assert risk.check_trailing_stops(0.05) == []


def test_a_forgotten_peak_stays_forgotten_after_a_restart(data_dir):
    risk = RealTimeRiskManager({}, None)
    for price in (83134, 87400):
        risk.sync_broker_positions([seen(price, 83134)])
    risk.forget_peak('BTC-USD')
    after = RealTimeRiskManager({}, None)                                              # a redeploy
    after.sync_broker_positions([seen(83085, 83134)])
    assert after.positions['BTC-USD']['high_watermark'] == 83085.0


def test_forgetting_a_symbol_that_was_never_held_is_harmless(data_dir):
    RealTimeRiskManager({}, None).forget_peak('NOPE')


def make_agent(tmp_path, risk):
    broker = SimpleNamespace(is_connected=True, positions=[], placed=[])
    broker.get_positions = lambda: broker.positions
    broker.place_order = lambda o: broker.placed.append(o) or SimpleNamespace(order_id=f'o{len(broker.placed)}', status='new')
    agent = SimpleNamespace(risk_manager=risk, order_journal=OrderJournal(str(tmp_path / 'orders.db')),
                            monitoring_service=None, config=CONFIG)
    agent._core_symbols = lambda: set()
    agent._stop_distances_for = lambda *a: {'stop_loss_pct': 0.20, 'trailing_stop_pct': 0.05, 'source': 'test'}
    agent._has_open_close_order = lambda *a: False
    return agent, broker


def test_the_agents_own_stop_out_does_not_leave_a_peak_for_the_next_position(data_dir, tmp_path, monkeypatch):
    """The screenshot's scenario, through the real stop-loss code. A stop order
    is de-duplicated within a five-minute window, which is why the loop beat
    once every five minutes; the clock here moves on like the real one does."""
    clock = {'t': 1_000_000.0}
    monkeypatch.setattr('src.agent.main.time', SimpleNamespace(time=lambda: clock['t'], sleep=lambda s: None))
    risk = RealTimeRiskManager({}, None)
    agent, broker = make_agent(tmp_path, risk)

    def cycle(price, avg):
        broker.positions = [btc(price, avg)]
        risk.sync_broker_positions(broker.positions)
        TradingAgent._enforce_stops_on(agent, broker, 0.20, 0.05)
        clock['t'] += 60                                                 # the loop runs every minute

    for price in (83134, 85000, 87400):
        cycle(price, 83134)
    assert broker.placed == []
    cycle(83000, 83134)                                                  # the stop fires, once
    assert [(o.side, o.symbol) for o in broker.placed] == [('sell', 'BTC-USD')]
    broker.positions = []
    risk.sync_broker_positions([])                                       # it is sold
    clock['t'] += 6 * 60                                                 # the next five-minute window
    for _ in range(4):
        cycle(83085, 83085)                                              # bought back at nearly the same price
        cycle(83000, 83085)                                              # and a few ticks down
        clock['t'] += 5 * 60
    assert len(broker.placed) == 1                                       # no second stop: the loop is broken


def test_a_signal_exit_of_the_whole_position_also_forgets_its_peak(data_dir, tmp_path):
    forgotten = []
    risk = SimpleNamespace(forget_peak=forgotten.append)
    TradingAgent._forget_peak(SimpleNamespace(risk_manager=risk), 'ETH-USD')
    assert forgotten == ['ETH-USD']
    TradingAgent._forget_peak(SimpleNamespace(risk_manager=None), 'ETH-USD')            # no risk manager: nothing happens
    TradingAgent._forget_peak(SimpleNamespace(risk_manager=SimpleNamespace(forget_peak=lambda s: 1 / 0)), 'ETH-USD')


# ------------------------------------------------------------ the re-entry pause

def journal_with_sell(tmp_path, strategy, status='submitted', side='sell', symbol='BTC-USD'):
    j = OrderJournal(str(tmp_path / 'orders.db'))
    j.record_intent('c1', symbol, side, 0.03, 'market', strategy=strategy)
    if status == 'submitted':
        j.mark_submitted('c1', 'b1', 'new')
    elif status == 'failed':
        j.mark_failed('c1', 'no reply')
    return j


def agent_with(journal, hours=None):
    cfg = {'trading': {}} if hours is None else {'trading': {'reentry_cooldown_hours': hours}}
    return SimpleNamespace(config=cfg, order_journal=journal)


@pytest.mark.parametrize('strategy', ['trailing_stop', 'stop_loss', 'crypto_trend', 'ensemble'])
def test_a_symbol_sold_by_a_strategy_or_a_stop_is_not_bought_back_for_a_day(tmp_path, strategy):
    agent = agent_with(journal_with_sell(tmp_path, strategy))
    now = datetime.utcnow()
    assert TradingAgent._in_reentry_cooldown(agent, 'BTC-USD', now + timedelta(minutes=5))
    assert TradingAgent._in_reentry_cooldown(agent, 'BTC-USD', now + timedelta(hours=23))
    assert not TradingAgent._in_reentry_cooldown(agent, 'BTC-USD', now + timedelta(hours=25))
    assert not TradingAgent._in_reentry_cooldown(agent, 'ETH-USD', now)              # another symbol is free


def test_the_pause_ignores_what_it_should(tmp_path):
    now = datetime.utcnow()
    for kwargs in ({'strategy': 'kill_switch'}, {'strategy': 'core'}, {'strategy': 'ensemble', 'side': 'buy'},
                   {'strategy': 'ensemble', 'status': 'failed'}):
        (tmp_path / 'orders.db').unlink(missing_ok=True)
        agent = agent_with(journal_with_sell(tmp_path, **kwargs))
        assert not TradingAgent._in_reentry_cooldown(agent, 'BTC-USD', now), kwargs


def test_the_pause_can_be_changed_or_turned_off(tmp_path):
    journal = journal_with_sell(tmp_path, 'trailing_stop')
    now = datetime.utcnow() + timedelta(hours=2)
    assert not TradingAgent._in_reentry_cooldown(agent_with(journal, 1), 'BTC-USD', now)
    assert TradingAgent._in_reentry_cooldown(agent_with(journal, 3), 'BTC-USD', now)
    assert not TradingAgent._in_reentry_cooldown(agent_with(journal, 0), 'BTC-USD', now)
    assert not TradingAgent._in_reentry_cooldown(agent_with(None), 'BTC-USD', now)
    assert CONFIG['trading']['reentry_cooldown_hours'] == 24


def gate(agent, symbol, action, held=0.0):
    broker = SimpleNamespace(fetch_positions=lambda: [SimpleNamespace(symbol=symbol, quantity=held, avg_entry_price=100.0)]
                             if held else [], fetch_open_orders=lambda s=None: [])
    return TradingAgent._position_gate(agent, broker, symbol, action)


def test_the_position_gate_blocks_a_new_buy_in_the_pause_but_not_an_add_or_a_sell(tmp_path):
    agent = agent_with(journal_with_sell(tmp_path, 'trailing_stop'))
    assert gate(agent, 'BTC-USD', 'buy')[:2] == (False, 'reentry_cooldown')
    assert gate(agent, 'BTC-USD', 'buy', held=0.03)[:2] == (True, None)              # an add is decided elsewhere
    assert gate(agent, 'BTC-USD', 'sell', held=0.03)[:2] == (True, None)             # an exit is never held back
    assert gate(agent, 'ETH-USD', 'buy')[:2] == (True, None)


def test_the_new_skip_reason_is_known_to_the_journal_and_the_notifications():
    from src.agent.anomaly_scan import HELD_BACK
    from src.agent.decision_journal import SKIP_REASONS
    assert 'reentry_cooldown' in SKIP_REASONS and 'reentry_cooldown' in HELD_BACK


# ----------------------------------------------------------------- seeing churn

def fills(rounds, symbol='BTC-USD', minutes=4, start=None, gap_minutes=1, exit_tag='trailing_stop'):
    """`rounds` buy-then-sell pairs, each held `minutes`, ending at `start`."""
    end = start or datetime.utcnow()
    t = end - timedelta(minutes=rounds * (minutes + gap_minutes))
    out = []
    for i in range(rounds):
        out.append({'symbol': symbol, 'side': 'buy', 'filled_quantity': 0.03, 'filled_avg_price': 83000.0,
                    'created_at': t.isoformat(), 'strategy': 'crypto_trend'})
        t += timedelta(minutes=minutes)
        out.append({'symbol': symbol, 'side': 'sell', 'filled_quantity': 0.03, 'filled_avg_price': 82900.0,
                    'created_at': t.isoformat(), 'strategy': exit_tag})
        t += timedelta(minutes=gap_minutes)
    return out


def test_round_trips_now_say_when_they_closed():
    [trip] = closed_round_trips(fills(1), CONFIG)
    assert trip['closed_at'] and trip['days'] * 1440 == pytest.approx(4, abs=0.01)


def test_only_round_trips_closed_within_the_hour_count_and_they_are_grouped_by_symbol():
    f = fills(6) + fills(2, 'ETH-USD', minutes=3) + fills(3, 'SOL-USD', minutes=60 * 30)
    trips = closed_round_trips(f, CONFIG)
    quick = churn.quick_trips(trips)
    assert len(quick) == 8                                                  # SOL's were held 30 hours
    rows = churn.by_symbol(quick)
    assert list(rows) == ['BTC-USD', 'ETH-USD'] and rows['BTC-USD']['n'] == 6
    assert rows['BTC-USD']['exits'] == {'trailing_stop': 6} and rows['BTC-USD']['median_minutes'] == pytest.approx(4, abs=0.1)
    assert rows['BTC-USD']['mean_return_pct'] < 0
    recent = churn.recent(f, CONFIG, hours=24)
    assert recent['BTC-USD']['n'] == 6
    assert churn.recent(f, CONFIG, hours=24, now=datetime.now(timezone.utc) + timedelta(days=3)) == {}   # all older than a day


NOW = datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)


def churn_metric(f):
    inputs = {'run_started_at': (NOW - timedelta(days=8)).isoformat(), 'planned_days': 90, 'decisions': [], 'fills': f,
              'equity': [], 'spy': None, 'closes_for': None, 'healing': None, 'alerter': None, 'alerts': [], 'tuner': None}
    card = scorecard.build(inputs, CONFIG, NOW)
    return next(m for s in card['sections'] for m in s['metrics'] if m['id'] == 'churn')


def test_the_scorecard_fails_when_most_trades_are_in_and_out_within_the_hour():
    m = churn_metric(fills(40, start=NOW.replace(tzinfo=None) - timedelta(hours=2)))
    assert m['status'] == 'fail' and m['value'] == 40
    assert 'BTC-USD 40' in m['note'] and 'over and over' in m['note']
    assert m['detail']['BTC-USD']['n'] == 40


def test_a_few_quick_trades_are_a_watch_and_normal_trading_passes():
    quick_few = fills(3, start=NOW.replace(tzinfo=None) - timedelta(hours=2))
    held_days = fills(4, minutes=60 * 48, start=NOW.replace(tzinfo=None) - timedelta(hours=2))
    assert churn_metric(quick_few)['status'] == 'watch'
    assert churn_metric(held_days)['status'] == 'pass' and churn_metric(held_days)['value'] == 0
    assert churn_metric([])['status'] == 'info'


# ------------------------------------------------------------------- the alert

def healing_with(tmp_path, fill_rows, config=None):
    sent = []
    alerter = EmailAlerter({}, env={'ALERT_EMAIL_TO': 'a@x.com', 'SMTP_HOST': 'h', 'SMTP_USER': 'u@x.com'},
                           log=AlertLog(tmp_path / 'a.jsonl'), transport=lambda s, m: sent.append(m), background=False)
    agent = SimpleNamespace(order_journal=SimpleNamespace(filled_orders=lambda: fill_rows),
                            config=config or CONFIG, trading_halted=False, halt_reason=None)
    clock = {'t': 1000.0}
    sh = SelfHealing(agent, alerter, config or CONFIG, clock=lambda: clock['t'],
                     run_state=RunState(tmp_path / 'rs.json'))
    return sh, sent, clock


def test_the_agent_emails_when_a_symbol_is_bought_and_sold_again_and_again(tmp_path):
    sh, sent, clock = healing_with(tmp_path, fills(6) + fills(2, 'ETH-USD'))
    sh._check_churn()
    assert len(sent) == 1 and 'CRITICAL' in sent[0]['Subject'] and '1 position(s)' in sent[0]['Subject']
    body = sent[0].get_content()
    assert 'BTC-USD: 6 times' in body and 'ETH-USD' not in body and 'trailing_stop' in body
    clock['t'] += 11 * 60
    sh._check_churn()
    assert len(sent) == 1                                                   # the same alert is not repeated for hours


def test_a_couple_of_quick_trades_or_none_is_not_an_alarm_and_the_check_is_rate_limited(tmp_path):
    sh, sent, clock = healing_with(tmp_path, fills(2))
    sh._check_churn()
    assert sent == []
    quiet, sent2, _ = healing_with(tmp_path, [])
    quiet._check_churn()
    assert sent2 == []
    calls = []
    sh.agent.order_journal = SimpleNamespace(filled_orders=lambda: calls.append(1) or [])
    sh._check_churn()
    sh._check_churn()
    assert len(calls) == 0                                                  # inside the ten-minute window: not read again
    clock['t'] += 11 * 60
    sh._check_churn()
    assert len(calls) == 1
