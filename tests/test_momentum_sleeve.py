"""The momentum sleeve (src/agent/momentum_sleeve.py): 12-1 relative strength,
run live and monthly on its own funds.

Before: momentum_rotation.py, the approach with the strongest published
record, ran only in the backtest script; the live agent traded fast technical
rules on ten closely related names.
"""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent import momentum_sleeve as ms
from src.agent.main import TradingAgent

FIRST_MONDAY = datetime(2026, 12, 7, 15, 0, tzinfo=timezone.utc)     # 10:00 New York


def _config(**sleeve):
    with open('config/config.json') as f:
        cfg = json.load(f)
    cfg['momentum_sleeve'] = {**cfg['momentum_sleeve'], **sleeve}
    return cfg


# ------------------------------------------------------------ the parts

def test_the_universe_never_overlaps_another_book():
    cfg = _config()
    sleeve = ms.config_from_dict(cfg['momentum_sleeve'], cfg['data_manager']['symbols'],
                                 cfg['core_satellite']['core_symbols'])
    assert sleeve.enabled and len(sleeve.universe) == 13
    taken = set(cfg['data_manager']['symbols']) | set(cfg['core_satellite']['core_symbols'])
    assert not set(sleeve.universe) & taken
    clashing = ms.config_from_dict({'enabled': True, 'universe': ['XLF', 'VOO', 'XLK']}, ['XLF'], ['VOO'])
    assert clashing.universe == ('XLK',)


def test_daily_bars_become_month_steps_counted_back_from_the_latest():
    assert ms.monthly_closes(list(range(1, 64))) == [21, 42, 63]
    assert ms.monthly_closes([0, -1, 5.0]) == [5.0]


def test_it_rebalances_on_the_first_session_of_each_month():
    assert ms.due(None, date(2026, 12, 7)) == 'build'
    assert ms.due('2026-11-02', date(2026, 12, 7)) == 'monthly'
    assert ms.due('2026-12-07', date(2026, 12, 21)) is None


def _paths(n=330):
    """Daily closes: two strong risers, one weaker riser, two fallers."""
    def path(daily):
        return [100 * (1 + daily) ** i for i in range(n)]
    return {'XLK': path(0.002), 'XLI': path(0.0015), 'XLU': path(0.0005),
            'TLT': path(-0.001), 'DBC': path(-0.002)}


def test_it_holds_the_leaders_still_rising_and_cash_for_the_rest():
    sleeve = ms.config_from_dict(_config()['momentum_sleeve'])
    plan = ms.targets(_paths(), sleeve)
    assert plan['weights'] == pytest.approx({'XLK': 1 / 3, 'XLI': 1 / 3, 'XLU': 1 / 3})
    assert [r['symbol'] for r in plan['ranking']] == ['XLK', 'XLI', 'XLU', 'TLT', 'DBC']
    falling = {s: list(reversed(c)) for s, c in _paths().items()}
    assert ms.targets(falling, sleeve)['weights'] == {'DBC': 0.5, 'TLT': 0.5}   # only the risers, max 50% each


# ------------------------------------------------------------ in the agent

class _Broker:
    is_connected = True
    broker_name = 'paper'

    def __init__(self, positions=(), cash=100000.0, equity=100000.0):
        self.positions = list(positions)
        self.orders = []
        self.acct = SimpleNamespace(equity=equity, cash=cash)

    def get_account_info(self):
        return self.acct

    def fetch_positions(self):
        return self.positions

    def fetch_open_orders(self, symbol):
        return []

    def place_order(self, order):
        self.orders.append(order)
        return SimpleNamespace(order_id=f'o{len(self.orders)}', status='accepted')


def _agent(tmp_path, monkeypatch, broker, limits=None, **sleeve):
    import src.agent.main as main_mod
    monkeypatch.setattr(main_mod, 'DATA_DIR', tmp_path)
    return SimpleNamespace(config=_config(**sleeve), trading_halted=False, order_journal=None,
                           loss_limits=limits,
                           components={'broker_manager': SimpleNamespace(get_broker=lambda *a: broker)})


def _history(paths):
    return lambda symbols, bars: {s: {'close': paths[s]} for s in symbols if s in paths}


def test_the_first_build_buys_the_leaders_with_the_sleeves_share(tmp_path, monkeypatch):
    broker = _Broker()
    agent = _agent(tmp_path, monkeypatch, broker)
    placed = TradingAgent._run_momentum_cycle(agent, now=FIRST_MONDAY, fetch_history=_history(_paths()),
                                              price_for=lambda s: 50.0)
    assert {o['symbol'] for o in placed} == {'XLK', 'XLI', 'XLU'} and all(o['side'] == 'buy' for o in placed)
    # 100,000 equity x 0.55 active x 0.3 sleeve = 16,500, a third each.
    assert sum(o['notional'] for o in placed) == pytest.approx(16500, rel=0.01)
    assert all('-momentum_rotation-' in o.client_order_id for o in broker.orders)
    saved = json.loads((tmp_path / 'momentum_sleeve.json').read_text())
    assert saved['reason'] == 'build' and saved['last_rebalance'] == '2026-12-07'
    # Nothing again until next month.
    agent._sleeve_last_check = 0.0
    assert TradingAgent._run_momentum_cycle(agent, now=FIRST_MONDAY.replace(day=8),
                                            fetch_history=_history(_paths()), price_for=lambda s: 50.0) == []


def test_a_fund_that_drops_out_of_the_leaders_is_sold(tmp_path, monkeypatch):
    held = [SimpleNamespace(symbol='DBC', quantity=100.0, current_price=50.0, market_value=5000.0)]
    broker = _Broker(positions=held)
    agent = _agent(tmp_path, monkeypatch, broker)
    placed = TradingAgent._run_momentum_cycle(agent, now=FIRST_MONDAY, fetch_history=_history(_paths()),
                                              price_for=lambda s: 50.0)
    sells = [o for o in placed if o['side'] == 'sell']
    assert sells == [{'symbol': 'DBC', 'side': 'sell', 'quantity': 100.0, 'notional': 5000.0,
                      'client_order_id': sells[0]['client_order_id']}]
    assert placed[0]['side'] == 'sell'                    # sells first


def test_while_a_loss_limit_is_reached_it_only_sells(tmp_path, monkeypatch):
    held = [SimpleNamespace(symbol='DBC', quantity=100.0, current_price=50.0, market_value=5000.0)]
    broker = _Broker(positions=held)
    agent = _agent(tmp_path, monkeypatch, broker, limits=SimpleNamespace(blocks_buys=lambda: True))
    placed = TradingAgent._run_momentum_cycle(agent, now=FIRST_MONDAY, fetch_history=_history(_paths()),
                                              price_for=lambda s: 50.0)
    assert [o['side'] for o in placed] == ['sell']


def test_it_trades_only_in_the_us_session_and_never_while_halted(tmp_path, monkeypatch):
    broker = _Broker()
    agent = _agent(tmp_path, monkeypatch, broker)
    saturday = datetime(2026, 12, 5, 15, 0, tzinfo=timezone.utc)
    assert TradingAgent._run_momentum_cycle(agent, now=saturday, fetch_history=_history(_paths())) == []
    agent.trading_halted = True
    assert TradingAgent._run_momentum_cycle(agent, now=FIRST_MONDAY, fetch_history=_history(_paths())) == []
    assert broker.orders == []


def test_the_technical_strategies_size_from_what_the_sleeve_leaves(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, _Broker())
    assert TradingAgent._technical_equity(agent, 100000) == pytest.approx(100000 * 0.55 * 0.7)
    off = _agent(tmp_path, monkeypatch, _Broker(), enabled=False)
    assert TradingAgent._technical_equity(off, 100000) == pytest.approx(55000)


def test_sleeve_holdings_carry_no_stops(tmp_path, monkeypatch):
    held = [SimpleNamespace(symbol='XLK', quantity=10.0, cost_basis=1000.0, unrealized_pl=-300.0,
                            current_price=70.0)]
    broker = _Broker(positions=held)
    broker.get_positions = lambda: held
    agent = _agent(tmp_path, monkeypatch, broker)
    agent.risk_manager = None
    agent.monitoring_service = None
    agent._stop_distances_for = lambda *a: {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03, 'source': 'test'}
    agent._has_open_close_order = lambda *a: False
    TradingAgent._enforce_stops_on(agent, broker, 0.05, 0.03)     # down 30%: a stock would be stopped
    assert broker.orders == []
