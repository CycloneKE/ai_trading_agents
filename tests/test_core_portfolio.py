"""The core of the core-satellite split (src/agent/core_portfolio.py) and
how the trading loop runs it (TradingAgent._run_core_cycle)."""
import math
import types
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent import core_portfolio as core
from src.agent.main import TradingAgent
from src.agent.order_journal import OrderJournal
from tests.test_live_execution_rules import FillingBroker


def _walk(vol_daily, n=80, start=100.0):
    """Closes alternating up and down by `vol_daily`, a steady known swing."""
    px, out = start, []
    for i in range(n):
        px *= math.exp(vol_daily if i % 2 else -vol_daily)
        out.append(px)
    return out


# -------------------------------------------------------------- weights

def test_a_fund_that_swings_twice_as_much_gets_half_the_money():
    w = core.inverse_vol_weights({'VOO': _walk(0.01), 'IEF': _walk(0.005)}, max_weight=1.0)
    assert w['IEF'] == pytest.approx(2 * w['VOO'], rel=1e-6)
    assert sum(w.values()) == pytest.approx(1.0)


def test_no_fund_exceeds_the_cap_and_the_excess_is_shared():
    w = core.inverse_vol_weights({'VOO': _walk(0.01), 'IEF': _walk(0.001), 'IAU': _walk(0.01)},
                                 max_weight=0.6)
    assert w['IEF'] == pytest.approx(0.6)
    assert w['VOO'] == pytest.approx(0.2) and w['IAU'] == pytest.approx(0.2)


def test_without_history_the_core_is_equal_weighted():
    w = core.inverse_vol_weights({'VOO': [], 'IEF': [100.0], 'IAU': []})
    assert list(w.values()) == pytest.approx([1 / 3] * 3)


def test_a_fund_without_enough_history_gets_the_average_swing():
    w = core.inverse_vol_weights({'VOO': _walk(0.01), 'IAU': _walk(0.01), 'NEW': [1.0, 1.1]})
    assert w['NEW'] == pytest.approx(w['VOO'])


# -------------------------------------------------------------- config

def test_a_core_fund_the_strategies_trade_is_left_out():
    cfg = core.config_from_dict({'enabled': True, 'core_symbols': ['SPY', 'IEF']}, ['SPY', 'AAPL'])
    assert cfg.symbols == ('IEF',) and cfg.enabled


def test_active_share_is_clamped_and_sets_the_strategies_equity():
    cfg = core.config_from_dict({'enabled': True, 'active_share': 1.7})
    assert cfg.active_share == 1.0
    cfg = core.config_from_dict({'enabled': True, 'active_share': 0.55})
    assert core.active_equity(100000, cfg) == pytest.approx(55000)
    assert core.active_equity(100000, core.config_from_dict({'enabled': False})) == 100000


def test_the_shipped_config_keeps_core_funds_out_of_the_active_universe():
    from src.utils.config_validator import load_config
    c = load_config('config/config.json')
    cfg = core.config_from_dict(c['core_satellite'], c['data_manager']['symbols'])
    assert cfg.enabled and 0.5 <= cfg.active_share <= 0.6
    assert set(cfg.symbols) == {'VOO', 'IEF', 'IAU'}


# ------------------------------------------------------------ schedule

def test_when_the_core_rebalances():
    targets = {'VOO': 0.5, 'IEF': 0.5}
    on_target = {'VOO': 500.0, 'IEF': 500.0}
    today = date(2026, 10, 14)
    assert core.rebalance_reason(None, today, {}, targets, 0.05) == 'build'
    assert core.rebalance_reason('2026-10-01', today, {}, targets, 0.05) == 'build'
    assert core.rebalance_reason('2026-09-30', today, on_target, targets, 0.05) == 'monthly'
    assert core.rebalance_reason('2026-10-01', today, on_target, targets, 0.05) is None
    drifted = {'VOO': 600.0, 'IEF': 400.0}  # 60/40 against 50/50
    assert core.rebalance_reason('2026-10-01', today, drifted, targets, 0.05) == 'drift'


# -------------------------------------------------------------- orders

def test_orders_sell_first_and_skip_small_adjustments():
    orders = core.plan_orders({'VOO': 0.5, 'IEF': 0.5}, 10000,
                              held={'VOO': 14.0, 'IEF': 2.0, 'GLD': 1.0},
                              prices={'VOO': 500.0, 'IEF': 100.0, 'GLD': 200.0},
                              min_trade_usd=100)
    assert [(o['symbol'], o['side']) for o in orders] == \
        [('GLD', 'sell'), ('VOO', 'sell'), ('IEF', 'buy')]
    assert orders[1]['quantity'] == pytest.approx(4.0)       # 7000 held, 5000 wanted
    assert orders[2]['notional'] == pytest.approx(4800.0)
    assert core.plan_orders({'VOO': 1.0}, 10000, {'VOO': 19.9}, {'VOO': 500.0}, 100) == []


def test_buys_never_spend_more_than_the_cash_and_sale_proceeds():
    orders = [{'symbol': 'VOO', 'side': 'sell', 'quantity': 2, 'notional': 1000.0},
              {'symbol': 'IEF', 'side': 'buy', 'notional': 6000.0},
              {'symbol': 'IAU', 'side': 'buy', 'notional': 2000.0}]
    fitted = core.fit_to_cash(orders, cash=3000.0, headroom=0.0)
    assert sum(o['notional'] for o in fitted if o['side'] == 'buy') == pytest.approx(4000.0)
    assert fitted[1]['notional'] == pytest.approx(3000.0)
    assert core.fit_to_cash(orders, cash=100000.0) == orders


def test_the_session_is_new_york_hours_on_weekdays():
    assert core.us_session_open(datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc))      # 10:00 ET
    assert not core.us_session_open(datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc))  # 09:00 ET
    assert not core.us_session_open(datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc))  # Saturday


# ---------------------------------------------------- in the trading loop

class _Broker(FillingBroker):
    def __init__(self, prices, cash=100000.0):
        super().__init__(price=0.0)
        self.prices = prices
        self.cash = cash

    def get_account_info(self):
        equity = self.cash + sum(q * self.prices[s] for s, q in self.held.items())
        return SimpleNamespace(equity=equity, cash=self.cash)

    def fetch_positions(self):
        from src.connectors.base_broker import Position
        return [Position(symbol=s, quantity=q, avg_entry_price=self.prices[s],
                         current_price=self.prices[s], market_value=q * self.prices[s],
                         unrealized_pl=0.0, unrealized_pl_percent=0.0,
                         cost_basis=q * self.prices[s]) for s, q in self.held.items() if q]

    def place_order(self, order):
        self.price = self.prices[order.symbol]
        return super().place_order(order)


PRICES = {'VOO': 500.0, 'IEF': 100.0, 'IAU': 50.0}
SESSION = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)


def _agent(tmp_path, broker, halted=False):
    cfg = {'core_satellite': {'enabled': True, 'active_share': 0.55,
                              'core_symbols': ['VOO', 'IEF', 'IAU']},
           'data_manager': {'symbols': ['SPY', 'AAPL']},
           'trading': {'allow_fractional': True}}
    agent = SimpleNamespace(
        config=cfg, trading_halted=halted,
        components={'broker_manager': SimpleNamespace(get_broker=lambda *a: broker)},
        order_journal=OrderJournal(db_path=str(tmp_path / 'orders.db')),
        _core_cfg=core.config_from_dict(cfg['core_satellite'], ['SPY', 'AAPL']),
        _core_state=core.CoreState(str(tmp_path / 'core.json')),
        _core_last_check=0.0, _core_value=0.0)
    for name in ('_core', '_run_core_cycle', '_place_core_orders'):
        setattr(agent, name, types.MethodType(getattr(TradingAgent, name), agent))
    return agent


def _history(symbols, bars):
    return {'VOO': {'close': _walk(0.01)}, 'IEF': {'close': _walk(0.005)},
            'IAU': {'close': _walk(0.01)}}


def test_the_first_session_builds_the_core_at_its_share_of_equity(tmp_path):
    broker = _Broker(PRICES)
    agent = _agent(tmp_path, broker)
    placed = agent._run_core_cycle(now=SESSION, fetch_history=_history, price_for=PRICES.get)
    assert {o['symbol'] for o in placed} == {'VOO', 'IEF', 'IAU'}
    assert all(o.side == 'buy' and o.client_order_id.startswith('aegis-core-')
               for o in broker.orders)
    held_value = sum(q * PRICES[s] for s, q in broker.held.items())
    assert held_value == pytest.approx(45000, rel=0.001)     # 45% of $100k
    assert broker.held['IEF'] * 100 == pytest.approx(2 * broker.held['VOO'] * 500, rel=0.001)
    state = core.CoreState(str(tmp_path / 'core.json'))
    assert state.last_rebalance == '2026-10-14' and state.data['reason'] == 'build'
    # The next check that day finds it on target and leaves it alone.
    assert agent._run_core_cycle(now=SESSION, fetch_history=_history, price_for=PRICES.get) == []


def test_the_core_does_not_trade_outside_the_session_or_while_halted(tmp_path):
    broker = _Broker(PRICES)
    assert _agent(tmp_path, broker)._run_core_cycle(
        now=datetime(2026, 10, 14, 2, 0, tzinfo=timezone.utc), fetch_history=_history, price_for=PRICES.get) == []
    assert _agent(tmp_path, broker, halted=True)._run_core_cycle(
        now=SESSION, fetch_history=_history, price_for=PRICES.get) == []
    assert broker.orders == []


def test_the_core_never_borrows_when_the_strategies_hold_most_of_the_cash(tmp_path):
    broker = _Broker({**PRICES, 'AAPL': 100.0}, cash=20000.0)
    broker.held['AAPL'] = 800.0                           # $80k in the satellite
    agent = _agent(tmp_path, broker)
    agent._run_core_cycle(now=SESSION, fetch_history=_history, price_for=PRICES.get)
    assert broker.cash >= 0
    assert sum(q * PRICES[s] for s, q in broker.held.items() if s != 'AAPL') <= 20000


def test_stops_leave_the_core_alone(tmp_path):
    from src.connectors.base_broker import Position
    losing = [Position(symbol=s, quantity=10, avg_entry_price=100.0, current_price=50.0,
                       market_value=500.0, unrealized_pl=-500.0, unrealized_pl_percent=-0.5,
                       cost_basis=1000.0) for s in ('VOO', 'AAPL')]
    broker = SimpleNamespace(is_connected=True, get_positions=lambda: losing, orders=[],
                             get_orders=lambda *a: [], place_order=lambda o: broker.orders.append(o))
    agent = _agent(tmp_path, broker)
    agent._core_symbols = types.MethodType(TradingAgent._core_symbols, agent)
    agent._stop_distances_for = lambda *a: {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03,
                                            'source': 'test'}
    agent._has_open_close_order = lambda *a: False
    agent.order_journal = None
    agent.risk_manager = None
    TradingAgent._enforce_stop_losses(agent, 0.05, 0.03)
    assert [o.symbol for o in broker.orders] == ['AAPL']
