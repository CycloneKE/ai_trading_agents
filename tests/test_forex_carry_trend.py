"""The forex carry and trend book: signals, sizing, the margin account that
can sell short, and the weekly cycle in the agent."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.agent import forex_carry_trend as fx
from src.connectors.base_broker import OrderRequest
from src.connectors.forex_margin_broker import ForexMarginBroker

CFG = fx.config_from_dict({'enabled': True, 'fallback_rates_pct': {'_comment': 'x', 'USD': 4.0}})


def _book(tmp_path, prices, **over):
    cfg = {'initial_cash': 10000, 'spread_per_side': 0.0, 'max_leverage': 3.0,
           'rollover_markup_pct': 0.0, 'max_drawdown': 0.15, **over}
    b = ForexMarginBroker(cfg, price_for=lambda p: prices.get(p), state_dir=str(tmp_path))
    b.connect()
    return b


# ------------------------------------------------------------------ settings

def test_config_keeps_dollar_pairs_only_and_skips_comment_keys():
    cfg = fx.config_from_dict({'enabled': True, 'pairs': ['EUR_USD', 'EUR_GBP', 'usd_jpy'],
                               'fallback_rates_pct': {'_comment': 'note', 'EUR': 2}})
    assert cfg.pairs == ('EUR_USD', 'USD_JPY') and cfg.fallback_rates_pct == {'EUR': 2.0}
    assert CFG.enabled and CFG.fallback_rates_pct == {'USD': 4.0}


# ------------------------------------------------------------------ signals

def test_usd_jpy_is_turned_round_so_up_means_the_yen_rose():
    assert fx.foreign('USD_JPY') == 'JPY' and fx.foreign('EUR_USD') == 'EUR'
    assert fx.oriented('USD_JPY', [100.0, 50.0]) == [0.01, 0.02]


def test_carry_buys_above_the_median_rate_gap_and_sells_below():
    s = fx.carry_signals({'AUD_USD': 1.0, 'EUR_USD': -2.0, 'USD_JPY': -3.0, 'GBP_USD': -0.5, 'NZD_USD': 0.5})
    assert s == {'AUD_USD': 1.0, 'NZD_USD': 1.0, 'GBP_USD': 0.0, 'EUR_USD': -1.0, 'USD_JPY': -1.0}


def test_trend_needs_its_full_lookback():
    assert fx.trend_signal([1.0] * 10, 10) is None
    assert fx.trend_signal([1.0] * 10 + [1.1], 10) == 1.0
    assert fx.trend_signal([1.0] * 10 + [0.9], 10) == -1.0


def test_the_trend_counts_the_interest_a_forward_earns():
    flat_dip = [1.0] * 10 + [0.99]                       # price down 1% over the lookback
    assert fx.trend_signal(flat_dip, 10) == -1.0          # on price alone: falling
    assert fx.trend_signal(flat_dip, 10, interest=0.03) == 1.0   # +3% interest: positive excess return


def test_a_high_yielder_drifting_slightly_lower_still_trends_up_on_excess_return():
    cfg = fx.config_from_dict({'enabled': True, 'pairs': ['AUD_USD', 'EUR_USD']})
    closes = {'AUD_USD': _wave(0.66, -0.005), 'EUR_USD': _wave(1.10, -0.005)}
    rates = {'USD': 4.0, 'AUD': 8.0, 'EUR': 4.0}
    on_excess = fx.target_weights(closes, rates, cfg)['signals']['AUD_USD']
    assert on_excess['trend'] == 1.0 and on_excess['trend_interest_pct'] == pytest.approx(4.0)
    price_only = fx.config_from_dict({'enabled': True, 'pairs': ['AUD_USD', 'EUR_USD'], 'trend_on_excess': False})
    assert fx.target_weights(closes, rates, price_only)['signals']['AUD_USD']['trend'] == -1.0
    # A year's average rate, when given, is used instead of today's.
    avg = fx.target_weights(closes, rates, cfg, avg_rates_pct={'USD': 4.0, 'AUD': 4.2})
    assert avg['signals']['AUD_USD']['trend'] == -1.0


def test_the_years_average_rate_fills_unpublished_months_with_the_current_rate():
    row = {'history': [['2026-08-01', 3.0], ['2026-09-01', 3.0], ['2026-10-01', 3.0]]}
    # Nov 2025 to Oct 2026: three published months at 3%, nine others at the current 5%.
    assert fx.average_rate(row, 5.0, date(2026, 11, 21)) == pytest.approx((3 * 3 + 9 * 5) / 12)
    assert fx.average_rate(None, 2.0, date(2026, 11, 21)) == pytest.approx(2.0)


def test_rates_come_with_their_twelve_month_averages(tmp_path):
    cfg = fx.config_from_dict({'enabled': True})
    hist = [(date(2026, m, 1), 4.0 + m / 10) for m in range(1, 11)]
    out = fx.current_rates(['USD'], cfg, str(tmp_path / 'r.json'), today=date(2026, 11, 21),
                           fetch=lambda code: hist)
    assert out['rates_pct']['USD'] == pytest.approx(5.0)
    expected = (sum(4.0 + m / 10 for m in range(1, 11)) + 5.0 * 2) / 12     # Nov, Dec 2025 at today's rate
    assert out['avg_12m_pct']['USD'] == pytest.approx(expected)


def _wave(start, drift, n=400):
    import math
    return [start * (1 + drift * i / n) * (1 + 0.01 * math.sin(i)) for i in range(n)]


def test_weights_long_a_rising_high_yielder_short_a_falling_low_yielder():
    cfg = fx.config_from_dict({'enabled': True, 'pairs': ['AUD_USD', 'USD_JPY', 'EUR_USD']})
    closes = {'AUD_USD': _wave(0.65, 0.10),       # Aussie dollar rising
              'USD_JPY': _wave(140.0, 0.10),      # dollar rising against the yen: the yen falling
              'EUR_USD': _wave(1.10, 0.0)}
    rates = {'USD': 4.0, 'AUD': 4.8, 'JPY': 1.0, 'EUR': 3.0}
    plan = fx.target_weights(closes, rates, cfg)
    w = plan['weights']
    assert plan['signals']['AUD_USD']['signal'] == 1.0 and w['AUD_USD'] > 0
    assert plan['signals']['USD_JPY']['signal'] == -1.0 and w['USD_JPY'] < 0
    units = fx.target_units(w, 10000.0, {'AUD_USD': 0.65, 'USD_JPY': 150.0, 'EUR_USD': 1.1})
    assert units['AUD_USD'] > 0                    # long AUD_USD
    assert units['USD_JPY'] > 0                    # short the yen = LONG the pair USD_JPY


def test_a_pair_without_a_rate_or_history_gets_no_position():
    plan = fx.target_weights({'EUR_USD': [1.1] * 20}, {'USD': 4.0}, CFG)
    assert plan['weights'] == {} and plan['signals']['EUR_USD']['signal'] is None


def test_gross_exposure_is_scaled_down_to_the_leverage_cap():
    cfg = fx.config_from_dict({'enabled': True, 'pairs': ['AUD_USD', 'NZD_USD'], 'vol_target': 5.0,
                               'max_leverage': 2.0})
    closes = {'AUD_USD': _wave(0.65, 0.2), 'NZD_USD': _wave(0.60, 0.2)}
    plan = fx.target_weights(closes, {'USD': 1.0, 'AUD': 5.0, 'NZD': 3.0}, cfg)
    assert plan['gross_leverage'] == pytest.approx(2.0) and plan['scaled_by'] < 1


def test_small_drift_is_left_alone_but_a_flip_or_an_exit_always_trades():
    px = {'EUR_USD': 1.0}
    assert fx.plan_changes({'EUR_USD': 1000}, {'EUR_USD': 1050}, px, CFG) == {}          # 5% drift, $50
    assert fx.plan_changes({'EUR_USD': 1000}, {'EUR_USD': 1500}, px, CFG) == {'EUR_USD': 500}
    assert fx.plan_changes({'EUR_USD': 100}, {'EUR_USD': -100}, px, CFG) == {'EUR_USD': -200}
    assert fx.plan_changes({'EUR_USD': 100}, {}, px, CFG) == {'EUR_USD': -100}


def test_rebalance_is_due_once_per_iso_week():
    assert fx.due(None, date(2026, 11, 23)) == 'build'
    assert fx.due('2026-11-23', date(2026, 11, 27)) is None                 # same week
    assert fx.due('2026-11-27', date(2026, 11, 30)) == 'weekly'             # next Monday


def test_rates_prefer_a_recent_fred_month_and_fall_back_when_stale(tmp_path):
    cfg = fx.config_from_dict({'enabled': True, 'fallback_rates_pct': {'EUR': 2.0, 'USD': 9.9}})
    fred = {'US': [(date(2026, 10, 1), 4.1)], 'EZ': [(date(2026, 1, 1), 2.5)]}
    out = fx.current_rates(['USD', 'EUR'], cfg, str(tmp_path / 'r.json'), today=date(2026, 11, 21),
                           fetch=lambda code: fred[code])
    assert out['rates_pct'] == {'USD': 4.1, 'EUR': 2.0}
    assert out['sources']['EUR'] == 'config fallback_rates_pct'
    # Cached: a second call within the week does not ask FRED again.
    again = fx.current_rates(['USD', 'EUR'], cfg, str(tmp_path / 'r.json'), today=date(2026, 11, 22),
                             fetch=lambda code: (_ for _ in ()).throw(AssertionError('fetched')))
    assert again['rates_pct']['USD'] == 4.1


def test_fred_csv_skips_missing_months():
    rows = fx.parse_fred_csv("observation_date,IR\n2026-08-01,4.2\n2026-09-01,.\n2026-10-01,4.1\n")
    assert rows == [(date(2026, 8, 1), 4.2), (date(2026, 10, 1), 4.1)]


# ------------------------------------------------------------------ the margin book

def test_a_short_profits_when_the_price_falls(tmp_path):
    prices = {'EUR_USD': 1.10}
    b = _book(tmp_path, prices)
    assert b.place_order(OrderRequest('EUR_USD', 10000, 'sell', 'market', 'gtc')).status == 'filled'
    prices['EUR_USD'] = 1.05
    assert b.unrealised('EUR_USD') == pytest.approx(500.0)
    assert b.equity() == pytest.approx(10500.0)
    pos = b.get_positions()[0]
    assert pos.quantity == -10000 and pos.market_value < 0 and pos.unrealized_pl == pytest.approx(500.0)


def test_a_usd_jpy_profit_is_counted_in_dollars(tmp_path):
    prices = {'USD_JPY': 150.0}
    b = _book(tmp_path, prices)
    b.trade('USD_JPY', 10000)                     # long $10,000 against the yen
    prices['USD_JPY'] = 160.0
    # 10,000 x 10 yen = 100,000 yen, which is $625 at 160.
    assert b.unrealised('USD_JPY') == pytest.approx(625.0)


def test_flipping_from_long_to_short_realises_the_long(tmp_path):
    prices = {'AUD_USD': 0.60}
    b = _book(tmp_path, prices)
    b.trade('AUD_USD', 10000)
    prices['AUD_USD'] = 0.66
    b.trade('AUD_USD', -15000)
    assert b.balance == pytest.approx(10600.0)
    assert b.current_units() == {'AUD_USD': pytest.approx(-5000)}
    assert b.positions['AUD_USD']['entry'] == pytest.approx(0.66)
    trip = b.own_round_trips()[-1]
    assert trip['direction'] == 'long' and trip['pnl_usd'] == pytest.approx(600.0)
    assert trip['ret'] == pytest.approx(0.10)


def test_the_spread_is_paid_on_every_fill(tmp_path):
    prices = {'EUR_USD': 1.0}
    b = _book(tmp_path, prices, spread_per_side=0.0001)
    b.trade('EUR_USD', 10000)
    b.trade('EUR_USD', -10000)
    assert b.equity() == pytest.approx(10000 - 2.0)


def test_exposure_past_the_leverage_cap_is_refused_but_a_reduction_is_not(tmp_path):
    prices = {'EUR_USD': 1.0}
    b = _book(tmp_path, prices)
    assert b.trade('EUR_USD', 31000).status == 'rejected'
    assert b.trade('EUR_USD', 29000).status == 'filled'
    assert b.trade('EUR_USD', 2000).status == 'rejected'
    assert b.trade('EUR_USD', -5000).status == 'filled'


def test_no_order_fills_without_a_live_price(tmp_path):
    b = _book(tmp_path, {})
    assert b.trade('EUR_USD', 1000) is None


def test_rollover_pays_the_rate_gap_less_the_markup(tmp_path):
    prices = {'AUD_USD': 0.50}
    b = _book(tmp_path, prices, rollover_markup_pct=0.5)
    b.trade('AUD_USD', 20000)                    # $10,000 long AUD
    rates = {'AUD': 4.8, 'USD': 3.8}
    assert b.accrue_rollover(rates, today=date(2026, 1, 1)) == 0.0     # starts the clock
    earned = b.accrue_rollover(rates, today=date(2026, 1, 11))         # ten days
    assert earned == pytest.approx(10000 * (0.01 - 0.005) * 10 / 365)
    b.trade('AUD_USD', -40000)                   # now $10,000 short AUD: pays the gap
    paid = b.accrue_rollover(rates, today=date(2026, 1, 12))
    assert paid == pytest.approx(10000 * (-0.01 - 0.005) / 365)


def test_past_the_drawdown_limit_it_closes_everything_and_refuses_new_risk(tmp_path):
    prices = {'EUR_USD': 1.00}
    b = _book(tmp_path, prices)
    b.trade('EUR_USD', 20000)
    prices['EUR_USD'] = 0.92                     # -1,600: 16% down
    reason = b.check_drawdown()
    assert reason and b.positions == {} and b.halted
    assert b.trade('EUR_USD', 1000).status == 'rejected'


def test_the_book_survives_a_restart(tmp_path):
    prices = {'GBP_USD': 1.25}
    b = _book(tmp_path, prices)
    b.trade('GBP_USD', -8000)
    again = _book(tmp_path, prices)
    assert again.current_units() == {'GBP_USD': -8000} and again.balance == pytest.approx(b.balance)
    state = json.loads((tmp_path / 'forex_margin_state.json').read_text())
    assert state['positions']['GBP_USD']['units'] == -8000


# ------------------------------------------------------------------ the agent's weekly cycle

def _agent(book, tmp_path, monkeypatch):
    from src.agent import main as main_mod
    monkeypatch.setattr(main_mod, 'DATA_DIR', tmp_path)
    from src.agent.main import TradingAgent
    bm = SimpleNamespace(get_broker=lambda name=None: book if name == 'forex_paper' else None)
    agent = SimpleNamespace(
        config={'forex_carry_trend': {'enabled': True, 'pairs': ['AUD_USD', 'USD_JPY']}},
        components={'broker_manager': bm}, trading_halted=False, alerter=None)
    for name in ('_fx_carry', '_forex_broker', '_run_fx_carry_cycle'):
        setattr(agent, name, getattr(TradingAgent, name).__get__(agent))
    return agent, TradingAgent


def _history(pairs, bars):
    return {'AUD_USD': {'close': _wave(0.60, 0.10)}, 'USD_JPY': {'close': _wave(140.0, 0.10)}}


RATES = lambda ccys: {'rates_pct': {'USD': 4.0, 'AUD': 4.8, 'JPY': 1.0}, 'sources': {}}


def test_the_cycle_rebalances_once_a_week_while_the_market_is_open(tmp_path, monkeypatch):
    prices = {'AUD_USD': 0.66, 'USD_JPY': 154.0}
    book = _book(tmp_path, prices)
    agent, TA = _agent(book, tmp_path, monkeypatch)
    saturday = datetime(2026, 11, 21, 15, tzinfo=timezone.utc)
    assert TA._run_fx_carry_cycle(agent, now=saturday, fetch_history=_history, rates_for=RATES) == []
    monday = datetime(2026, 11, 23, 15, tzinfo=timezone.utc)
    fills = TA._run_fx_carry_cycle(agent, now=monday, fetch_history=_history, rates_for=RATES)
    assert {f['symbol'] for f in fills} == {'AUD_USD', 'USD_JPY'}
    units = book.current_units()
    assert units['AUD_USD'] > 0 and units['USD_JPY'] > 0      # long Aussie, short yen
    tuesday = datetime(2026, 11, 24, 15, tzinfo=timezone.utc)
    assert TA._run_fx_carry_cycle(agent, now=tuesday, fetch_history=_history, rates_for=RATES) == []


def test_a_history_outage_on_rebalance_day_waits_instead_of_closing_the_book(tmp_path, monkeypatch):
    prices = {'AUD_USD': 0.66, 'USD_JPY': 154.0}
    book = _book(tmp_path, prices)
    agent, TA = _agent(book, tmp_path, monkeypatch)
    monday = datetime(2026, 11, 23, 15, tzinfo=timezone.utc)
    TA._run_fx_carry_cycle(agent, now=monday, fetch_history=_history, rates_for=RATES)
    held = book.current_units()
    next_monday = datetime(2026, 11, 30, 15, tzinfo=timezone.utc)
    outage = lambda pairs, bars: {}                          # Yahoo returned nothing
    assert TA._run_fx_carry_cycle(agent, now=next_monday, fetch_history=outage, rates_for=RATES) == []
    assert book.current_units() == held                       # nothing was closed
    assert agent._fx_state.data['last_rebalance'] == '2026-11-23'   # the week is still to do


def test_the_cycle_places_nothing_while_the_agent_is_halted(tmp_path, monkeypatch):
    book = _book(tmp_path, {'AUD_USD': 0.66, 'USD_JPY': 154.0})
    agent, TA = _agent(book, tmp_path, monkeypatch)
    agent.trading_halted = True
    monday = datetime(2026, 11, 23, 15, tzinfo=timezone.utc)
    assert TA._run_fx_carry_cycle(agent, now=monday, fetch_history=_history, rates_for=RATES) == []
    assert book.current_units() == {}


def test_the_fast_loop_and_the_daily_stops_leave_the_margin_book_alone(tmp_path):
    from src.agent.main import TradingAgent
    book = _book(tmp_path, {'EUR_USD': 1.1})
    primary = SimpleNamespace(is_connected=True)
    bm = SimpleNamespace(get_broker=lambda name=None: book if name == 'forex_paper' else primary)
    agent = SimpleNamespace(components={'broker_manager': bm}, config={}, trading_halted=False)
    agent._forex_broker = TradingAgent._forex_broker.__get__(agent)
    assert TradingAgent._books(agent) == [primary]
    agent._loss_limit_blocks_buy = lambda *a: False
    assert TradingAgent._held_back_before_review(
        agent, 'EUR_USD', 'buy', 1.1, now=datetime(2026, 11, 25, 15, tzinfo=timezone.utc)) == 'self_managed_book'


def test_the_dashboard_shows_the_books_own_trades(tmp_path):
    from src.agent.paper_accounts import build_view
    prices = {'EUR_USD': 1.10}
    book = _book(tmp_path, prices)
    book.trade('EUR_USD', -10000)
    prices['EUR_USD'] = 1.00
    book.trade('EUR_USD', 10000)
    cfg = {'data_manager': {'forex_symbols': ['EUR_USD']}}
    view = build_view('forex', book, None, cfg, starting_capital=10000)
    assert view['stats']['closed_trades'] == 1 and view['stats']['wins'] == 1
    assert view['stats']['realised_pnl'] == pytest.approx(1000.0)
    assert [f['side'] for f in view['fills']] == ['buy', 'sell']
    assert view['book_summary']['max_leverage'] == 3.0
