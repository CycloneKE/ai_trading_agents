"""The agent's NSE paper account, and the rules its fills follow.

Auto paper fills used to be booked at the decision price with no cost, no
cash and no holding rules: a buy signal that lasted all day bought again
every cycle, and a sell signal with nothing held opened a short. These
tests drive the real NSE decision loop against a real ticket queue.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from src.agent.main import TradingAgent
from src.agent.nse_order_queue import NseOrderQueue
from src.agent.nse_paper_account import NsePaperAccount, order_size

COSTS = {'nse': {'commission_pct': 0.017, 'min_commission': 0.0, 'slippage_pct': 0.003,
                 'annual_fee_kes': 0}}


# The holding, turnover, liquidity and settlement limits have their own tests
# below; these mechanics tests buy and sell within one run, so they are off.
NO_LIMITS = {'min_holding_days': 0, 'max_new_positions_per_week': 1000,
             'max_adv_fraction': 0, 'settlement_days': 0}


def _config(capital=200000, limits=NO_LIMITS, cash_yield=0.0):
    # Interest on idle cash has its own tests; elsewhere it would move every
    # cash figure a test computes by hand.
    return {'nse_paper_trading': {'starting_capital_kes': capital, 'trading_limits': limits,
                                  'cash_yield_pct': cash_yield},
            'costs': COSTS,
            'data_manager': {'nse_symbols': ['SCOM'], 'nse_eval_interval': 1800},
            'nse_order_tickets': {'trade_notional_kes': 50000}}


@pytest.fixture
def queue(tmp_path):
    q = NseOrderQueue(str(tmp_path / 'escalations.db'))
    yield q
    q.close()


def _buy(paper, queue, symbol='SCOM', price=36.2, notional=50000):
    qty, reason = paper.plan(symbol, 'buy', price, notional)
    assert reason is None
    tid = queue.create_ticket(symbol, 'buy', qty, suggested_limit_price=price)
    ok, fill = paper.fill(tid, 'buy', price, qty)
    assert ok
    return qty, fill


# ------------------------------------------------------------- the account

def test_a_new_account_holds_its_starting_capital(queue):
    paper = NsePaperAccount(queue, _config())
    assert paper.enabled and paper.cash() == 200000 and paper.positions() == {}


def test_a_buy_pays_slippage_and_commission_out_of_cash(queue):
    paper = NsePaperAccount(queue, _config())
    qty, fill = _buy(paper, queue)
    assert fill['fill_price'] == 36.31  # 36.20 moved 0.3% against us, to the cent
    assert fill['fees_kes'] == pytest.approx(qty * 36.31 * 0.017, abs=0.01)
    assert paper.cash() == pytest.approx(200000 - qty * 36.31 - fill['fees_kes'], abs=0.01)
    # The order used as much of the KES 50,000 as fits, and no more.
    assert qty * 36.31 + fill['fees_kes'] <= 50000 < (qty + 1) * 36.31 * 1.017


def test_a_holding_that_has_not_proved_itself_is_not_added_to(queue):
    paper = NsePaperAccount(queue, _config())
    _buy(paper, queue)
    assert paper.plan('SCOM', 'buy', 36.2, 50000) == (0, 'add_not_profitable')
    # Up 4%: still short of the 5% step, and a sale now would not clear costs.
    assert paper.plan('SCOM', 'buy', 37.65, 50000) == (0, 'add_not_profitable')


def test_a_winning_holding_is_added_to_at_half_size(queue):
    paper = NsePaperAccount(queue, _config())
    first, _ = _buy(paper, queue)
    qty, reason = paper.plan('SCOM', 'buy', 40.0, 50000)
    assert reason is None
    assert qty * 40.12 * 1.017 == pytest.approx(25000, rel=0.01)  # add_size_pct 0.5


def test_each_add_needs_the_move_to_continue(queue):
    paper = NsePaperAccount(queue, _config())
    _buy(paper, queue)
    _buy(paper, queue, price=40.0, notional=25000)
    # 40.12 was the add's fill; the next add needs 5% above that.
    assert paper.plan('SCOM', 'buy', 41.5, 50000) == (0, 'add_not_profitable')
    assert paper.plan('SCOM', 'buy', 42.2, 50000)[1] in (None, 'position_cap')


def test_adds_stop_at_the_limit(queue):
    config = _config()
    config['nse_paper_trading']['add_to_winners'] = {'max_adds': 1, 'max_position_pct': 1.0}
    paper = NsePaperAccount(queue, config)
    _buy(paper, queue)
    _buy(paper, queue, price=40.0, notional=25000)
    assert paper.plan('SCOM', 'buy', 50.0, 50000) == (0, 'max_adds')


def test_a_position_never_outgrows_its_share_of_the_account(queue):
    config = _config()
    config['nse_paper_trading']['add_to_winners'] = {'max_position_pct': 0.25}
    paper = NsePaperAccount(queue, config)
    _buy(paper, queue)  # ~KES 50,000 of 200,000: already at 25%
    assert paper.plan('SCOM', 'buy', 40.0, 50000) == (0, 'position_cap')


def test_adding_can_be_switched_off(queue):
    config = _config()
    config['nse_paper_trading']['add_to_winners'] = {'enabled': False}
    paper = NsePaperAccount(queue, config)
    _buy(paper, queue)
    assert paper.plan('SCOM', 'buy', 40.0, 50000) == (0, 'already_held')


def test_a_closed_position_starts_counting_adds_afresh(queue):
    config = _config()
    config['nse_paper_trading']['add_to_winners'] = {'max_adds': 1, 'max_position_pct': 1.0}
    paper = NsePaperAccount(queue, config)
    qty, _ = _buy(paper, queue)
    add, _ = _buy(paper, queue, price=40.0, notional=25000)
    tid = queue.create_ticket('SCOM', 'sell', qty + add)
    paper.fill(tid, 'sell', 41.0, qty + add)
    _buy(paper, queue, price=41.0)
    assert paper.plan('SCOM', 'buy', 44.0, 50000)[1] is None


def test_a_sell_closes_the_whole_position_and_books_the_result(queue):
    paper = NsePaperAccount(queue, _config())
    qty, buy = _buy(paper, queue)
    assert paper.plan('SCOM', 'sell', 38.0, 50000) == (qty, None)
    tid = queue.create_ticket('SCOM', 'sell', qty, suggested_limit_price=38.0)
    ok, sell = paper.fill(tid, 'sell', 38.0, qty)
    assert ok and sell['fill_price'] == 37.89
    s = paper.summary()
    cost = qty * 36.31 + buy['fees_kes']
    proceeds = qty * 37.89 - sell['fees_kes']
    assert paper.positions() == {}
    assert s['realised_pnl_kes'] == pytest.approx(proceeds - cost, abs=0.02)
    assert s['cash_kes'] == pytest.approx(200000 + proceeds - cost, abs=0.02)
    assert s['fees_paid_kes'] == pytest.approx(buy['fees_kes'] + sell['fees_kes'], abs=0.01)


def test_nothing_held_means_nothing_to_sell(queue):
    paper = NsePaperAccount(queue, _config())
    assert paper.plan('SCOM', 'sell', 36.2, 50000) == (0, 'no_position')


def test_a_buy_is_cut_to_the_cash_available(queue):
    paper = NsePaperAccount(queue, _config(capital=1000))
    qty, reason = paper.plan('SCOM', 'buy', 36.2, 50000)
    assert reason is None and 0 < qty * 36.31 * 1.017 <= 1000


def test_no_cash_for_one_share_is_reported_as_such(queue):
    paper = NsePaperAccount(queue, _config(capital=30))
    assert paper.plan('SCOM', 'buy', 36.2, 50000) == (0, 'insufficient_cash')


def test_an_order_too_small_for_one_share_is_min_notional(queue):
    paper = NsePaperAccount(queue, _config())
    assert paper.plan('BAT', 'buy', 570.0, 300) == (0, 'min_notional')


def test_holdings_are_valued_at_market_and_unpriced_ones_at_cost(queue):
    paper = NsePaperAccount(queue, _config())
    qty, buy = _buy(paper, queue)
    cost = qty * 36.31 + buy['fees_kes']
    priced = paper.summary({'SCOM': 40.0})
    assert priced['holdings'][0]['market_value_kes'] == pytest.approx(qty * 40.0)
    assert priced['equity_kes'] == pytest.approx(priced['cash_kes'] + qty * 40.0, abs=0.01)
    unpriced = paper.summary({})
    assert unpriced['holdings'][0]['priced'] is False
    assert unpriced['equity_kes'] == pytest.approx(200000.0, abs=0.02)
    assert unpriced['holdings'][0]['market_value_kes'] == pytest.approx(cost, abs=0.01)


def test_the_account_keeps_its_opening_date_across_restarts(queue):
    first = NsePaperAccount(queue, _config()).started_at
    assert NsePaperAccount(queue, _config()).started_at == first


def test_fills_from_before_the_account_opened_are_not_its_own(queue):
    tid = queue.create_ticket('SCOM', 'sell', 500)
    queue.mark_filled(tid, fill_price=30.0, fill_quantity=500)  # an old short
    paper = NsePaperAccount(queue, _config())
    assert paper.cash() == 200000 and paper.positions() == {}


def test_without_capital_there_is_no_paper_account(queue):
    paper = NsePaperAccount(queue, {'costs': COSTS})
    assert not paper.enabled and paper.started_at is None


def test_the_holding_rules_apply_to_manual_tickets_too(queue):
    """No paper account: the recorded trading-book fills decide."""
    assert order_size('SCOM', 'sell', 36.2, 50000, queue, None) == (0, 'no_position')
    assert order_size('SCOM', 'buy', 36.2, 50000, queue, None) == (1381, None)
    tid = queue.create_ticket('SCOM', 'buy', 100)
    queue.mark_filled(tid, fill_price=36.2, fill_quantity=100)
    assert order_size('SCOM', 'buy', 36.2, 50000, queue, None) == (0, 'already_held')
    assert order_size('SCOM', 'sell', 36.2, 50000, queue, None) == (100, None)


def test_an_older_database_gains_the_fees_column(tmp_path):
    db = tmp_path / 'old.db'
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nse_order_tickets (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 " created_at TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,"
                 " quantity INTEGER NOT NULL, suggested_limit_price REAL, rationale TEXT,"
                 " ensemble_confidence REAL, llm_reasoning TEXT, status TEXT DEFAULT 'pending',"
                 " fill_price REAL, fill_quantity INTEGER, fill_at TEXT, operator_notes TEXT,"
                 " resolved_by TEXT)")
    conn.commit()
    conn.close()
    q = NseOrderQueue(str(db))
    tid = q.create_ticket('SCOM', 'buy', 10)
    ok, fill = q.mark_filled(tid, fill_price=36.0, fill_quantity=10, fees_kes=6.12)
    assert ok and fill['fees_kes'] == 6.12
    q.close()


# ---------------------------------------------------- the real decision loop

class _Strategies:
    """Each step: an action, or (action, confidence), or (action, confidence, votes)."""

    def __init__(self, actions):
        self.actions = list(actions)

    def generate_signals(self, data):
        step = self.actions.pop(0)
        action, conf, votes = (step, 0.8, None) if isinstance(step, str) else (tuple(step) + (None,))[:3]
        return {'action': action, 'confidence': conf, 'position_size': 1.0,
                'per_strategy': votes or {}}


class _LLM:
    def __init__(self):
        self.calls = 0

    def validate_trade(self, symbol, signals, data, news):
        self.calls += 1
        return dict(signals, reasoning='approved')


class _Recorder:
    def __init__(self):
        self.records = []

    def record(self, dec):
        self.records.append(dec)


def _agent(queue, actions, prices, capital=200000, llm=None, journal=None):
    prices = list(prices)
    nse = SimpleNamespace(is_market_open=lambda: True,
                          get_quote=lambda s: {'symbol': s, 'price_kes': prices.pop(0),
                                               'volume': 1000, 'source': 'nse_ticker'})
    config = _config(capital)
    return SimpleNamespace(
        components={'nse_order_queue': queue, 'nse_paper_account': NsePaperAccount(queue, config),
                    'data_manager': SimpleNamespace(connectors={'nse': nse}),
                    'strategy_manager': _Strategies(actions), 'llm_orchestrator': llm},
        config=config, decision_journal=_Recorder(), order_journal=journal,
        _nse_last_price={}, trading_halted=False, _last_nse_eval=0.0)


def _run(agent, cycles):
    for _ in range(cycles):
        agent._last_nse_eval = 0.0  # every call is a new evaluation window
        TradingAgent._evaluate_nse_symbols(agent)


def test_a_buy_signal_that_persists_buys_once(queue):
    """Previously: one fresh full-size buy every cycle while the signal held."""
    agent = _agent(queue, ['buy'] * 4, [36.2, 36.3, 36.4, 36.5])
    _run(agent, 4)
    buys = [f for f in queue.fills(book='trading') if f['side'] == 'buy']
    assert len(buys) == 1
    assert [d['skip_reason'] for d in agent.decision_journal.records] == \
        [None, 'add_not_profitable', 'add_not_profitable', 'add_not_profitable']
    assert agent.decision_journal.records[0]['executed'] is True


def test_each_cycle_records_the_days_account_value(queue):
    agent = _agent(queue, ['buy'], [36.2])
    _run(agent, 1)
    (today,) = queue.paper_equity_history()
    held = agent.components['nse_paper_account'].positions()['SCOM']['quantity']
    assert today['holdings_kes'] == pytest.approx(held * 36.2, abs=0.01)
    assert today['equity_kes'] < 200000  # the buy's costs, marked at the decision price


def test_the_loop_adds_to_a_winner_and_a_sell_closes_it_all(queue):
    agent = _agent(queue, ['buy', 'buy', 'sell'], [36.2, 40.0, 41.0])
    _run(agent, 3)
    first, add, sell = queue.fills(book='trading')
    assert (first['side'], add['side'], sell['side']) == ('buy', 'buy', 'sell')
    assert add['quantity'] < first['quantity']
    assert sell['quantity'] == first['quantity'] + add['quantity']
    assert agent.components['nse_paper_account'].positions() == {}


def test_a_sell_signal_with_nothing_held_does_not_short(queue):
    agent = _agent(queue, ['sell'], [36.2])
    _run(agent, 1)
    assert queue.fills() == [] and queue.get_pending() == []
    assert agent.decision_journal.records[0]['skip_reason'] == 'no_position'


def test_a_sell_after_a_buy_closes_it(queue):
    agent = _agent(queue, ['buy', 'sell'], [36.2, 38.0])
    _run(agent, 2)
    buy, sell = queue.fills(book='trading')
    assert (buy['side'], sell['side']) == ('buy', 'sell') and sell['quantity'] == buy['quantity']
    paper = agent.components['nse_paper_account']
    assert paper.positions() == {} and paper.summary()['fees_paid_kes'] > 0


def test_a_trade_the_book_cannot_take_never_costs_an_llm_call(queue):
    llm = _LLM()
    agent = _agent(queue, ['buy', 'buy', 'buy'], [36.2, 36.3, 36.4], llm=llm)
    _run(agent, 3)
    assert llm.calls == 1


def test_without_a_paper_account_tickets_wait_for_the_operator(queue):
    agent = _agent(queue, ['buy'], [36.2], capital=0)
    _run(agent, 1)
    assert queue.fills() == []
    (ticket,) = queue.get_pending()
    assert (ticket['symbol'], ticket['side']) == ('SCOM', 'buy')


# ----------------------------------------------------------------- stops

def test_a_position_that_falls_below_its_stop_loss_is_sold(queue):
    paper = NsePaperAccount(queue, _config())
    qty, _ = _buy(paper, queue)          # cost about 36.93 a share with fees
    assert paper.stop_check('SCOM', 34.2) is None
    reason, n, why = paper.stop_check('SCOM', 33.9)
    assert (reason, n) == ('stop_loss', qty) and 'below the cost' in why


def test_a_trailing_stop_follows_the_high(queue):
    paper = NsePaperAccount(queue, _config())
    _buy(paper, queue)
    assert paper.stop_check('SCOM', 42.0) is None     # sets the high
    assert paper.stop_check('SCOM', 38.0) is None     # 9.5% off the high
    assert paper.stop_check('SCOM', 37.7)[0] == 'trailing_stop'


def test_the_high_survives_a_restart(queue):
    _buy(NsePaperAccount(queue, _config()), queue)
    NsePaperAccount(queue, _config()).stop_check('SCOM', 42.0)
    assert NsePaperAccount(queue, _config()).stop_check('SCOM', 37.7)[0] == 'trailing_stop'


def test_nse_stops_scale_with_volatility_but_never_tighten_below_the_floor(queue):
    config = _config()
    config['risk_limits'] = {'stop_loss_atr_mult': 2.5, 'trailing_stop_atr_mult': 3.0}
    paper = NsePaperAccount(queue, config)
    assert paper.stop_distances(None) == (0.08, 0.10)
    assert paper.stop_distances(0.01) == (0.08, 0.10)          # 2.5% / 3% ATR stops: too tight
    assert paper.stop_distances(0.05) == pytest.approx((0.125, 0.15))


def test_the_loop_sells_a_stopped_position_without_asking_the_llm(queue):
    llm = _LLM()
    agent = _agent(queue, ['buy', 'hold'], [36.2, 33.5], llm=llm)
    _run(agent, 2)
    buy, sell = queue.fills(book='trading')
    assert sell['side'] == 'sell' and sell['quantity'] == buy['quantity']
    assert llm.calls == 1   # the entry only
    stop = agent.decision_journal.records[-1]
    assert stop['executed'] is True and 'stop_loss' in stop['per_strategy']


# ------------------------------------------------------------ partial exits

def test_the_loop_trims_on_a_weak_sell_once_a_day_and_closes_on_a_strong_one(queue):
    agent = _agent(queue, ['buy', ('sell', 0.6), ('sell', 0.6), ('sell', 0.9)],
                   [36.2, 36.4, 36.5, 36.6])
    _run(agent, 4)
    buy, trim, close = queue.fills(book='trading')
    assert trim['side'] == 'sell' and trim['quantity'] == buy['quantity'] // 2
    assert close['quantity'] == buy['quantity'] - trim['quantity']
    assert agent.decision_journal.records[2]['skip_reason'] == 'trimmed_today'
    assert agent.components['nse_paper_account'].positions() == {}


# ------------------------------------------------ credit for the strategies

def test_nse_trades_credit_the_strategies_that_voted_for_them(queue, tmp_path):
    from src.agent.order_journal import OrderJournal
    from src.agent.strategy_attribution import compute_attribution
    journal = OrderJournal(db_path=str(tmp_path / 'orders.db'))
    votes = {'momentum': {'action': 'buy', 'confidence': 0.9},
             'rsi_strategy': {'action': 'buy', 'confidence': 0.3}}
    agent = _agent(queue, [('buy', 0.8, votes), 'hold'], [36.2, 33.5], journal=journal)
    _run(agent, 2)   # a buy, then a stop-loss exit
    out = compute_attribution(journal.filled_orders())
    assert set(out) == {'momentum', 'rsi_strategy'}      # nothing for 'stop_loss' or 'nse_manual'
    assert out['momentum']['closed_trades'] == 1 and out['momentum']['realized_pnl'] < 0
    assert out['momentum']['realized_pnl'] == pytest.approx(3 * out['rsi_strategy']['realized_pnl'], rel=0.01)
    assert out['momentum']['trade_returns'][0] < -0.05


def test_the_brokers_yearly_account_fee_is_charged_each_account_year(queue):
    from datetime import datetime, timedelta
    cfg = {**_config(), 'costs': {'nse': {**COSTS['nse'], 'annual_fee_kes': 200}}}
    paper = NsePaperAccount(queue, cfg)
    assert paper.cash() == 199800.0 and paper.summary()['account_fees_kes'] == 200
    opened = datetime.fromisoformat(paper.started_at[:19])
    assert paper.account_fees(opened + timedelta(days=364)) == 200
    assert paper.account_fees(opened + timedelta(days=366)) == 400


# ------------------------------------------------------------ trading limits

LIMITS = {'min_holding_days': 30, 'max_new_positions_per_week': 2,
          'max_adv_fraction': 0.10, 'settlement_days': 3}


def _buy_now(paper, queue, symbol, price=10.0, notional=20000):
    qty, why = paper.plan(symbol, 'buy', price, notional)
    assert why is None, why
    tid = queue.create_ticket(symbol, 'buy', qty, suggested_limit_price=price)
    paper.fill(tid, 'buy', price, qty)
    return qty


def test_a_signal_cannot_sell_before_the_minimum_holding_period(queue):
    from datetime import datetime, timedelta
    paper = NsePaperAccount(queue, _config(limits=LIMITS))
    _buy_now(paper, queue, 'SCOM')
    assert paper.plan('SCOM', 'sell', 10.5, 0, confidence=0.9) == (0, 'min_holding')
    later = datetime.utcnow() + timedelta(days=31)
    qty, why = paper.plan('SCOM', 'sell', 10.5, 0, confidence=0.9, now=later)
    assert why is None and qty > 0


def test_a_stop_still_sells_inside_the_holding_period(queue):
    paper = NsePaperAccount(queue, _config(limits=LIMITS))
    qty = _buy_now(paper, queue, 'SCOM', price=10.0)
    reason, shares, _ = paper.stop_check('SCOM', 9.0)
    assert reason == 'stop_loss' and shares == qty


def test_new_positions_are_limited_each_week_but_adds_are_not(queue):
    paper = NsePaperAccount(queue, _config(limits=LIMITS))
    _buy_now(paper, queue, 'SCOM')
    _buy_now(paper, queue, 'KCB')
    assert paper.plan('EQTY', 'buy', 10.0, 20000) == (0, 'turnover_budget')
    assert paper.new_positions_since(__import__('datetime').datetime(2000, 1, 1)) == 2


def test_an_order_is_capped_at_a_tenth_of_average_daily_volume(queue):
    paper = NsePaperAccount(queue, _config(limits=LIMITS))
    assert paper.plan('SCOM', 'buy', 10.0, 50000, adv=20000) == (2000, None)
    assert paper.plan('SCOM', 'buy', 10.0, 50000, adv=5) == (0, 'liquidity_cap')
    qty, why = paper.plan('SCOM', 'buy', 10.0, 50000, adv=None)  # no volume data: no cap
    assert why is None and qty > 2000


def test_sale_proceeds_are_not_spendable_until_settled(queue):
    from datetime import datetime, timedelta
    from src.agent.nse_paper_account import trading_days_since
    limits = {**LIMITS, 'min_holding_days': 0}
    paper = NsePaperAccount(queue, _config(capital=30000, limits=limits))
    qty = _buy_now(paper, queue, 'SCOM', price=10.0, notional=29000)
    tid = queue.create_ticket('SCOM', 'sell', qty, suggested_limit_price=10.0)
    paper.fill(tid, 'sell', 10.0, qty)
    s = paper.summary()
    assert s['unsettled_kes'] > 25000
    assert s['available_cash_kes'] == pytest.approx(s['cash_kes'] - s['unsettled_kes'], abs=0.01)
    assert paper.plan('KCB', 'buy', 5000.0, 20000)[1] == 'insufficient_cash'
    # Three trading days later the cash is back and the same order fits.
    later = datetime.utcnow() + timedelta(days=5)
    assert paper._ledger(later)['unsettled'] == 0
    qty, why = paper.plan('KCB', 'buy', 5000.0, 20000, now=later)
    assert why is None and qty > 0
    # Friday's sale settles on Wednesday.
    assert trading_days_since('2026-09-25T10:00:00', datetime(2026, 9, 30, 9)) == 3


# ------------------------------------------------------------------ dividends

def _with_events(tmp_path, events, **extra):
    path = tmp_path / 'events.json'
    path.write_text(__import__('json').dumps({'events': events}))
    cfg = _config()
    cfg['nse_paper_trading'].update({'dividend_events_path': str(path), **extra})
    return cfg


def test_a_dividend_is_paid_net_of_withholding_tax_on_its_payment_date(queue, tmp_path):
    from datetime import datetime, timedelta
    today = datetime.utcnow()
    ex = (today + timedelta(days=10)).date().isoformat()
    pay = (today + timedelta(days=30)).date().isoformat()
    cfg = _with_events(tmp_path, [{'symbol': 'SCOM', 'dividend_kes': 1.2, 'ex_date': ex,
                                   'payment_date': pay}])
    paper = NsePaperAccount(queue, cfg)
    qty = _buy_now(paper, queue, 'SCOM', price=10.0, notional=20000)
    cash = paper.cash()
    assert paper.dividends() == []                       # not paid yet
    later = today + timedelta(days=31)
    (d,) = paper.dividends(later)
    assert (d['shares'], d['gross_kes']) == (qty, round(qty * 1.2, 2))
    assert d['tax_kes'] == pytest.approx(d['gross_kes'] * 0.05, abs=0.01)
    assert paper._ledger(later)['cash'] == pytest.approx(cash + d['net_kes'], abs=0.01)


def test_shares_bought_on_or_after_the_ex_date_do_not_qualify(queue, tmp_path):
    from datetime import datetime, timedelta
    today = datetime.utcnow()
    cfg = _with_events(tmp_path, [{'symbol': 'SCOM', 'dividend_kes': 1.2,
                                   'ex_date': (today - timedelta(days=1)).date().isoformat(),
                                   'payment_date': today.date().isoformat()}])
    paper = NsePaperAccount(queue, cfg)
    _buy_now(paper, queue, 'SCOM')
    assert paper.dividends(today + timedelta(days=1)) == []


def test_without_an_ex_date_book_closure_less_three_trading_days_is_used(queue, tmp_path):
    from datetime import datetime
    from src.agent.nse_paper_account import subtract_trading_days
    assert subtract_trading_days(datetime(2026, 7, 31), 3).date().isoformat() == '2026-07-28'
    cfg = _with_events(tmp_path, [{'symbol': 'SCOM', 'dividend_kes': 1.0,
                                   'book_closure': '2026-07-31', 'payment_date': '2026-08-29'},
                                  {'symbol': 'KCB', 'dividend_kes': 1.0}])  # incomplete: ignored
    paper = NsePaperAccount(queue, cfg)
    assert [e['symbol'] for e in paper.dividend_events()] == ['SCOM']


# ------------------------------------------------------- interest on cash

def test_idle_cash_earns_the_rate_daily_after_withholding_tax():
    from datetime import datetime
    from src.agent.nse_paper_account import cash_interest
    start = datetime(2026, 1, 1)
    net, tax = cash_interest(start, datetime(2027, 1, 1), 100000, [], 0.0878)
    # 8.78% less 15% tax is 7.463% a year, compounded daily.
    assert net == pytest.approx(100000 * ((1 + 0.0878 * 0.85 / 365) ** 365 - 1), abs=0.01)
    assert tax == pytest.approx(net / 0.85 * 0.15, rel=1e-3)
    assert cash_interest(start, datetime(2026, 1, 1, 23), 100000, [], 0.0878) == (0.0, 0.0)


def test_cash_spent_on_shares_stops_earning_from_that_day():
    from datetime import datetime
    from src.agent.nse_paper_account import cash_interest
    start = datetime(2026, 1, 1)
    spent_day_one = cash_interest(start, datetime(2026, 1, 11), 100000,
                                  [(datetime(2026, 1, 1, 9), -100000)], 0.10)
    assert spent_day_one == (0.0, 0.0)
    half = cash_interest(start, datetime(2026, 1, 11), 100000,
                         [(datetime(2026, 1, 1, 9), -50000)], 0.10)
    whole = cash_interest(start, datetime(2026, 1, 11), 100000, [], 0.10)
    assert half[0] == pytest.approx(whole[0] / 2, rel=1e-3)


def test_the_account_credits_interest_to_cash_and_reports_it(queue):
    from datetime import datetime, timedelta
    paper = NsePaperAccount(queue, _config(cash_yield=0.0878))
    opened = datetime.fromisoformat(paper.started_at[:19])
    ledger = paper._ledger(opened + timedelta(days=30))
    assert ledger['interest_net'] > 0
    assert ledger['cash'] == pytest.approx(200000 + ledger['interest_net'], abs=0.01)
    assert paper.summary()['cash_yield_pct'] == 0.0878


def test_the_cash_rate_defaults_to_the_benchmark_tbill_rate(queue):
    cfg = _config()
    del cfg['nse_paper_trading']['cash_yield_pct']
    cfg['benchmarks'] = {'tbill_rate_pct': 0.09}
    assert NsePaperAccount(queue, cfg).cash_yield == 0.09


def test_a_new_position_after_a_sell_out_has_its_own_average_price(queue):
    for side, qty, px in (('buy', 100, 40.0), ('sell', 100, 45.0), ('buy', 50, 20.0)):
        t = queue.create_ticket('KCB', side, qty, suggested_limit_price=px, book='long_term')
        queue.mark_filled(t, fill_price=px, fill_quantity=qty)
    assert queue.positions(book='long_term')['KCB'] == {'quantity': 50, 'avg_entry_price_kes': 20.0}
