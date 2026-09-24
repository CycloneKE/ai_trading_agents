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

COSTS = {'nse': {'commission_pct': 0.017, 'min_commission': 0.0, 'slippage_pct': 0.003}}


def _config(capital=200000):
    return {'nse_paper_trading': {'starting_capital_kes': capital}, 'costs': COSTS,
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


def test_a_holding_is_not_added_to(queue):
    paper = NsePaperAccount(queue, _config())
    _buy(paper, queue)
    assert paper.plan('SCOM', 'buy', 36.2, 50000) == (0, 'already_held')


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
    def __init__(self, actions):
        self.actions = list(actions)

    def generate_signals(self, data):
        return {'action': self.actions.pop(0), 'confidence': 0.8, 'position_size': 1.0}


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


def _agent(queue, actions, prices, capital=200000, llm=None):
    prices = list(prices)
    nse = SimpleNamespace(is_market_open=lambda: True,
                          get_quote=lambda s: {'symbol': s, 'price_kes': prices.pop(0),
                                               'volume': 1000, 'source': 'nse_ticker'})
    config = _config(capital)
    return SimpleNamespace(
        components={'nse_order_queue': queue, 'nse_paper_account': NsePaperAccount(queue, config),
                    'data_manager': SimpleNamespace(connectors={'nse': nse}),
                    'strategy_manager': _Strategies(actions), 'llm_orchestrator': llm},
        config=config, decision_journal=_Recorder(), order_journal=None,
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
        [None, 'already_held', 'already_held', 'already_held']
    assert agent.decision_journal.records[0]['executed'] is True


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
