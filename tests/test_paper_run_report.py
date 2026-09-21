"""The weekly read on a paper run.

The report's job is to make an absence visible. These tests pin the cases
where a naive reading would be wrong: a run that stopped, a run that is
deciding but never trading, and a run whose P&L is really one lucky symbol.
"""
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'paper_run_report', os.path.join(_ROOT, 'scripts', 'paper_run_report.py'))
report = importlib.util.module_from_spec(_SPEC)
sys.modules['paper_run_report'] = report
_SPEC.loader.exec_module(report)


class Args:
    """Stands in for the argparse namespace build() reads."""

    def __init__(self, **kw):
        self.since = None
        self.orders = None
        self.decisions = None
        self.manifest = None
        self.config = 'config/config.json'
        self.out = None
        self.json = False
        self.__dict__.update(kw)


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def journals(tmp_path, monkeypatch):
    """Real journals at real paths, written through the real classes.

    Hand-rolling the schema here would let the report drift away from what the
    agent actually writes, which is the one thing these tests exist to catch.
    """
    monkeypatch.setattr(report, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(report, 'RUNS_DIR', tmp_path / 'paper_runs')
    from src.agent.decision_journal import DecisionJournal
    from src.agent.order_journal import OrderJournal
    oj = OrderJournal(db_path=str(tmp_path / 'order_journal.db'))
    dj = DecisionJournal(db_path=str(tmp_path / 'decision_journal.db'),
                         heartbeat_cycles=1)
    yield tmp_path, oj, dj
    oj.close()
    dj.close()


def _fill(oj, coid, symbol, side, qty, price, strategy='momentum'):
    oj.record_intent(coid, symbol, side, qty, 'market', strategy=strategy)
    oj.mark_submitted(coid, f'broker-{coid}')
    oj.mark_final(coid, 'filled', qty, price)


def _manifest(tmp_path, **over):
    runs = tmp_path / 'paper_runs'
    runs.mkdir(exist_ok=True)
    body = {'started_at': datetime.now(timezone.utc).isoformat(),
            'planned_days': 90, 'git_commit': 'abc123def456',
            'universe': {'loop_symbols': ['AAPL'], 'slow_pass_symbols': []},
            'started_with_blockers': [], 'warnings_at_start': []}
    body.update(over)
    path = runs / 'run_20260101T000000Z.json'
    path.write_text(json.dumps(body))
    return path


# ------------------------------------------------------------ FIFO matching

def test_round_trips_match_first_in_first_out():
    fills = [
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 10,
         'filled_avg_price': 100.0, 'strategy': 'momentum', 'created_at': 'a'},
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 10,
         'filled_avg_price': 120.0, 'strategy': 'rsi', 'created_at': 'b'},
        {'symbol': 'AAPL', 'side': 'sell', 'filled_quantity': 10,
         'filled_avg_price': 110.0, 'strategy': 'stop', 'created_at': 'c'},
    ]
    trips, open_lots = report.round_trips(fills)
    assert len(trips) == 1
    # The first lot is the one closed, so the profit is 110 - 100, not 110 - 120.
    assert trips[0]['pnl'] == pytest.approx(100.0)
    assert open_lots['AAPL'][0]['price'] == 120.0


def test_a_sell_spanning_two_lots_produces_two_trips():
    fills = [
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 5,
         'filled_avg_price': 100.0, 'strategy': 'momentum'},
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 5,
         'filled_avg_price': 200.0, 'strategy': 'rsi'},
        {'symbol': 'AAPL', 'side': 'sell', 'filled_quantity': 10,
         'filled_avg_price': 150.0, 'strategy': 'exit'},
    ]
    trips, open_lots = report.round_trips(fills)
    assert len(trips) == 2
    assert sum(t['pnl'] for t in trips) == pytest.approx(0.0)
    assert not open_lots


def test_pnl_is_attributed_to_the_strategy_that_opened_the_position():
    """A trailing stop closes the trade but did not choose it."""
    fills = [
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 1,
         'filled_avg_price': 100.0, 'strategy': 'momentum'},
        {'symbol': 'AAPL', 'side': 'sell', 'filled_quantity': 1,
         'filled_avg_price': 90.0, 'strategy': 'trailing_stop'},
    ]
    trips, _ = report.round_trips(fills)
    assert trips[0]['strategy'] == 'momentum'
    by_strategy = report.aggregate(trips, 'strategy')
    assert 'trailing_stop' not in by_strategy
    assert by_strategy['momentum']['pnl'] == pytest.approx(-10.0)


def test_unfilled_and_unpriced_orders_are_ignored():
    fills = [
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 0,
         'filled_avg_price': 100.0},
        {'symbol': 'AAPL', 'side': 'buy', 'filled_quantity': 5,
         'filled_avg_price': None},
    ]
    trips, open_lots = report.round_trips(fills)
    assert not trips and not open_lots


def test_a_sell_with_no_matching_lot_does_not_invent_a_trip():
    """Short sales and out-of-window opens must not become phantom profit."""
    fills = [{'symbol': 'AAPL', 'side': 'sell', 'filled_quantity': 5,
              'filled_avg_price': 100.0}]
    trips, _ = report.round_trips(fills)
    assert trips == []


# -------------------------------------------------------------- aggregation

def test_profit_factor_and_win_rate():
    trips = [{'symbol': 'A', 'strategy': 's', 'pnl': 200.0},
             {'symbol': 'A', 'strategy': 's', 'pnl': -100.0},
             {'symbol': 'A', 'strategy': 's', 'pnl': 100.0}]
    a = report.aggregate(trips, 'symbol')['A']
    assert a['trips'] == 3
    assert a['win_rate'] == pytest.approx(2 / 3)
    assert a['profit_factor'] == pytest.approx(3.0)


def test_profit_factor_is_infinite_when_nothing_lost():
    a = report.aggregate([{'symbol': 'A', 'strategy': 's', 'pnl': 5.0}],
                         'symbol')['A']
    assert a['profit_factor'] == float('inf')


# ------------------------------------------------------- reading a live run

def test_a_run_that_decides_but_never_trades_is_named_as_such(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    for cycle in range(5):
        dj.record({'symbol': 'AAPL', 'cycle': cycle, 'action': 'hold',
                   'skip_reason': 'hold', 'ensemble_confidence': 0.05})
    rep = report.build(Args())
    assert rep['orders']['total'] == 0
    assert rep['decisions']['total'] > 0
    text = report.render(rep, 'config/config.json')
    assert 'deciding, but has placed no orders' in text
    assert 'No evidence yet' in text


def test_fallback_prices_are_reported_as_degradation_not_strategy(journals):
    """The difference between "found nothing" and "could not see"."""
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    for cycle in range(4):
        dj.record({'symbol': 'AAPL', 'cycle': cycle, 'action': 'buy',
                   'skip_reason': 'fallback_price', 'ensemble_confidence': 0.7})
    text = report.render(report.build(Args()), 'config/config.json')
    assert 'DEGRADED' in text
    assert 'blocked by degradation, not by the strategy' in text
    assert 'refused to trade, and correctly learned nothing' in text


def test_stuck_orders_are_surfaced(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    oj.record_intent('stuck-1', 'AAPL', 'buy', 10, 'market', strategy='momentum')
    oj.record_intent('failed-1', 'MSFT', 'buy', 10, 'market', strategy='momentum')
    oj.mark_failed('failed-1', 'broker timeout')
    rep = report.build(Args())
    text = report.render(rep, 'config/config.json')
    assert 'never reached a final status' in text
    assert 'failed or were aborted' in text


def test_realised_pnl_comes_from_the_journal_end_to_end(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    _fill(oj, 'b1', 'AAPL', 'buy', 10, 100.0, 'momentum')
    _fill(oj, 's1', 'AAPL', 'sell', 10, 115.0, 'trailing_stop')
    _fill(oj, 'b2', 'MSFT', 'buy', 5, 200.0, 'rsi_strategy')
    rep = report.build(Args())
    assert rep['pnl']['closed_trips'] == 1
    assert rep['pnl']['realised'] == pytest.approx(150.0)
    assert rep['pnl']['by_strategy']['momentum']['pnl'] == pytest.approx(150.0)
    # The open MSFT lot is cost basis, never counted as profit.
    assert rep['pnl']['open_positions']['MSFT']['cost_basis'] == pytest.approx(1000.0)
    text = report.render(rep, 'config/config.json')
    assert 'Too early to judge' in text


def test_a_run_started_with_blockers_says_so(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path, started_with_blockers=['real market data'])
    dj.record({'symbol': 'AAPL', 'cycle': 0, 'action': 'hold',
               'skip_reason': 'hold'})
    text = report.render(report.build(Args()), 'config/config.json')
    assert 'started with known blockers' in text
    assert 'results are suspect' in text


def test_a_silent_run_is_called_stopped_not_quiet(journals, monkeypatch):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    dj.record({'symbol': 'AAPL', 'cycle': 0, 'action': 'hold',
               'skip_reason': 'hold'})
    # Age the only row past the staleness threshold.
    old = (datetime.utcnow() - timedelta(hours=48)).isoformat()
    conn = sqlite3.connect(str(tmp_path / 'decision_journal.db'))
    conn.execute('UPDATE decisions SET ts = ?', (old,))
    conn.commit()
    conn.close()
    text = report.render(report.build(Args()), 'config/config.json')
    assert 'most likely stopped' in text


def test_missing_manifest_is_reported_rather_than_assumed(journals):
    tmp_path, oj, dj = journals
    dj.record({'symbol': 'AAPL', 'cycle': 0, 'action': 'hold',
               'skip_reason': 'hold'})
    rep = report.build(Args())
    assert rep['manifest'] is None
    text = report.render(rep, 'config/config.json')
    assert 'No run manifest found' in text


def test_empty_journals_render_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(report, 'RUNS_DIR', tmp_path / 'paper_runs')
    rep = report.build(Args())
    text = report.render(rep, 'config/config.json')
    assert 'recorded nothing at all' in text


def test_the_window_excludes_older_rows(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    dj.record({'symbol': 'AAPL', 'cycle': 0, 'action': 'hold',
               'skip_reason': 'hold'})
    dj.record({'symbol': 'MSFT', 'cycle': 1, 'action': 'buy',
               'skip_reason': 'below_confidence'})
    conn = sqlite3.connect(str(tmp_path / 'decision_journal.db'))
    conn.execute("UPDATE decisions SET ts = ? WHERE symbol = 'AAPL'",
                 ((datetime.utcnow() - timedelta(days=30)).isoformat(),))
    conn.commit()
    conn.close()
    assert report.build(Args())['decisions']['total'] == 2
    assert report.build(Args(since=7))['decisions']['total'] == 1


def test_journals_are_opened_read_only(journals):
    """Safe to run against a live agent: the report must never write."""
    tmp_path, oj, dj = journals
    conn = report.read_only(str(tmp_path / 'order_journal.db'))
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO orders (client_order_id, symbol, side,"
                     " quantity, order_type, status, created_at, updated_at)"
                     " VALUES ('x','A','buy',1,'market','intent','t','t')")
    conn.close()


# --------------------------------------------------------- concentration

def test_profit_concentrated_in_one_symbol_is_flagged():
    """A backtest here once showed a convincing edge that was entirely AAPL."""
    by_symbol = {
        'AAPL': {'pnl': 9000.0, 'gross_win': 9000.0, 'gross_loss': 0.0},
        'MSFT': {'pnl': 500.0, 'gross_win': 500.0, 'gross_loss': 0.0},
        'GOOG': {'pnl': -200.0, 'gross_win': 0.0, 'gross_loss': 200.0},
    }
    note = report.concentration_note(by_symbol)
    assert 'AAPL alone' in note
    assert 'anecdote' in note


def test_evenly_spread_profit_is_not_flagged():
    by_symbol = {
        'AAPL': {'pnl': 1000.0, 'gross_win': 1000.0, 'gross_loss': 0.0},
        'MSFT': {'pnl': 900.0, 'gross_win': 900.0, 'gross_loss': 0.0},
        'GOOG': {'pnl': 800.0, 'gross_win': 800.0, 'gross_loss': 0.0},
    }
    assert report.concentration_note(by_symbol) == ''


def test_a_losing_book_is_not_described_as_concentrated():
    by_symbol = {'AAPL': {'pnl': -500.0, 'gross_win': 0.0, 'gross_loss': 500.0}}
    assert report.concentration_note(by_symbol) == ''


def test_enough_trips_switches_the_verdict_to_a_real_reading(journals):
    tmp_path, oj, dj = journals
    _manifest(tmp_path)
    for i in range(25):
        _fill(oj, f'b{i}', 'AAPL', 'buy', 1, 100.0, 'momentum')
        _fill(oj, f's{i}', 'AAPL', 'sell', 1, 110.0 if i % 2 else 95.0, 'exit')
    rep = report.build(Args())
    assert rep['pnl']['closed_trips'] == 25
    text = report.render(rep, 'config/config.json')
    assert 'Too early to judge' not in text
    assert 'profit factor' in text
    assert 'money market fund' in text
