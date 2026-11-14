"""NSE bonus issues, splits and dividends (src/agent/nse_corporate_actions.py).

Before: NSE prices were used raw. A bonus issue or an ex-dividend day looked
like a fall: it could trip the paper account's 8% stop-loss on a move that
was no loss, and it distorted the history the rotation strategy ranks on.
"""
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.agent import nse_corporate_actions as nca
from src.agent.nse_order_queue import NseOrderQueue
from src.agent.nse_paper_account import NsePaperAccount

TODAY = datetime.utcnow().date()
EX = TODAY - timedelta(days=10)


def _write(path, events):
    path.write_text(json.dumps({'events': events}))
    return str(path)


# ------------------------------------------------------------- the module

def test_ratios_read_as_shares_after_over_shares_before():
    assert nca.parse_ratio('bonus', '1:10') == pytest.approx(1.1)     # 11 shares for 10
    assert nca.parse_ratio('bonus', '1:2') == pytest.approx(1.5)
    assert nca.parse_ratio('split', '2:1') == 2.0
    assert nca.parse_ratio('split', '1:5') == 0.2                     # a consolidation
    assert nca.parse_ratio('bonus', 'one for ten') is None and nca.parse_ratio('bonus', '0:10') is None


def test_actions_load_from_both_files_and_bad_entries_are_skipped(tmp_path):
    actions = _write(tmp_path / 'a.json', [
        {'symbol': 'kcb', 'type': 'bonus', 'ratio': '1:10', 'ex_date': '2026-05-20'},
        {'symbol': 'EQTY', 'type': 'split', 'ratio': '2:1', 'ex_date': '2026-03-02'},
        {'symbol': 'BAD', 'type': 'bonus', 'ratio': 'lots', 'ex_date': '2026-01-01'},
        {'symbol': 'BAD', 'type': 'rights', 'ratio': '1:4', 'ex_date': '2026-01-01'},
        {'symbol': 'BAD', 'type': 'bonus', 'ratio': '1:4'}])
    dividends = _write(tmp_path / 'd.json', [
        {'symbol': 'SCOM', 'dividend_kes': 0.85, 'ex_date': '2026-07-28', 'payment_date': '2026-08-29'},
        {'symbol': 'ABSA', 'dividend_kes': 1.55, 'book_closure': '2026-04-10', 'payment_date': '2026-05-15'}])
    got = nca.load_actions(actions, dividends)
    assert [(a.symbol, a.kind) for a in got] == [
        ('EQTY', 'split'), ('ABSA', 'dividend'), ('KCB', 'bonus'), ('SCOM', 'dividend')]
    absa = next(a for a in got if a.symbol == 'ABSA')
    assert absa.ex_date == date(2026, 4, 7)            # three trading days before book closure


def test_a_bonus_issue_is_no_fall_in_the_adjusted_history():
    bonus = nca.Action('KCB', 'bonus', date(2026, 5, 20), share_factor=1.1)
    rows = [(date(2026, 5, 18), 44.5, 43.5, 44.0), (date(2026, 5, 19), 44.5, 43.5, 44.0),
            (date(2026, 5, 20), 40.5, 39.8, 40.0)]
    adjusted = nca.adjust_rows(rows, [bonus])
    assert [round(r[3], 4) for r in adjusted] == [40.0, 40.0, 40.0]
    assert adjusted[-1] == rows[-1]                      # on and after the ex-date: as traded


def test_a_dividend_leaves_only_the_unexplained_part_of_the_fall():
    div = nca.Action('ABSA', 'dividend', date(2026, 4, 7), dividend=2.0)
    rows = [(date(2026, 4, 6), 44.0, 44.0, 44.0), (date(2026, 4, 7), 40.0, 40.0, 40.0)]
    first = nca.adjust_rows(rows, [div])[0]
    assert first[3] == pytest.approx(42.0)               # 44 x (44 - 2) / 44


def test_an_action_outside_the_series_changes_nothing():
    rows = [(date(2026, 5, 18), 44.0, 44.0, 44.0), (date(2026, 5, 19), 44.0, 44.0, 44.0)]
    later = nca.Action('KCB', 'bonus', date(2026, 6, 1), share_factor=1.1)
    earlier = nca.Action('KCB', 'bonus', date(2026, 1, 1), share_factor=1.1)
    assert nca.adjust_rows(rows, [later, earlier]) == rows


def test_nse_history_is_read_adjusted(tmp_path):
    from src.agent.history_warmstart import read_nse_history
    lines = ['date,symbol,open,high,low,close,volume,change_pct,source']
    start = EX - timedelta(days=5)
    for i in range(10):
        d = start + timedelta(days=i)
        close = 44.0 if d < EX else 40.0
        lines.append(f'{d.isoformat()},KCB,{close},{close},{close},{close},1000,0,nse_pricelist')
    (tmp_path / 'KCB.csv').write_text('\n'.join(lines) + '\n')
    bonus = nca.Action('KCB', 'bonus', EX, share_factor=1.1)
    hist = read_nse_history(['KCB'], tmp_path, bars=60, actions=[bonus])['KCB']
    assert max(hist['close']) == pytest.approx(40.0) and min(hist['close']) == pytest.approx(40.0)
    assert hist['last_date'] == (start + timedelta(days=9)).isoformat()


# ------------------------------------------------- held prices while running

def test_an_action_that_takes_effect_while_running_rescales_held_prices_once():
    from src.agent.main import TradingAgent
    from src.agent.strategy_manager import StrategyManager
    from src.agent.volatility import VolatilityTracker
    with open('config/config.json') as f:
        cfg = json.load(f)
    sm = StrategyManager(cfg)
    sm.warm_start({'KCB': [44.0] * 150})
    vol = VolatilityTracker(length=14)
    vol.warm_start('KCB', [44.0] * 30, [44.5] * 30, [43.5] * 30)
    agent = SimpleNamespace(components={'strategy_manager': sm}, volatility=vol,
                            _nse_adjusted_through={'KCB': (EX - timedelta(days=1)).isoformat()})
    bonus = nca.Action('KCB', 'bonus', EX, share_factor=1.1)
    factor = TradingAgent._apply_nse_actions(agent, 'KCB', [bonus], today=EX)
    assert factor == pytest.approx(1 / 1.1)
    assert sm.strategies['nse_rotation'].historical_data['KCB'][-1] == pytest.approx(40.0)
    assert list(sm.regime_detector._closes['KCB'])[-1] == pytest.approx(40.0)
    assert vol._close['KCB'][-1] == pytest.approx(40.0) and vol._high['KCB'][-1] == pytest.approx(40.4545, rel=1e-4)
    assert TradingAgent._apply_nse_actions(agent, 'KCB', [bonus], today=EX + timedelta(days=1)) == 1.0


# ------------------------------------------------------------ the paper account

@pytest.fixture
def queue(tmp_path):
    q = NseOrderQueue(str(tmp_path / 'escalations.db'))
    started = (datetime.utcnow() - timedelta(days=30)).isoformat()
    q._conn.execute("INSERT INTO nse_paper_account (id, started_at) VALUES (1, ?)", (started,))
    q._conn.commit()
    yield q
    q.close()


def _account(queue, tmp_path, actions=(), dividends=()):
    cfg = {'nse_paper_trading': {
               'starting_capital_kes': 200000, 'cash_yield_pct': 0.0,
               'trading_limits': {'min_holding_days': 0, 'max_new_positions_per_week': 1000,
                                  'max_adv_fraction': 0, 'settlement_days': 0},
               'stop_loss': {'enabled': True, 'stop_loss_pct': 0.08, 'trailing_stop_pct': 0.10},
               'corporate_actions_path': _write(tmp_path / 'actions.json', list(actions)),
               'dividend_events_path': _write(tmp_path / 'dividends.json', list(dividends))},
           'costs': {'nse': {'commission_pct': 0.0, 'min_commission': 0.0, 'slippage_pct': 0.0}},
           'data_manager': {'nse_symbols': ['KCB']}}
    return NsePaperAccount(queue, cfg)


def _bought(paper, queue, qty, price, days_ago):
    tid = queue.create_ticket('KCB', 'buy', qty, suggested_limit_price=price)
    ok, _ = paper.fill(tid, 'buy', price, qty)
    assert ok
    when = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
    queue._conn.execute("UPDATE nse_order_tickets SET fill_at = ? WHERE id = ?", (when, tid))
    queue._conn.commit()


BONUS = {'symbol': 'KCB', 'type': 'bonus', 'ratio': '1:10', 'ex_date': EX.isoformat()}


def test_a_bonus_issue_adds_free_shares_at_no_cost(queue, tmp_path):
    paper = _account(queue, tmp_path, actions=[BONUS])
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    pos = paper.positions()['KCB']
    assert pos['quantity'] == 1100 and pos['cost_kes'] == pytest.approx(44000)
    assert pos['avg_cost_kes'] == pytest.approx(40.0, abs=1e-3)
    assert pos['entries'] == 1 and pos['last_entry_kes'] == pytest.approx(40.0)
    assert paper.cash() == pytest.approx(200000 - 44000 - paper.account_fees())   # no money moved


def test_a_buy_on_or_after_the_ex_date_gets_no_bonus(queue, tmp_path):
    paper = _account(queue, tmp_path, actions=[BONUS])
    _bought(paper, queue, 1000, 40.0, days_ago=5)
    assert paper.positions()['KCB']['quantity'] == 1000


def test_a_later_dividend_is_paid_on_the_bonus_shares_too(queue, tmp_path):
    div = {'symbol': 'KCB', 'dividend_kes': 1.0, 'ex_date': (EX + timedelta(days=3)).isoformat(),
           'payment_date': (EX + timedelta(days=5)).isoformat()}
    paper = _account(queue, tmp_path, actions=[BONUS], dividends=[div])
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    (paid,) = paper.dividends()
    assert paid['shares'] == 1100 and paid['gross_kes'] == pytest.approx(1100.0)


def test_a_bonus_issue_does_not_trip_the_stops(queue, tmp_path):
    paper = _account(queue, tmp_path, actions=[BONUS])
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    queue.paper_high('KCB', paper.states()['KCB'].opened_at, 44.0)    # the high, recorded before
    assert paper.stop_check('KCB', 40.0) is None                       # 9% below 44, 0% below 40
    assert queue.paper_high_peek('KCB', paper.states()['KCB'].opened_at) == pytest.approx(40.0)
    # A real fall still stops it: 8% below the adjusted cost of 40.
    assert paper.stop_check('KCB', 36.7)[0] == 'stop_loss'


def test_without_the_announcement_the_same_bonus_would_have_stopped_it_out(queue, tmp_path):
    paper = _account(queue, tmp_path)
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    assert paper.stop_check('KCB', 40.0)[0] == 'stop_loss'


def test_an_ex_dividend_drop_does_not_trip_the_stop_loss(queue, tmp_path):
    div = {'symbol': 'KCB', 'dividend_kes': 2.5, 'ex_date': EX.isoformat(),
           'payment_date': (EX + timedelta(days=20)).isoformat()}
    paper = _account(queue, tmp_path, dividends=[div])
    paper.close_before = lambda symbol, day: 44.0
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    # 40.3 is 8.4% below the cost, but 2.5 of the 3.7 fall is the dividend:
    # 2.9% below the cost scaled by (44 - 2.5) / 44.
    assert paper.stop_check('KCB', 40.3) is None
    levels = paper.stop_levels('KCB')
    assert levels['stop_loss_kes'] == pytest.approx(41.5 * 0.92, abs=0.01)


def test_the_stored_high_is_scaled_once_across_restarts(queue, tmp_path):
    paper = _account(queue, tmp_path, actions=[BONUS])
    _bought(paper, queue, 1000, 44.0, days_ago=20)
    opened = paper.states()['KCB'].opened_at
    queue.paper_high('KCB', opened, 44.0)
    paper.stop_check('KCB', 40.0)
    _account(queue, tmp_path, actions=[BONUS]).stop_check('KCB', 40.0)   # a restart
    assert queue.paper_high_peek('KCB', opened) == pytest.approx(40.0)


def test_the_weekly_screen_ranks_on_adjusted_closes(tmp_path):
    from src.agent.nse_screener import real_rows
    lines = ['date,symbol,open,high,low,close,volume,change_pct,source',
             f'{(EX - timedelta(days=1)).isoformat()},KCB,44,44,44,44,1000,0,nse_pricelist',
             f'{EX.isoformat()},KCB,40,40,40,40,1000,-9.1,nse_pricelist']
    (tmp_path / 'KCB.csv').write_text('\n'.join(lines) + '\n')
    bonus = nca.Action('KCB', 'bonus', EX, share_factor=1.1)
    assert [round(r['close'], 4) for r in real_rows('KCB', tmp_path, actions=[bonus])] == [40.0, 40.0]
    assert [r['close'] for r in real_rows('KCB', tmp_path, actions=[])] == [44.0, 40.0]
