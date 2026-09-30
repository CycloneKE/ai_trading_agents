"""Learning from every signal's later price move (src/agent/signal_ledger.py),
not only from closed trades.

What matters: outcomes are recorded once known and never change; a rising
market does not make a strategy look skilful; nothing is tilted on too little
evidence or outside the guardrails; and with the tilt off the strategy manager
behaves exactly as it did.
"""
import json
import os
import random
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agent import chart_data, scorecard
from src.agent import signal_ledger as sl
from src.agent import strategy_tuner as st
from src.agent.decision_journal import DecisionJournal
from src.agent.guardrails import signal_tilt
from src.agent.strategy_manager import StrategyManager
from src.agent.technical_strategy import TechnicalStrategy
from src.utils.config_validator import load_config

CONFIG = load_config('config/config.json')
START = date(2026, 1, 5)


def closes(values, start=START):
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def walk(n=300, seed=3):
    rng, v, out = random.Random(seed), 100.0, []
    for _ in range(n):
        v *= 1 + rng.gauss(0, 0.01)
        out.append(v)
    return out


def row(symbol, day, action='hold', per=None, price=100.0, version='v1'):
    ts = datetime.combine(START + timedelta(days=day), datetime.min.time()).replace(hour=15).isoformat()
    return {'ts': ts, 'symbol': symbol, 'action': action, 'price': price, 'per_strategy': per or {},
            'code_version': version}


# ------------------------------------------------------------------ votes

def test_a_row_carries_the_ensembles_vote_and_each_strategys_own():
    d = row('AAPL', 1, 'buy', {'momentum': {'action': 'buy', 'confidence': 0.7},
                              'mean_reversion': {'action': 'sell', 'confidence': 0.4},
                              'rsi_strategy': {'action': 'hold', 'confidence': 0.0}})
    assert sorted(sl.votes(d)) == [('ensemble', 'buy'), ('mean_reversion', 'sell'), ('momentum', 'buy')]


def test_a_strategy_that_voted_is_counted_even_when_the_ensemble_held():
    d = row('AAPL', 1, 'hold', {'momentum': {'action': 'buy', 'confidence': 0.3}})
    assert sl.votes(d) == [('momentum', 'buy')]


def test_abstentions_and_empty_votes_are_not_signals():
    d = row('BTC-USD', 1, 'hold', {'crypto_trend': {'action': 'buy', 'confidence': 0.9, 'abstain': True},
                                  'momentum': {'action': 'buy', 'confidence': 0.0}, 'x': 'garbage'})
    assert sl.votes(d) == []
    assert sl.votes({'action': None, 'per_strategy': None}) == []


def test_a_signal_that_persists_is_counted_once_a_day_per_source():
    per = {'momentum': {'action': 'buy', 'confidence': 0.6}}
    rows = [row('AAPL', 1, 'buy', per)] * 5 + [row('AAPL', 2, 'buy', per), row('EUR_USD', 1, 'hold', per)]
    sigs = sl.signals_from(rows, CONFIG)
    assert sorted((s['symbol'], s['day'].day, s['source']) for s in sigs) == [
        ('AAPL', 6, 'ensemble'), ('AAPL', 6, 'momentum'), ('AAPL', 7, 'ensemble'), ('AAPL', 7, 'momentum'),
        ('EUR_USD', 6, 'momentum')]
    assert {s['symbol']: s['market'] for s in sigs} == {'AAPL': 'us_equity', 'EUR_USD': 'forex'}
    assert all(s['version'] == 'v1' for s in sigs)


# --------------------------------------------------------------- outcomes

def test_only_outcomes_that_are_known_are_recorded_and_a_sell_is_judged_by_a_fall():
    vals = [100.0] * 10 + [110.0] * 10                       # a jump at day 10
    sigs = [{'symbol': 'AAPL', 'day': START + timedelta(days=d), 'source': 'ensemble', 'action': a,
             'market': 'us_equity', 'version': 'v1'} for d, a in ((6, 'buy'), (6, 'sell'), (19, 'buy'))]
    rows = sl.outcomes_for(sigs, lambda s: closes(vals), horizons=(5,))
    by = {((r['day'] - START).days, r['action']): r for r in rows}
    assert set(by) == {(6, 'buy'), (6, 'sell')}              # day 19's outcome is not known yet
    assert by[(6, 'buy')]['ret'] == pytest.approx(0.10) and by[(6, 'sell')]['ret'] == pytest.approx(-0.10)


def test_the_chance_rate_is_how_often_the_symbol_went_up_anyway():
    vals = [100 * 1.004 ** i for i in range(60)]              # only ever rises
    sigs = [{'symbol': 'AAPL', 'day': START + timedelta(days=3), 'source': 'ensemble', 'action': 'buy',
             'market': 'us_equity', 'version': 'v1'},
            {'symbol': 'AAPL', 'day': START + timedelta(days=3), 'source': 'ensemble', 'action': 'sell',
             'market': 'us_equity', 'version': 'v1'}]
    rows = {r['action']: r for r in sl.outcomes_for(sigs, lambda s: closes(vals), horizons=(5,))}
    assert rows['buy']['chance'] == 1.0 and rows['sell']['chance'] == 0.0


def test_a_symbol_with_no_price_history_produces_nothing_rather_than_an_error():
    sigs = [{'symbol': 'ZZZ', 'day': START, 'source': 'ensemble', 'action': 'buy', 'market': 'us_equity', 'version': 'v1'}]
    assert sl.outcomes_for(sigs, lambda s: {}) == []
    assert sl.outcomes_for(sigs, lambda s: 1 / 0) == []


# ----------------------------------------------------------------- ledger

def outcome(symbol='AAPL', day=1, source='momentum', action='buy', horizon=5, ret=0.02, chance=0.5, market='us_equity'):
    return {'symbol': symbol, 'day': START + timedelta(days=day), 'source': source, 'action': action,
            'horizon': horizon, 'ret': ret, 'chance': chance, 'market': market, 'version': 'v1'}


def test_an_outcome_is_recorded_once_and_never_changes(tmp_path):
    ledger = sl.SignalLedger(tmp_path / 'l.db')
    assert ledger.record([outcome(ret=0.02), outcome(day=2)]) == 2
    assert ledger.record([outcome(ret=-0.5), outcome(day=3)]) == 1          # day 1 again is ignored
    got = {r['day']: r['ret'] for r in ledger.rows()}
    assert got[str(START + timedelta(days=1))] == 0.02 and len(got) == 3
    ledger.close()
    assert len(sl.read_ledger(str(tmp_path / 'l.db'))) == 3                 # it survives a restart
    assert [r['day'] for r in sl.read_ledger(str(tmp_path / 'l.db'), str(START + timedelta(days=3)))] == [str(START + timedelta(days=3))]


def test_a_missing_ledger_or_journal_reads_as_empty(tmp_path):
    assert sl.read_ledger(str(tmp_path / 'nope.db')) == []
    assert sl.read_decisions(str(tmp_path / 'nope.db')) == []


def journal_with(tmp_path, rows):
    dj = DecisionJournal(str(tmp_path / 'decision_journal.db'))
    for d in rows:
        dj._conn.execute(
            "INSERT INTO decisions (ts, cycle, symbol, action, executed, skip_reason, price, per_strategy_json, code_version)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (d['ts'], 1, d['symbol'], d['action'], 0, 'hold', d['price'], json.dumps(d['per_strategy']), d['code_version']))
    dj._conn.commit()
    return str(tmp_path / 'decision_journal.db')


def test_only_journal_rows_with_a_directional_vote_are_read(tmp_path):
    path = journal_with(tmp_path, [
        row('AAPL', 1, 'hold', {}), row('AAPL', 2, 'buy', {}),
        row('MSFT', 3, 'hold', {'momentum': {'action': 'sell', 'confidence': 0.5}}),
        row('NVDA', 4, 'hold', {'momentum': {'action': 'hold', 'confidence': 0.0}})])
    got = sl.read_decisions(path)
    assert [(d['symbol'], d['action']) for d in got] == [('AAPL', 'buy'), ('MSFT', 'hold')]
    assert got[1]['per_strategy']['momentum']['action'] == 'sell'


# ------------------------------------------------------------------ skill

def test_skill_is_the_hit_rate_against_chance_with_an_honest_range():
    rows = [outcome(ret=0.01, chance=0.5, day=d) for d in range(80)] + [outcome(ret=-0.01, chance=0.5, day=100 + d) for d in range(20)]
    [s] = sl.skill(rows, horizon=5, min_signals=50)
    assert (s['source'], s['market'], s['n'], s['hit_rate'], s['chance']) == ('momentum', 'us_equity', 100, 0.8, 0.5)
    assert s['lift'] == 0.3 and s['z'] == 6.0 and s['enough'] and s['low'] < 0.8 < s['high']
    assert sl.skill(rows[:10], horizon=5, min_signals=50)[0]['enough'] is False
    assert sl.skill(rows, horizon=20) == []                                  # nothing at another horizon


def test_a_strategy_that_only_looks_right_because_the_market_rose_has_no_skill():
    rows = [outcome(ret=0.01, chance=1.0, day=d) for d in range(80)]
    [s] = sl.skill(rows, 5)
    assert s['hit_rate'] == 1.0 and s['lift'] == 0.0 and s['z'] == 0.0


def test_skill_is_kept_apart_by_strategy_and_market():
    rows = [outcome(source='momentum', market='forex', ret=0.01, day=d) for d in range(5)] + \
           [outcome(source='momentum', market='us_equity', ret=-0.01, day=d) for d in range(5)] + \
           [outcome(source='rsi_strategy', market='forex', ret=0.01, day=d) for d in range(5)]
    assert [(s['source'], s['market'], s['hit_rate']) for s in sl.skill(rows, 5)] == [
        ('momentum', 'forex', 1.0), ('momentum', 'us_equity', 0.0), ('rsi_strategy', 'forex', 1.0)]


# ---------------------------------------------------------------- tilting

def test_the_tilt_guardrail_needs_evidence_and_stays_within_bounds():
    assert signal_tilt(49, 5.0) == 1.0                       # too few signals: no change
    assert signal_tilt(50, 0.0) == 1.0
    assert signal_tilt(200, 3.0) == 1.5 and signal_tilt(200, 99.0) == 1.5
    assert signal_tilt(200, -3.0) == 0.5 and signal_tilt(200, -99.0) == 0.5    # halved, never switched off
    assert 1.0 < signal_tilt(200, 1.0) < 1.5 and 0.5 < signal_tilt(200, -1.0) < 1.0
    for junk in (None, 'x', float('nan')):
        assert signal_tilt(200, junk) == 1.0 and signal_tilt(junk, 3.0) == 1.0


def test_only_strategies_are_tilted_never_the_ensemble():
    rows = [outcome(source=s, ret=0.01, chance=0.5, day=d) for s in ('momentum', 'ensemble') for d in range(80)]
    scale = sl.tilts(sl.skill(rows, 5), 50)
    assert list(scale) == [('momentum', 'us_equity')] and scale[('momentum', 'us_equity')] == 1.5


def ensemble_scores(manager, symbol):
    votes = {'momentum': {'action': 'buy', 'confidence': 0.8, 'position_size': 0.04},
             'mean_reversion': {'action': 'sell', 'confidence': 0.8, 'position_size': 0.04}}
    return manager._adaptive_confidence_ensemble(votes, symbol, {'volatility': 0.01})['action_scores']


def test_with_no_tilt_the_ensemble_is_exactly_what_it_was_and_a_tilt_moves_only_its_own_market():
    fresh = StrategyManager(CONFIG)
    before = ensemble_scores(fresh, 'AAPL')
    fresh.set_signal_skill({})
    assert ensemble_scores(fresh, 'AAPL') == before                          # off by default: no change at all
    assert fresh.signal_tilt_for('momentum', 'AAPL') == 1.0
    fresh.set_signal_skill({('momentum', 'us_equity'): 1.5, ('mean_reversion', 'us_equity'): 0.5})
    after = ensemble_scores(fresh, 'AAPL')
    assert after['buy'] > before['buy'] and after['sell'] < before['sell']
    assert ensemble_scores(fresh, 'EUR_USD') == ensemble_scores(StrategyManager(CONFIG), 'EUR_USD')   # another market
    fresh.set_signal_skill(None)
    assert ensemble_scores(fresh, 'AAPL') == before


# --------------------------------------------------------------- the job

def learning(tmp_path, enabled, decisions, closes_for, sm=None, clock=None):
    config = {**CONFIG, 'learning': {'signal_skill_tilt': {'enabled': enabled, 'min_signals': 5, 'refresh_hours': 6}}}
    path = journal_with(tmp_path, decisions)
    ledger = sl.SignalLedger(tmp_path / 'signal_outcomes.db')
    return sl.SignalLearning(config, ledger, sm, decisions_path=path, closes_for=closes_for,
                             clock=clock or (lambda: 1e9)), ledger


class FakeManager:
    def __init__(self):
        self.calls = []

    def set_signal_skill(self, tilts):
        self.calls.append(dict(tilts))


def good_decisions(vals):
    per = {'momentum': {'action': 'buy', 'confidence': 0.7}}
    return [row('AAPL', d, 'hold', per) for d in range(0, 200) if vals[d + 5] > vals[d]]


def test_the_job_fills_the_ledger_and_by_default_only_measures(tmp_path):
    vals = walk()
    sm = FakeManager()
    job, ledger = learning(tmp_path, False, good_decisions(vals), lambda s: closes(vals), sm)
    result = job.run()
    assert result['new_outcomes'] > 5 and result['applied'] is False
    assert sm.calls == [{}]                                                  # measured, but nothing tilted
    assert job.run()['new_outcomes'] == 0                                    # a second pass adds nothing


def test_switched_on_it_hands_the_strategy_manager_the_skill_scales(tmp_path):
    vals = walk()
    sm = FakeManager()
    job, _ = learning(tmp_path, True, good_decisions(vals), lambda s: closes(vals), sm)
    result = job.run()
    assert result['applied'] and sm.calls[-1][('momentum', 'us_equity')] > 1.0
    assert any('momentum/us_equity' in t for t in result['tilted'])


def test_the_job_refreshes_on_schedule_and_a_failure_never_raises(tmp_path):
    now = {'t': 1e9}
    job, _ = learning(tmp_path, False, [], lambda s: {}, clock=lambda: now['t'])
    assert job.due() and job.maybe_update()
    job._thread.join(5)
    assert not job.maybe_update()                                            # just ran
    now['t'] += 7 * 3600
    assert job.due()
    job.decisions_path = str(tmp_path / 'decision_journal.db')
    job.ledger.close()                                                       # a broken ledger
    assert job.run() == job.last_result                                      # logged, not raised


# ------------------------------------------------------------ price data

def test_a_kenyan_stock_is_read_from_the_nse_files_never_from_yahoo(tmp_path, monkeypatch):
    import src.connectors.nse_connector as nse_connector
    monkeypatch.setattr(nse_connector, 'NSE_CSV_DIR', tmp_path)
    (tmp_path / 'SCOM.csv').write_text('date,close,source\n2026-09-23,30.5,nse_pricelist\n2026-09-24,31.0,synthetic\n')

    def boom(*a, **k):
        raise AssertionError('Yahoo must not be asked for a Kenyan stock')
    monkeypatch.setattr(chart_data, 'yfinance_bars', boom)
    config = {'data_manager': {'nse_symbols': ['SCOM']}}
    got = sl.closes_provider(config)('SCOM')
    assert got == {date(2026, 9, 23): 30.5}                                  # real rows only


def test_other_symbols_come_from_yahoo_and_a_failure_is_remembered_for_the_hour(monkeypatch):
    asked = []

    def fetch(symbol, period='2y'):
        asked.append(symbol)
        if symbol == 'BAD':
            raise ConnectionError('throttled')
        return [{'time': '2026-09-23', 'close': 190.0}]
    monkeypatch.setattr(chart_data, 'yfinance_bars', fetch)
    now = {'t': 0.0}
    provider = sl.closes_provider({}, ttl=3600, clock=lambda: now['t'])
    assert provider('AAPL') == {date(2026, 9, 23): 190.0} and provider('AAPL') and asked == ['AAPL']
    assert provider('BAD') == {} and provider('BAD') == {} and asked == ['AAPL', 'BAD']
    now['t'] = 4000
    provider('BAD')
    assert asked == ['AAPL', 'BAD', 'BAD']


# ------------------------------------------------------------ the scorecard

NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)


def card(ledger, config=CONFIG, days=40):
    start = NOW - timedelta(days=days)
    inputs = {'run_started_at': start.isoformat(), 'planned_days': 90, 'decisions': [], 'fills': [], 'equity': [],
              'spy': None, 'closes_for': None, 'healing': None, 'alerter': None, 'alerts': [], 'tuner': None,
              'ledger': ledger}
    return scorecard.build(inputs, config, NOW)


def metric(c, mid):
    return next(m for s in c['sections'] for m in s['metrics'] if m['id'] == mid)


def test_the_forecast_check_reads_the_ledger_and_counts_only_the_run():
    day = lambda i: (date(2026, 8, 25) + timedelta(days=i)).isoformat()
    inside = [{**outcome(source='ensemble', ret=0.01, chance=0.5), 'day': day(20 + i % 30), 'symbol': f'S{i}'} for i in range(120)]
    before = [{**outcome(source='ensemble', ret=-0.01, chance=0.5), 'day': '2026-07-01', 'symbol': f'O{i}'} for i in range(200)]
    strategy_rows = [{**outcome(source='momentum', ret=-0.01, chance=0.5), 'day': day(25), 'symbol': f'M{i}'} for i in range(200)]
    m = metric(card(inside + before + strategy_rows), 'forecast_5d')
    assert m['status'] == 'pass' and m['value'] == 1.0 and '120' in m['display']      # not the old, wrong or per-strategy rows


def test_no_ledger_falls_back_to_the_old_check_and_says_skill_is_not_recorded_yet():
    c = card(None)
    assert metric(c, 'strategy_skill')['status'] == 'too_early' and 'not recorded yet' in metric(c, 'strategy_skill')['display']
    assert metric(c, 'forecast_5d')['status'] == 'too_early'


def test_the_skill_measure_names_the_best_and_worst_and_says_the_tilt_is_off():
    rows = [outcome(source='momentum', ret=0.01, chance=0.5, day=d) for d in range(80)] + \
           [outcome(source='rsi_strategy', ret=-0.01, chance=0.5, day=d) for d in range(80)] + \
           [outcome(source='mean_reversion', ret=0.01, chance=0.5, day=d) for d in range(10)]
    m = metric(card(rows), 'strategy_skill')
    assert m['status'] == 'info' and m['display'] == '2 of 3 pairs judged'
    assert 'Best: momentum in us_equity' in m['note'] and 'Worst: rsi_strategy' in m['note']
    assert 'Measuring only' in m['note']
    tilts = {(r['source'], r['market']): r['tilt'] for r in m['detail']}
    assert tilts[('momentum', 'us_equity')] == 1.5 and tilts[('rsi_strategy', 'us_equity')] == 0.5
    assert tilts[('mean_reversion', 'us_equity')] == 1.0                                # too few signals
    on = {**CONFIG, 'learning': {'signal_skill_tilt': {'enabled': True}}}
    assert 'Tilt is ON' in metric(card(rows, on), 'strategy_skill')['note']


def test_too_few_signals_is_too_early_and_says_the_most_so_far():
    rows = [outcome(ret=0.01, day=d) for d in range(12)]
    m = metric(card(rows), 'strategy_skill')
    assert m['status'] == 'too_early' and '12 is the most so far' in m['note']
    assert metric(card([]), 'strategy_skill')['display'] == 'no outcomes yet'


def test_the_terminal_command_no_longer_asks_yahoo_for_kenyan_stocks(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('agent_scorecard', 'scripts/agent_scorecard.py')
    cmd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd)
    assert cmd.price_history(True) == (None, None)
    asked = []
    monkeypatch.setattr(chart_data, 'yfinance_bars', lambda s, period='2y': asked.append(s) or [])
    closes_for, spy = cmd.price_history(False, {'data_manager': {'nse_symbols': ['KCB']}})
    assert asked == ['SPY'] and spy is None
    assert closes_for('KCB') == {} and asked == ['SPY']                        # the NSE's own files, not Yahoo


# --------------------------------------------------------------- the tuner

def test_a_market_with_its_own_settings_is_left_out_of_tuning_the_base_settings():
    seen = []

    def evaluator(name, cfg, params, series, holdout):
        seen.append(sorted(series))
        return {'is': (0.02, 10), 'oos': (0.01, 8)}
    config = {'strategies': {'momentum': {'market_overrides': {'_comment': 'x', 'forex': {'threshold': 0.006}}},
                             'mean_reversion': {}}}
    manager = SimpleNamespace(strategies={
        'momentum': TechnicalStrategy('momentum', {'threshold': 0.02}),
        'mean_reversion': TechnicalStrategy('mean_reversion', {'band': 0.02})}, strategy_performance={})
    tuner = st.StrategyTuner(manager, config, history_source=lambda: {}, params_path=__import__('pathlib').Path(os.devnull),
                             evaluator=evaluator)
    series = {'AAPL': ([100.0] * 200, 0.0), 'EUR_USD': ([1.1] * 200, 0.0)}
    entry = tuner.tune_one('momentum', series)
    assert entry['left_out'] == ['EUR_USD'] and entry['symbols'] == 1
    assert all(s == ['AAPL'] for s in seen)
    plain = tuner.tune_one('mean_reversion', series)                        # no override: forex stays in
    assert 'left_out' not in plain and plain['symbols'] == 2
