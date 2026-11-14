"""One guarded process changes strategy settings: the weekly tuner.

Before: the nightly growth cycle wrote strategy weights nothing read, after a
single trade, compounding on its own output; the practice session and the
tuner wrote the same file in two layouts, wiping each other out; and the
tuner adopted settings proven on five trades at a flat bar, with no check
that a change held up afterwards.
"""
import json
from types import SimpleNamespace

import pytest

from src.agent import strategy_tuner as st
from src.agent.technical_strategy import TechnicalStrategy

DAY = 86400.0
T0 = 1_800_000_000.0          # Friday 15 January 2027, 08:00 UTC


def _ev(is_ret, is_n, oos_ret, oos_n):
    return {'is': (is_ret, is_n), 'oos': (oos_ret, oos_n)}


def _manager():
    return SimpleNamespace(strategies={
        'momentum': TechnicalStrategy('momentum', {'threshold': 0.02, 'lookback_period': 50}),
        'rsi_strategy': TechnicalStrategy('rsi_strategy', {'rsi_oversold': 30, 'rsi_overbought': 70})},
        strategy_performance={})


class _Table:
    """Scores settings from a table that a test can change between runs."""

    def __init__(self, table, default=_ev(0.02, 40, 0.01, 35)):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, name, cfg, params, series, holdout):
        self.calls.append((name, dict(params), holdout))
        return self.table.get((name, params.get('threshold')), self.default)


def _tuner(tmp_path, evaluator, clock, manager=None):
    return st.StrategyTuner(manager or _manager(), {},
                            history_source=lambda: {'AAPL': ([100.0] * 520, 0.001)},
                            params_path=tmp_path / 'strategy_params.json',
                            evaluator=evaluator, clock=lambda: clock['t'])


# ------------------------------------------------------------- the hurdle

def test_the_hurdle_grows_with_the_number_of_candidates():
    assert st.hurdle(0.005, 1) == 0.005
    assert st.hurdle(0.005, 2) == pytest.approx(0.005 * 1.1774, rel=1e-3)
    assert st.hurdle(0.005, 4) > st.hurdle(0.005, 2)


def test_five_trades_no_longer_earn_an_adoption():
    base = _ev(0.02, 40, 0.01, 35)
    lucky = ({'threshold': 0.015}, _ev(0.03, 5, 0.05, 5))
    assert st.choose(base, [lucky], st.DEFAULTS) is None


def test_an_improvement_inside_the_scaled_hurdle_is_not_enough():
    base = _ev(0.02, 40, 0.01, 35)
    a = ({'threshold': 0.015}, _ev(0.03, 40, 0.0155, 35))    # +0.55%: clears 0.5% but not 0.59%
    b = ({'threshold': 0.025}, _ev(0.03, 40, 0.0100, 35))
    assert st.choose(base, [a], st.DEFAULTS) == a             # one candidate: the plain bar
    assert st.choose(base, [a, b], st.DEFAULTS) is None       # two tried: the bar rises


# ------------------------------------------------------------ probation

def _adopt(tmp_path, table, clock, manager=None):
    tuner = _tuner(tmp_path, table, clock, manager)
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'] == 'adopted' and entry['after'] == {'threshold': 0.025}
    return tuner


def test_a_new_setting_is_left_alone_until_its_probation_is_due(tmp_path):
    table = _Table({('momentum', 0.025): _ev(0.04, 40, 0.03, 35)})
    clock = {'t': T0}
    tuner = _adopt(tmp_path, table, clock)
    clock['t'] += 10 * DAY
    table.calls.clear()
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'].startswith('on probation: adopted 10 days ago')
    assert table.calls == []                      # not re-tuned, nothing stacked on it


def test_a_setting_that_holds_up_after_adoption_is_confirmed(tmp_path):
    table = _Table({('momentum', 0.025): _ev(0.04, 40, 0.03, 35)})
    clock = {'t': T0}
    manager = _manager()
    tuner = _adopt(tmp_path, table, clock, manager)
    clock['t'] += 30 * DAY
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'].startswith('confirmed')
    assert manager.strategies['momentum'].momentum_threshold == 0.025
    # Judged on the bars since adoption only: 30 days hold 21 weekdays.
    holdouts = {round(h, 4) for _n, _p, h in table.calls[-2:]}
    assert holdouts == {round(int(21 * 0.95) / 520, 4)}


def test_a_setting_that_does_worse_after_adoption_is_reverted(tmp_path):
    table = _Table({('momentum', 0.025): _ev(0.04, 40, 0.03, 35)})
    clock = {'t': T0}
    manager = _manager()
    tuner = _adopt(tmp_path, table, clock, manager)
    table.table[('momentum', 0.025)] = _ev(0.04, 40, -0.02, 30)   # since adoption: worse
    clock['t'] += 30 * DAY
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'].startswith('reverted')
    assert manager.strategies['momentum'].momentum_threshold == 0.02
    saved = json.loads((tmp_path / 'strategy_params.json').read_text())
    assert saved['params']['momentum'] == {'threshold': 0.02} and not saved['probation']


def test_with_too_few_trades_to_judge_it_waits_then_confirms_by_default(tmp_path):
    table = _Table({('momentum', 0.025): _ev(0.04, 40, 0.03, 35)})
    clock = {'t': T0}
    tuner = _adopt(tmp_path, table, clock)
    table.default = _ev(0.02, 40, 0.01, 3)
    table.table[('momentum', 0.025)] = _ev(0.04, 40, 0.0, 3)
    clock['t'] += 40 * DAY
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'] == 'on probation: 3 trades since adoption, 10 needed to judge'
    clock['t'] += 60 * DAY
    (entry,) = tuner.run(['momentum'])
    assert entry['outcome'].startswith('confirmed: too few trades to judge')


def test_weekdays_are_counted_from_the_day_after():
    from datetime import datetime, timezone
    ts = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    assert st._weekdays_between(ts(2026, 11, 13), ts(2026, 11, 16)) == 1      # Fri -> Mon
    assert st._weekdays_between(ts(2026, 11, 2), ts(2026, 11, 30)) == 20      # four weeks
    assert st._weekdays_between(ts(2026, 11, 16), ts(2026, 11, 16)) == 0


# ------------------------------------------------- one settings file, one owner

def test_an_old_practice_file_is_read_and_only_owned_settings_kept(tmp_path):
    (tmp_path / 'strategy_params.json').write_text(json.dumps(
        {'momentum': {'threshold': 0.03, 'rsi_oversold': 35, 'lookback_period': 30},
         'rsi_strategy': {'rsi_oversold': 25, 'lookback_period': 30}}))
    manager = _manager()
    tuner = _tuner(tmp_path, _Table({}), {'t': T0}, manager)
    assert tuner.state['params'] == {'momentum': {'threshold': 0.03}, 'rsi_strategy': {'rsi_oversold': 25}}
    assert manager.strategies['momentum'].momentum_threshold == 0.03
    assert manager.strategies['momentum'].lookback_period == 50
    assert manager.strategies['rsi_strategy'].rsi_oversold == 25


def test_the_practice_session_never_writes_the_settings_file(tmp_path, monkeypatch):
    import scripts.practice_session as ps
    wave = [100 * (1 + 0.001 * i) * (1 + 0.05 * ((i % 20) - 10) / 10) for i in range(400)]
    monkeypatch.setattr(ps, 'fetch_bars', lambda symbols: {'AAPL': wave, 'MSFT': wave[::-1]})
    monkeypatch.setattr(ps, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(ps, 'LOG_FILE', str(tmp_path / 'practice_log.jsonl'))
    monkeypatch.setattr('sys.argv', ['practice_session.py'])
    ps.main()
    assert not (tmp_path / 'strategy_params.json').exists()
    records = [json.loads(line) for line in (tmp_path / 'practice_log.jsonl').read_text().splitlines()]
    assert records and all((r['promoted'] or {}).get('applied', False) is False for r in records)


def test_the_growth_cycle_suggests_weights_but_writes_none(tmp_path):
    from scripts.run_nightly_growth_cycle import run_growth_cycle
    with open('config/config.json') as f:
        config = json.load(f)
    report = run_growth_cycle(config, data_dir=str(tmp_path))
    assert not (tmp_path / 'config_overlay.json').exists()
    assert report['strategies']['momentum'] == config['strategies']['momentum']['weight']
    # No closed trades yet: the evidence tilt suggests no change at all.
    assert report['suggested_weights'] == report['strategies']
    assert 'nothing is changed' in report['weights_note']
