"""Guarded re-tuning of strategy settings (src/agent/strategy_tuner.py)."""
import math
import threading
from types import SimpleNamespace

import pytest

from src.agent import strategy_tuner as st
from src.agent.technical_strategy import TechnicalStrategy

DAY = 86400.0


def _manager(perf=None):
    strategies = {'momentum': TechnicalStrategy('momentum', {'threshold': 0.02, 'lookback_period': 50}),
                  'mean_reversion': TechnicalStrategy('mean_reversion', {'band': 0.02, 'lookback_period': 20}),
                  'rsi_strategy': TechnicalStrategy('rsi_strategy', {'rsi_oversold': 30, 'rsi_overbought': 70})}
    return SimpleNamespace(strategies=strategies, strategy_performance=perf or {})


def _ev(is_ret, is_n, oos_ret, oos_n):
    return {'is': (is_ret, is_n), 'oos': (oos_ret, oos_n)}


class _Scripted:
    """An evaluator that scores settings from a table, recording its calls."""

    def __init__(self, table, default=_ev(0.02, 10, 0.01, 8)):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, name, cfg, params, series, holdout):
        self.calls.append((name, dict(params)))
        return self.table.get((name, tuple(sorted(params.items()))), self.default)


SERIES = {'AAPL': ([100.0] * 200, 0.001)}


def _tuner(tmp_path, evaluator, manager=None, clock=lambda: 10 * DAY, config=None):
    return st.StrategyTuner(manager or _manager(), config or {},
                            history_source=lambda: SERIES,
                            params_path=tmp_path / 'strategy_params.json',
                            evaluator=evaluator, clock=clock)


# ------------------------------------------------------------ the choice

def test_settings_move_one_step_within_bounds():
    assert st.neighbours('momentum', {'threshold': 0.02}) == [{'threshold': 0.015}, {'threshold': 0.025}]
    assert st.neighbours('momentum', {'threshold': 0.01}) == [{'threshold': 0.015}]
    rsi = st.neighbours('rsi_strategy', {'rsi_oversold': 20, 'rsi_overbought': 70})
    assert {'rsi_oversold': 25, 'rsi_overbought': 70} in rsi and len(rsi) == 3


RULE = {**st.DEFAULTS}


def test_a_better_held_out_result_that_holds_in_sample_is_adopted():
    base = _ev(0.02, 10, 0.01, 8)
    better = ({'threshold': 0.015}, _ev(0.03, 12, 0.03, 9))
    assert st.choose(base, [better], RULE) == better


@pytest.mark.parametrize('candidate, why', [
    (_ev(0.03, 12, 0.013, 9), 'improvement under the margin'),
    (_ev(0.01, 12, 0.05, 9), 'worse in-sample: fitted to the held-out stretch'),
    (_ev(0.03, 12, 0.05, 3), 'too few held-out trades to mean anything'),
    (_ev(0.03, 2, 0.05, 9), 'too few in-sample trades'),
])
def test_a_candidate_that_fails_any_guard_is_not_adopted(candidate, why):
    assert st.choose(_ev(0.02, 10, 0.01, 8), [({'threshold': 0.015}, candidate)], RULE) is None, why


def test_the_best_qualifying_candidate_wins():
    base = _ev(0.02, 10, 0.01, 8)
    a = ({'threshold': 0.015}, _ev(0.03, 12, 0.03, 9))
    b = ({'threshold': 0.025}, _ev(0.03, 12, 0.04, 9))
    assert st.choose(base, [a, b], RULE) == b


# ----------------------------------------------------------- a tuning run

def test_an_adopted_setting_is_applied_saved_and_restored(tmp_path):
    ev = _Scripted({('momentum', (('threshold', 0.025),)): _ev(0.04, 12, 0.03, 9)})
    manager = _manager()
    tuner = _tuner(tmp_path, ev, manager)
    report = {r['strategy']: r for r in tuner.run(['momentum'])}
    assert report['momentum']['outcome'] == 'adopted'
    assert manager.strategies['momentum'].momentum_threshold == 0.025
    assert manager.strategies['momentum'].config['threshold'] == 0.025
    # A restart applies the saved setting to fresh strategy objects.
    fresh = _manager()
    _tuner(tmp_path, ev, fresh)
    assert fresh.strategies['momentum'].momentum_threshold == 0.025


def test_nothing_changes_without_evidence(tmp_path):
    manager = _manager()
    report = _tuner(tmp_path, _Scripted({}), manager).run(['momentum', 'rsi_strategy'])
    assert all(r['outcome'].startswith('kept') for r in report)
    assert manager.strategies['momentum'].momentum_threshold == 0.02
    assert manager.strategies['rsi_strategy'].rsi_oversold == 30


def test_too_little_history_tunes_nothing(tmp_path):
    tuner = _tuner(tmp_path, _Scripted({}))
    tuner.history_source = lambda: {'SCOM': ([36.0] * 60, 0.04)}   # NSE today: 60 real bars
    (entry,) = tuner.run(['momentum'])
    assert 'no symbol has 120 daily bars' in entry['outcome']


def test_a_saved_setting_outside_the_bounds_is_clamped(tmp_path):
    (tmp_path / 'strategy_params.json').write_text(
        '{"params": {"momentum": {"threshold": 0.5}}, "last_run": 0, "log": []}')
    manager = _manager()
    _tuner(tmp_path, _Scripted({}), manager)
    assert manager.strategies['momentum'].momentum_threshold == 0.06


# ------------------------------------------------------------------ when

def test_every_strategy_is_due_on_the_schedule(tmp_path):
    clock = {'t': 10 * DAY}
    tuner = _tuner(tmp_path, _Scripted({}), clock=lambda: clock['t'])
    assert set(tuner.due()) == {'momentum', 'mean_reversion', 'rsi_strategy'}
    tuner.run(tuner.due())
    clock['t'] += 6 * DAY
    assert tuner.due() == []
    clock['t'] += 1 * DAY
    assert len(tuner.due()) == 3


def test_a_losing_strategy_is_retuned_early_but_at_most_daily(tmp_path):
    perf = {'momentum': {'closed_trades': 12, 'win_rate': 0.3, 'returns': [-0.02] * 12},
            'mean_reversion': {'closed_trades': 12, 'win_rate': 0.6, 'returns': [0.01] * 12},
            'rsi_strategy': {'closed_trades': 4, 'win_rate': 0.0, 'returns': [-0.05] * 4}}
    clock = {'t': 10 * DAY}
    tuner = _tuner(tmp_path, _Scripted({}), _manager(perf), clock=lambda: clock['t'])
    tuner.run(tuner.due())            # the scheduled run
    clock['t'] += 2 * DAY
    assert tuner.due() == ['momentum']   # rsi has too few trades to judge
    tuner.run(['momentum'])
    clock['t'] += 0.5 * DAY
    assert tuner.due() == []
    clock['t'] += 0.6 * DAY
    assert tuner.due() == ['momentum']


def test_tuning_can_be_switched_off(tmp_path):
    assert _tuner(tmp_path, _Scripted({}), config={'strategy_tuning': {'enabled': False}}).due() == []


def test_a_run_happens_in_the_background(tmp_path):
    gate = threading.Event()

    def slow(*args):
        gate.wait(5)
        return _ev(0.0, 0, 0.0, 0)

    tuner = _tuner(tmp_path, slow)
    assert tuner.maybe_tune() is True       # returned at once
    assert tuner.maybe_tune() is False      # one run at a time
    gate.set()
    tuner._thread.join(5)
    assert not tuner._thread.is_alive()


# ------------------------------------------------- the simulation itself

def _wave(n=300):
    return [100 * (1 + 0.0015 * i) * (1 + 0.08 * math.sin(i / 7)) for i in range(n)]


def test_the_simulation_trades_long_only_and_charges_costs():
    closes = _wave()
    free = st.simulate('mean_reversion', {'lookback_period': 20}, {'band': 0.02}, closes, 0.0)
    costly = st.simulate('mean_reversion', {'lookback_period': 20}, {'band': 0.02}, closes, 0.04)
    assert len(free) >= 3 and len(free) == len(costly)
    assert [round(a[1] - b[1], 9) for a, b in zip(free, costly)] == [0.04] * len(free)


def test_evaluation_splits_trades_into_in_sample_and_held_out():
    series = {'A': (_wave(), 0.001), 'B': (_wave()[::-1], 0.001)}
    ev = st.evaluate('mean_reversion', {'lookback_period': 20}, {'band': 0.02}, series, 0.3)
    everything = st.evaluate('mean_reversion', {'lookback_period': 20}, {'band': 0.02}, series, 0.0)
    assert ev['is'][1] > 0 and ev['oos'][1] > 0
    assert ev['is'][1] + ev['oos'][1] == everything['oos'][1] + everything['is'][1]


def test_a_real_run_on_real_strategies_completes(tmp_path):
    manager = _manager()
    tuner = st.StrategyTuner(manager, {}, history_source=lambda: {'A': (_wave(), 0.001)},
                             params_path=tmp_path / 'p.json', clock=lambda: 10 * DAY)
    report = tuner.run(list(tuner.tunable()))
    assert {r['strategy'] for r in report} == {'momentum', 'mean_reversion', 'rsi_strategy'}
    for r in report:
        assert r['outcome'] == 'adopted' or r['outcome'].startswith('kept')
        lo_hi = [(lo, hi) for _a, lo, hi, _s in st.TUNABLE[st.family(r['strategy'])].values()]
        assert lo_hi  # every tuned value stays within its bounds
    m = manager.strategies['momentum'].momentum_threshold
    assert 0.01 <= m <= 0.06 and abs(m - 0.02) <= 0.005 + 1e-9
