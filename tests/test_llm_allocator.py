"""The AI strategy allocator (llm_allocator.py): off by default, and when on,
held to a relative scale (1.0 neutral), a 0.5 to 1.5 range, a 0.25 step, and
asked again only when new trades have closed. It used to be handed the
manager's weights, which sum to 1 (about 0.17 each), and told 1.0 was
neutral, so a proposed cut was a raise and a proposed 0 switched a
strategy off."""
import json

import pytest

from src.agent.llm_allocator import LLMAllocator, WEIGHT_MAX, WEIGHT_MIN, MAX_STEP

CURRENT = {'momentum': 1.0, 'mean_reversion': 1.0, 'rsi_strategy': 0.9}   # relative to neutral


def test_a_sane_proposal_applies_within_the_step():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 1.2, 'mean_reversion': 0.8, 'rsi_strategy': 0.9}, CURRENT)
    assert out == {'momentum': 1.2, 'mean_reversion': 0.8, 'rsi_strategy': 0.9}


def test_extreme_values_are_held_to_the_range_and_the_step():
    out = LLMAllocator.apply_guardrails({'momentum': 99.0, 'mean_reversion': -5.0}, CURRENT)
    assert out['momentum'] == pytest.approx(1.0 + MAX_STEP)               # 99 -> 1.5 -> one step up
    assert out['mean_reversion'] == pytest.approx(1.0 - MAX_STEP)         # -5 -> 0.5 -> one step down
    far = {'a': 1.5, 'b': 0.5}
    out = LLMAllocator.apply_guardrails({'a': 3.0, 'b': 0.0}, far)
    assert out == {'a': WEIGHT_MAX, 'b': WEIGHT_MIN}                      # never beyond the range


def test_unknown_strategies_and_garbage_are_dropped():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 1.1, 'made_up_strategy': 1.5, 'strategy': 'llm_orchestrated',
         'mean_reversion': 'high', 'rsi_strategy': None}, CURRENT)
    assert out == {'momentum': 1.1}


def test_unusable_proposals_return_none():
    for bad in (None, {}, 'not a dict', {'unknown': 1.0}, {'momentum': float('nan')}):
        assert LLMAllocator.apply_guardrails(bad, CURRENT) is None


def test_a_strategy_with_too_few_trades_keeps_its_weight():
    attribution = {'momentum': {'closed_trades': 9}, 'mean_reversion': {'closed_trades': 2}}
    out = LLMAllocator.apply_guardrails({'momentum': 1.2, 'mean_reversion': 0.6}, CURRENT, attribution, 5)
    assert out == {'momentum': 1.2}


def test_it_is_off_by_default_and_a_noop_without_an_ai():
    class Mgr:
        strategy_weights = {'momentum': 0.5, 'mean_reversion': 0.5}

    class Off:
        enabled = False

    class Ai:
        enabled = True

        def propose_json(self, *a, **k):
            raise AssertionError('must not be asked')
    assert LLMAllocator(Ai(), Mgr(), None, {}).maybe_rebalance() is None          # not enabled
    assert LLMAllocator(Off(), Mgr(), None, {'enabled': True}).maybe_rebalance() is None


class _Mgr:
    """The ensemble's own scale: normalised to sum 1."""

    def __init__(self):
        self.strategy_weights = {'momentum': 1 / 3, 'mean_reversion': 1 / 3, 'rsi_strategy': 1 / 3}


class _Ai:
    enabled = True

    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def propose_json(self, system, user, model_override=None):
        self.asked.append(user)
        return self.reply


def _evidence(n_each):
    return {n: {'closed_trades': n_each, 'win_rate': 0.6, 'trade_returns': [0.01] * n_each}
            for n in ('momentum', 'mean_reversion', 'rsi_strategy')}


def _allocator(ai, mgr, evidence, **cfg):
    alloc = LLMAllocator(ai, mgr, None, {'enabled': True, 'interval_hours': 0, **cfg})
    alloc._attribution = lambda: evidence()
    return alloc


def test_a_cut_is_a_cut_on_the_ensembles_own_scale():
    """A proposal of 0.75 for momentum is a cut. On the old scale it was
    a jump from 0.33 to 0.75."""
    ai, mgr = _Ai({'momentum': 0.75, 'mean_reversion': 1.25, 'rsi_strategy': 1.0}), _Mgr()
    applied = _allocator(ai, mgr, lambda: _evidence(5)).maybe_rebalance()
    w = mgr.strategy_weights
    assert w['momentum'] < w['rsi_strategy'] < w['mean_reversion']
    assert sum(w.values()) == pytest.approx(1.0)                          # the overall level is kept
    assert w['momentum'] / w['mean_reversion'] == pytest.approx(0.75 / 1.25, rel=0.02)
    assert 'relative weights' in ai.asked[0] or '1.0' in ai.asked[0]
    assert json.loads(ai.asked[0].split('\n')[0].split(': ', 1)[1]) == {
        'momentum': 1.0, 'mean_reversion': 1.0, 'rsi_strategy': 1.0}      # what the AI is shown
    assert applied and set(applied) <= set(w)


def test_a_proposal_of_zero_cannot_switch_a_strategy_off():
    ai, mgr = _Ai({'momentum': 0.0, 'mean_reversion': 1.0, 'rsi_strategy': 1.0}), _Mgr()
    _allocator(ai, mgr, lambda: _evidence(5)).maybe_rebalance()
    assert mgr.strategy_weights['momentum'] / mgr.strategy_weights['mean_reversion'] == pytest.approx(0.75 / 1.0, rel=0.05)


def test_it_waits_for_closed_trades_and_then_for_new_ones():
    trades = {'n': 3}
    ai, mgr = _Ai({'momentum': 1.2}), _Mgr()
    alloc = _allocator(ai, mgr, lambda: _evidence(trades['n']))
    assert alloc.maybe_rebalance() is None and ai.asked == []             # 9 closed: under 10
    trades['n'] = 5
    assert alloc.maybe_rebalance() is not None and len(ai.asked) == 1     # 15 closed, each with 5
    assert alloc.maybe_rebalance() is None and len(ai.asked) == 1         # nothing new: not asked again
    trades['n'] = 6
    assert alloc.maybe_rebalance() is None and len(ai.asked) == 1         # 3 new: under 5
    trades['n'] = 7
    assert alloc.maybe_rebalance() is not None and len(ai.asked) == 2     # 6 new
