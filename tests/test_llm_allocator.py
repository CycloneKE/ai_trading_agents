"""Tests for the LLM allocator guardrails — the deterministic gate that
keeps LLM proposals inside safe bounds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.llm_allocator import LLMAllocator

CURRENT = {'momentum': 1.0, 'mean_reversion': 1.0, 'rsi_strategy': 0.8}


def test_sane_proposal_applies_with_drift_cap():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 1.4, 'mean_reversion': 0.6, 'rsi_strategy': 0.8}, CURRENT)
    assert out == {'momentum': 1.4, 'mean_reversion': 0.6, 'rsi_strategy': 0.8}


def test_extreme_values_clamped_to_bounds_and_step():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 99.0, 'mean_reversion': -5.0}, CURRENT)
    # 99 -> clamp 2.0 -> drift cap 1.0+0.5 = 1.5
    assert out['momentum'] == 1.5
    # -5 -> clamp 0.0 -> drift cap 1.0-0.5 = 0.5
    assert out['mean_reversion'] == 0.5


def test_unknown_strategies_dropped():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 1.2, 'made_up_strategy': 2.0, 'strategy': 'llm_orchestrated'},
        CURRENT)
    assert 'made_up_strategy' not in out
    assert 'strategy' not in out
    assert out == {'momentum': 1.2}


def test_garbage_values_skipped_not_fatal():
    out = LLMAllocator.apply_guardrails(
        {'momentum': 'high', 'mean_reversion': None, 'rsi_strategy': 1.1}, CURRENT)
    assert out == {'rsi_strategy': 1.1}


def test_unusable_proposals_return_none():
    assert LLMAllocator.apply_guardrails(None, CURRENT) is None
    assert LLMAllocator.apply_guardrails({}, CURRENT) is None
    assert LLMAllocator.apply_guardrails('not a dict', CURRENT) is None
    assert LLMAllocator.apply_guardrails({'unknown': 1.0}, CURRENT) is None


def test_cannot_zero_out_entire_ensemble():
    # All current weights small enough that a 0-proposal reaches 0 via drift cap
    tiny = {'a': 0.3, 'b': 0.4}
    out = LLMAllocator.apply_guardrails({'a': 0.0, 'b': 0.0}, tiny)
    assert out is None  # merged sum would be 0 -> rejected, keep current


def test_allocator_noop_without_llm():
    class DisabledLLM:
        enabled = False

    class MgrStub:
        strategy_weights = dict(CURRENT)

    alloc = LLMAllocator(DisabledLLM(), MgrStub(), None)
    assert alloc.maybe_rebalance() is None
    assert MgrStub.strategy_weights == CURRENT  # untouched


def test_allocator_applies_proposal_from_llm():
    class FakeLLM:
        enabled = True
        def propose_json(self, system_prompt, user_prompt):
            return {'momentum': 1.3, 'mean_reversion': 0.7, 'rsi_strategy': 0.8}

    class MgrStub:
        def __init__(self):
            self.strategy_weights = dict(CURRENT)

    mgr = MgrStub()
    alloc = LLMAllocator(FakeLLM(), mgr, None, {'interval_hours': 0})
    applied = alloc.maybe_rebalance()
    assert applied == {'momentum': 1.3, 'mean_reversion': 0.7, 'rsi_strategy': 0.8}
    assert mgr.strategy_weights['momentum'] == 1.3
    # second call inside the interval is rate-limited... interval 0 -> allowed;
    # use a real interval to check the limiter
    alloc2 = LLMAllocator(FakeLLM(), mgr, None, {'interval_hours': 6})
    assert alloc2.maybe_rebalance() is not None
    assert alloc2.maybe_rebalance() is None  # rate-limited
