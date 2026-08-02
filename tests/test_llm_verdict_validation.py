"""The LLM's output is untrusted free-form JSON on the money path.

`validate_trade` returned the model's parsed dict verbatim and main.py fed it
straight into execution:

    if validated_signal['action'] != 'hold':
        all_signals[symbol] = validated_signal

So a model answering "sell" to a proposed "buy" would have executed a sell,
and a missing 'confidence' key would have raised KeyError inside the
per-symbol try/except — silently dropping the symbol for that cycle.

The reviewer's remit is to APPROVE, RESIZE or REJECT a signal the ensemble
already produced. It is not to originate a trade in the opposite direction.

Also covered: the verdict cache was keyed on
{symbol}:{action}:{confidence_bucket} with a 900s TTL and no price
component, so a verdict formed at one price was reapplied for fifteen
minutes at a materially different one.
"""

import pytest

from src.agent.llm_orchestrator import LLMOrchestrator


def make_orch(response, **cfg):
    """An orchestrator whose provider call returns `response` verbatim."""
    base = {'llm_enabled': True, 'primary_llm_provider': 'gemini'}
    base.update(cfg)
    orch = LLMOrchestrator(base)
    orch.enabled = True
    orch.gemini_api_key = 'test-key'
    orch.openrouter_api_key = None
    orch.calls = []

    def fake_gemini(system_prompt, user_prompt, fallback, model_override=None):
        orch.calls.append(user_prompt)
        return response if response is not None else fallback

    orch._call_gemini = fake_gemini
    return orch


BUY = {'action': 'buy', 'confidence': 0.7, 'position_size': 0.02, 'strategy': 'ensemble'}
SELL = {'action': 'sell', 'confidence': 0.7, 'position_size': 0.02, 'strategy': 'ensemble'}
DATA = {'close': 310.0, 'open': 308.0, 'price': 310.0}


# ------------------------------------------------------------ direction

def test_llm_may_approve_the_proposed_direction():
    orch = make_orch({'action': 'buy', 'confidence': 0.9, 'position_size': 0.03})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'buy'
    assert out['confidence'] == pytest.approx(0.9)


def test_llm_may_veto_down_to_hold():
    orch = make_orch({'action': 'hold', 'confidence': 0.1, 'reasoning': 'bad news'})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'hold'


def test_llm_cannot_flip_a_buy_into_a_sell():
    """THE guard: a reviewer reversing the ensemble would open a position
    nothing in the system actually proposed."""
    orch = make_orch({'action': 'sell', 'confidence': 0.95, 'position_size': 0.05})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'hold'


def test_llm_cannot_flip_a_sell_into_a_buy():
    orch = make_orch({'action': 'buy', 'confidence': 0.95, 'position_size': 0.05})
    out = orch.validate_trade('TSLA', dict(SELL), DATA)
    assert out['action'] == 'hold'


def test_unrecognised_action_becomes_hold():
    orch = make_orch({'action': 'YOLO', 'confidence': 0.9})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'hold'


def test_action_casing_and_whitespace_are_tolerated():
    orch = make_orch({'action': '  BUY  ', 'confidence': 0.8, 'position_size': 0.02})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'buy'


# --------------------------------------------------------------- clamping

@pytest.mark.parametrize('bad,expected', [(1.9, 1.0), (-0.4, 0.0), (float('nan'), 0.0)])
def test_confidence_is_clamped_into_range(bad, expected):
    orch = make_orch({'action': 'buy', 'confidence': bad, 'position_size': 0.02})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['confidence'] == pytest.approx(expected)


def test_position_size_is_clamped_into_range():
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 5.0})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert 0.0 <= out['position_size'] <= 1.0


def test_missing_confidence_falls_back_to_the_base_signal_value():
    """main.py does `validated_signal['confidence'] *= 0.5` unguarded."""
    orch = make_orch({'action': 'buy', 'position_size': 0.02})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['confidence'] == pytest.approx(BUY['confidence'])


def test_missing_position_size_falls_back_to_the_base_signal_value():
    orch = make_orch({'action': 'buy', 'confidence': 0.8})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['position_size'] == pytest.approx(BUY['position_size'])


def test_non_numeric_confidence_falls_back():
    orch = make_orch({'action': 'buy', 'confidence': 'high', 'position_size': 0.02})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['confidence'] == pytest.approx(BUY['confidence'])


def test_validated_signal_always_carries_the_execution_keys():
    """Downstream reads action/confidence/position_size unguarded."""
    orch = make_orch({'action': 'buy'})
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert {'action', 'confidence', 'position_size'} <= set(out)


def test_non_dict_response_falls_back_to_the_base_signal():
    orch = make_orch(['not', 'a', 'dict'])
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == BUY['action']
    assert out['confidence'] == pytest.approx(BUY['confidence'])


def test_provider_outage_still_returns_the_base_signal_untouched():
    """An LLM outage must not force-hold a trade — existing behaviour."""
    orch = make_orch(None)
    out = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert out['action'] == 'buy'


def test_symbol_is_preserved_on_the_verdict():
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    signal = dict(BUY, symbol='TSLA')
    out = orch.validate_trade('TSLA', signal, DATA)
    assert out.get('symbol') == 'TSLA'


# ------------------------------------------------------------ cache & price

def test_identical_request_is_served_from_cache():
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    orch.validate_trade('TSLA', dict(BUY), DATA)
    orch.validate_trade('TSLA', dict(BUY), DATA)
    assert len(orch.calls) == 1


def test_a_materially_different_price_is_not_served_from_cache():
    """900s TTL with no price in the key reapplied a stale verdict across a
    move large enough to invalidate the reasoning behind it."""
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    orch.validate_trade('TSLA', dict(BUY), {'close': 310.0, 'price': 310.0})
    orch.validate_trade('TSLA', dict(BUY), {'close': 340.0, 'price': 340.0})
    assert len(orch.calls) == 2


def test_a_trivial_price_wiggle_still_hits_the_cache():
    """The point is to avoid staleness, not to defeat caching entirely."""
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    orch.validate_trade('TSLA', dict(BUY), {'close': 310.00, 'price': 310.00})
    orch.validate_trade('TSLA', dict(BUY), {'close': 310.05, 'price': 310.05})
    assert len(orch.calls) == 1


def test_a_flipped_verdict_is_not_cached_as_a_real_answer():
    """A rejected flip must not poison the cache for the whole TTL."""
    orch = make_orch({'action': 'sell', 'confidence': 0.9, 'position_size': 0.05})
    first = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert first['action'] == 'hold'
    orch._call_gemini = lambda s, u, f, model_override=None: (
        orch.calls.append(u) or {'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    second = orch.validate_trade('TSLA', dict(BUY), DATA)
    assert second['action'] == 'buy'


def test_missing_price_does_not_break_caching():
    orch = make_orch({'action': 'buy', 'confidence': 0.8, 'position_size': 0.02})
    orch.validate_trade('TSLA', dict(BUY), {})
    orch.validate_trade('TSLA', dict(BUY), {})
    assert len(orch.calls) == 1
