"""One position sizing policy, used by both the live loop and the backtest.

The config carried three settings that appeared to answer the same question
and disagreed: cash_policy.base_risk_per_trade (2%), trading.risk_per_trade
(0.5%) and risk_limits.max_position_size (5%).

The disagreement was a symptom. The live trading loop sized positions as a
flat fraction of equity capped by the cash policy, while the backtest sized
off the stop distance. Two sizers, two answers, and the live one ignored
volatility even though the stops immediately below it did not. Backtest
results therefore did not describe the agent that would actually run.

These tests pin the single policy:

    notional = equity * risk_per_trade * cash_policy_multiplier / stop distance
    notional = min(notional, equity * max_position_size)
"""
import ast
import json
import os

import pytest

from src.agent.cash_policy import CashDeploymentPolicy
from src.agent.position_sizing import risk_contribution, volatility_scaled_value

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY = 100_000.0
STOP_MULT = 2.5
MAX_POS = 0.05
RISK = 0.005


def _size(atr_pct, confidence=1.0, risk=RISK):
    return volatility_scaled_value(EQUITY, atr_pct, confidence=confidence,
                                   risk_per_trade=risk, stop_atr_mult=STOP_MULT,
                                   max_position_pct=MAX_POS)


# ------------------------------------------------- what the policy guarantees

def test_risk_is_equalised_across_volatilities():
    """The whole point: the same signal risks the same amount either way.

    Before this change the live loop put an identical notional into a 1% ATR
    bank and a 6% ATR crypto pair, which is six times the risk for the same
    conviction.
    """
    # Strictly above the crossover where the 5% cap stops binding:
    # atr = risk / (stop_mult * cap) = 0.005 / (2.5 * 0.05) = 4%.
    risks = []
    for atr in (0.05, 0.06, 0.08, 0.10):
        notional = _size(atr)
        risks.append(risk_contribution(notional, EQUITY, atr, STOP_MULT))
    assert max(risks) - min(risks) < 1e-9
    assert risks[0] == pytest.approx(RISK)


def test_volatile_instruments_get_smaller_positions():
    # Above the 4% crossover, size falls as volatility rises. Below it every
    # position is the capped size, which is the cap doing its job.
    assert _size(0.10) < _size(0.06) < _size(0.05)
    assert _size(0.05) < _size(0.02) == _size(0.01)


def test_the_position_cap_binds_for_quiet_instruments():
    """Below roughly 4% ATR the 5% cap decides, by design."""
    assert _size(0.01) == pytest.approx(EQUITY * MAX_POS)
    assert _size(0.02) == pytest.approx(EQUITY * MAX_POS)
    # And nothing ever exceeds it, however quiet or however boosted.
    for atr in (0.001, 0.01, 0.02, 0.06):
        assert _size(atr, risk=RISK * 2.5) <= EQUITY * MAX_POS + 1e-9


def test_confidence_scales_the_position():
    assert _size(0.06, confidence=0.5) == pytest.approx(_size(0.06, confidence=1.0) / 2)
    assert _size(0.06, confidence=0.0) == 0.0


def test_no_atr_falls_back_to_the_cap_rather_than_refusing_to_trade():
    """A cold start must still trade, at the capped size."""
    assert _size(None, confidence=1.0) == pytest.approx(EQUITY * MAX_POS)


# --------------------------------------------- the cash policy composes with it

def test_the_cash_policy_multiplies_risk_and_the_cap_still_holds():
    cfg = {'cash_policy': {'enabled': True, 'target_deployment': 0.80,
                           'min_confidence_to_boost': 0.55,
                           'max_risk_multiplier': 2.5}}
    policy = CashDeploymentPolicy(cfg)

    flush = policy.risk_multiplier(deployed_pct=0.0, confidence=1.0)
    assert flush > 1.0

    atr = 0.06                                   # volatile enough that the cap is slack
    boosted = _size(atr, risk=RISK * flush)
    assert boosted > _size(atr)
    assert boosted <= EQUITY * MAX_POS

    # A weak signal is never boosted, however much cash is idle.
    assert policy.risk_multiplier(deployed_pct=0.0, confidence=0.3) == 1.0


# --------------------------------------------------- the config is coherent

def _config():
    with open(os.path.join(ROOT, 'config', 'config.json'), encoding='utf-8') as f:
        return json.load(f)


def test_config_has_exactly_one_risk_authority():
    cfg = _config()
    assert cfg['trading']['risk_per_trade'] > 0
    cash = cfg['cash_policy']
    for legacy in ('base_risk_per_trade', 'max_risk_per_trade'):
        assert legacy not in cash, (
            f"cash_policy.{legacy} is a notional cap that competes with "
            f"trading.risk_per_trade and risk_limits.max_position_size. "
            f"Use cash_policy.max_risk_multiplier instead.")
    assert cash['max_risk_multiplier'] >= 1.0


def test_configured_risk_and_cap_leave_the_cap_as_a_ceiling_not_the_decider():
    """A risk budget so large that the cap always binds would silently turn
    volatility scaling off, which is how the old 2% setting behaved."""
    cfg = _config()
    risk = cfg['trading']['risk_per_trade']
    cap = cfg['risk_limits']['max_position_size']
    mult = cfg['cash_policy']['max_risk_multiplier']
    stop = cfg['risk_limits']['stop_loss_atr_mult']
    # ATR at which the cap starts binding, at maximum boost.
    crossover = (risk * mult) / (cap * stop)
    assert crossover < 0.20, (
        f"At {risk:.3%} risk boosted {mult}x with {stop}x ATR stops, the "
        f"{cap:.0%} cap only stops binding above {crossover:.1%} ATR, so "
        f"nearly every position would be the capped size.")


# ------------------------------- the live loop uses this policy, not another

def _live_sizing_source():
    with open(os.path.join(ROOT, 'src', 'agent', 'main.py'), encoding='utf-8') as f:
        return f.read()


def test_the_live_loop_sizes_with_the_shared_function():
    src = _live_sizing_source()
    assert 'volatility_scaled_value(' in src, (
        "The live loop must size with the same function the backtest uses, "
        "or backtest results do not describe the running agent.")


def test_the_live_loop_no_longer_uses_the_old_notional_cap():
    src = _live_sizing_source()
    assert 'risk_cap(' not in src, (
        "risk_cap() was the competing notional sizer that ignored volatility.")


def test_the_live_sizing_call_is_syntactically_reachable():
    """Guards against the call being left inside dead or unparsed code."""
    tree = ast.parse(_live_sizing_source())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'volatility_scaled_value']
    assert calls, 'no call to volatility_scaled_value found in main.py'
