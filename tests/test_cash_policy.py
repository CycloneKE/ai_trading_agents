"""The cash policy stretches the risk budget; it does not size positions.

It used to return a notional cap (2% to 5% of equity) that the trading loop
applied as the position size outright, which made it a second position sizer
competing with src/agent/position_sizing.py. These tests pin the new
contract: a multiplier on risk, never below 1.0, composing with
volatility-scaled sizing rather than overriding it.
"""
import pytest

from src.agent.cash_policy import DEFAULT_MAX_MULTIPLIER, CashDeploymentPolicy

CFG = {'cash_policy': {
    'enabled': True, 'target_deployment': 0.80, 'min_cash_buffer': 0.10,
    'min_confidence_to_boost': 0.55, 'max_risk_multiplier': 2.5,
}}


def test_no_boost_when_fully_deployed():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_multiplier(deployed_pct=0.85, confidence=0.9) == 1.0


def test_boost_scales_with_gap_and_conviction():
    p = CashDeploymentPolicy(CFG)
    strong = p.risk_multiplier(deployed_pct=0.10, confidence=0.9)
    assert 1.0 < strong <= 2.5
    # Same spare cash, weak conviction: idle cash goes to better ideas, not this one.
    assert p.risk_multiplier(deployed_pct=0.10, confidence=0.4) == 1.0
    # Less spare cash, same conviction: smaller boost.
    assert p.risk_multiplier(deployed_pct=0.60, confidence=0.9) < strong


def test_never_exceeds_the_ceiling():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_multiplier(deployed_pct=0.0, confidence=1.0) <= 2.5


def test_never_shrinks_risk_below_the_configured_budget():
    """Cutting risk is the stop distance's job, not this one's."""
    p = CashDeploymentPolicy(CFG)
    for deployed in (0.0, 0.4, 0.8, 1.0, 1.5):
        for conf in (0.0, 0.3, 0.55, 1.0):
            assert p.risk_multiplier(deployed, conf) >= 1.0


def test_disabled_policy_is_a_no_op():
    cfg = {'cash_policy': dict(CFG['cash_policy'], enabled=False)}
    assert CashDeploymentPolicy(cfg).risk_multiplier(0.0, 1.0) == 1.0


# ------------------------------------------------------- config migration

def test_legacy_notional_caps_carry_their_ratio_over():
    """An old config keeps the operator's chosen aggressiveness.

    Silently resetting to a default would change live position sizes without
    anyone asking for it.
    """
    legacy = {'cash_policy': {
        'enabled': True, 'target_deployment': 0.80,
        'min_confidence_to_boost': 0.55,
        'base_risk_per_trade': 0.02, 'max_risk_per_trade': 0.05,
    }}
    p = CashDeploymentPolicy(legacy)
    assert p.max_multiplier == pytest.approx(2.5)


def test_explicit_setting_wins_over_legacy_keys():
    both = {'cash_policy': {
        'enabled': True, 'max_risk_multiplier': 1.5,
        'base_risk_per_trade': 0.02, 'max_risk_per_trade': 0.05,
    }}
    assert CashDeploymentPolicy(both).max_multiplier == pytest.approx(1.5)


def test_absent_config_falls_back_to_the_default():
    assert CashDeploymentPolicy({}).max_multiplier == DEFAULT_MAX_MULTIPLIER


def test_the_old_notional_cap_method_is_gone():
    """Anything still calling risk_cap() would be sizing the old way."""
    assert not hasattr(CashDeploymentPolicy(CFG), 'risk_cap')
