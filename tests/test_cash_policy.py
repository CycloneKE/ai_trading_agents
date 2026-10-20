from src.agent.cash_policy import CashDeploymentPolicy

CFG = {'cash_policy': {
    'enabled': True, 'target_deployment': 0.80, 'min_cash_buffer': 0.10,
    'base_risk_per_trade': 0.02, 'max_risk_per_trade': 0.05,
    'min_confidence_to_boost': 0.55,
}}


def test_base_cap_when_fully_deployed():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_cap(deployed_pct=0.85, confidence=0.9) == 0.02


def test_boost_scales_with_gap_and_conviction():
    p = CashDeploymentPolicy(CFG)
    # 10% deployed, high conviction -> boosted toward the 5% ceiling
    cap = p.risk_cap(deployed_pct=0.10, confidence=0.9)
    assert 0.02 < cap <= 0.05
    # same gap, weak conviction -> no boost
    assert p.risk_cap(deployed_pct=0.10, confidence=0.4) == 0.02


def test_never_exceeds_hard_ceiling():
    p = CashDeploymentPolicy(CFG)
    assert p.risk_cap(deployed_pct=0.0, confidence=1.0) <= 0.05


def test_disabled_policy_returns_base():
    cfg = {'cash_policy': dict(CFG['cash_policy'], enabled=False)}
    assert CashDeploymentPolicy(cfg).risk_cap(0.0, 1.0) == 0.02
