"""Guardrails on everything the agent changes about itself (guardrails.py):
the AI trade review may confirm, weaken or veto but never reverse or
strengthen a signal, and the adaptive goals layer may only make the agent
more careful."""
from src.agent import guardrails as g
from src.agent.adaptive_agent import Goal, GoalType, SelfAdaptiveAgent, AdaptationAction, MIN_SIZE_MULTIPLIER
from src.agent.adaptive_integration import AdaptiveStrategyIntegration

SIGNAL = {'symbol': 'AAPL', 'action': 'buy', 'confidence': 0.6, 'position_size': 0.04, 'price': 227.5,
          'per_strategy': {'momentum': {'action': 'buy', 'confidence': 0.6}}}


def test_the_ai_may_confirm_weaken_or_veto():
    confirmed = g.bound_verdict(SIGNAL, {'action': 'buy', 'confidence': 0.55, 'reasoning': 'fine'})
    assert confirmed['action'] == 'buy' and confirmed['confidence'] == 0.55
    assert confirmed['price'] == 227.5 and confirmed['per_strategy'] == SIGNAL['per_strategy']
    assert 'guardrail' not in confirmed
    vetoed = g.bound_verdict(SIGNAL, {'action': 'hold', 'confidence': 0.2, 'reasoning': 'earnings tomorrow'})
    assert vetoed['action'] == 'hold'


def test_the_ai_may_not_reverse_a_signal():
    out = g.bound_verdict(SIGNAL, {'action': 'sell', 'confidence': 0.9, 'reasoning': 'bearish'})
    assert out['action'] == 'hold'
    assert 'reversed a buy into a sell' in out['guardrail'] and '[Guardrail:' in out['reasoning']
    exit_signal = {**SIGNAL, 'action': 'sell'}
    assert g.bound_verdict(exit_signal, {'action': 'buy', 'confidence': 0.9})['action'] == 'hold'


def test_the_ai_may_not_raise_confidence_or_size():
    out = g.bound_verdict(SIGNAL, {'action': 'buy', 'confidence': 0.95, 'position_size': 0.5})
    assert out['confidence'] == 0.6 and out['position_size'] == 0.04
    assert 'raised confidence from 0.60 to 0.95' in out['guardrail']


def test_an_unreadable_verdict_keeps_the_signal():
    assert g.bound_verdict(SIGNAL, None) is SIGNAL
    assert g.bound_verdict(SIGNAL, SIGNAL) is SIGNAL                       # an AI outage returns the signal
    out = g.bound_verdict(SIGNAL, {'action': 'maybe', 'confidence': 'high'})
    assert out['action'] == 'buy' and out['confidence'] == 0.6


def test_validate_trade_bounds_fresh_and_cached_verdicts(monkeypatch):
    from src.agent import llm_orchestrator as lo
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    for key in ('ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY', 'GROQ_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    llm = lo.LLMOrchestrator({'primary_llm_provider': 'gemini'})
    calls = []

    def reply(system, user, fallback, model_override=None):
        calls.append(1)
        return {'action': 'sell', 'confidence': 0.99, 'reasoning': 'flip it', 'strategy': 'llm_orchestrated'}
    monkeypatch.setattr(llm, '_call_gemini', reply)
    assert llm.validate_trade('AAPL', dict(SIGNAL), {'close': 227.5})['action'] == 'hold'
    assert llm.validate_trade('AAPL', dict(SIGNAL), {'close': 227.6})['action'] == 'hold'
    assert len(calls) == 1                                                 # the second came from the cache


def test_the_step_and_evidence_helpers():
    assert g.step_toward(1.0, 5.0, 0.0, 2.0, 0.5) == 1.5
    assert g.step_toward(1.0, -3.0, 0.0, 2.0, 0.5) == 0.5
    assert g.enough_evidence(10, 10) and not g.enough_evidence(9, 10) and not g.enough_evidence(None, 1)


def _integration(regime='normal'):
    cfg = {'risk_limits': {'max_position_size': 0.05}, 'risk_tolerance': 0.02}
    ai = AdaptiveStrategyIntegration(cfg)
    ai.adaptive_agent.market_regime = regime
    return ai


def test_falling_behind_the_profit_goal_makes_nothing_more_aggressive():
    ai = _integration()
    behind = Goal(GoalType.PROFIT_TARGET, target_value=0.15, current_value=0.0, priority=1.0)
    assert ai._create_adaptation_for_goal(behind) is None
    ai.goal_manager.active_goals = [behind]
    ai._generate_strategy_adaptations()
    params = ai.get_strategy_parameters('technical')
    assert 'momentum_weight' not in params and 'position_size' not in params


def test_adaptive_sizes_never_exceed_the_position_cap_or_grow_in_calm_markets():
    calm, normal = _integration('low_volatility'), _integration('normal')
    assert calm.get_position_size('AAPL', 1.0, 100000) == normal.get_position_size('AAPL', 1.0, 100000)
    ai = _integration()
    ai.adaptive_agent.config['risk_tolerance'] = 0.5                       # far above sensible
    assert ai.get_position_size('AAPL', 1.0, 100000) == 5000.0             # capped at 5%


def test_the_adaptive_agent_only_cuts_size_down_to_a_floor():
    cfg = {}
    agent = SelfAdaptiveAgent(cfg)
    cut = AdaptationAction(action_type='reduce_position_size', parameters={'size_multiplier': 0.7},
                           expected_impact=0.4, confidence=0.8)
    for _ in range(10):
        agent._execute_adaptation(cut)
    assert cfg['position_size_multiplier'] == MIN_SIZE_MULTIPLIER
    agent.market_regime = 'high_volatility'
    assert all(a.action_type != 'switch_strategy_weights' for a in agent._generate_adaptation_actions())
    assert 'strategy_weights' not in cfg


# ---- the evidence tilt: how a strategy's own results scale its vote

def test_no_tilt_until_a_strategy_has_a_record():
    assert g.evidence_tilt(0, 5.0) == 1.0 and g.evidence_tilt(4, 5.0) == 1.0     # too few trades
    assert g.evidence_tilt(None, None) == 1.0 and g.evidence_tilt('x', 'y') == 1.0
    assert g.evidence_tilt(10, float('nan')) == 1.0


def test_the_tilt_is_symmetric_and_bounded_by_how_convincing_the_record_is():
    assert g.evidence_tilt(20, 0.0) == 1.0                                       # no edge: neutral
    assert g.evidence_tilt(20, 1.0) == 1.25 and g.evidence_tilt(20, -1.0) == 0.75
    assert g.evidence_tilt(20, 2.0) == 1.5 and g.evidence_tilt(20, 9.0) == 1.5   # t of 2: the most it can earn
    assert g.evidence_tilt(20, -2.0) == 0.5 and g.evidence_tilt(20, -9.0) == 0.5  # halved, never switched off


def _ensemble(perf):
    from src.agent.strategy_manager import StrategyManager
    m = StrategyManager({'ensemble_method': 'adaptive_confidence', 'min_trade_confidence': 0.0,
                         'regime_filter': {'enabled': False}})
    m.strategy_weights = {'a': 0.5, 'b': 0.5}
    m.strategy_performance = perf
    # a says buy with 0.60, b says sell with 0.65: with no record, b wins narrowly.
    sigs = {'a': {'action': 'buy', 'confidence': 0.60, 'position_size': 0.0},
            'b': {'action': 'sell', 'confidence': 0.65, 'position_size': 0.0}}
    return m._adaptive_confidence_ensemble(sigs, 'X', {'close': 100, 'open': 100})['action']


def test_two_lucky_trades_no_longer_double_a_strategys_weight():
    # Before: a's weight was x2.0 on a sharpe of 3 and a 100% win rate over two
    # trades, which overturned b. Now a record needs five trades to count at all.
    lucky_two = {'a': {'closed_trades': 2, 'sharpe_ratio': 3.0, 'win_rate': 1.0}, 'b': {}}
    assert _ensemble({'a': {}, 'b': {}}) == 'sell'
    assert _ensemble(lucky_two) == 'sell'


def test_a_real_record_does_tilt_the_vote():
    proven = {'a': {'closed_trades': 30, 'sharpe_ratio': 2.5, 'win_rate': 0.7}, 'b': {}}
    assert _ensemble(proven) == 'buy'                                            # a x1.5 beats b


def test_a_significantly_losing_strategy_is_now_reduced():
    # Before, a losing record was simply ignored. b is the loser here: it is halved.
    losing_b = {'a': {}, 'b': {'closed_trades': 30, 'sharpe_ratio': -2.5, 'win_rate': 0.3}}
    assert _ensemble(losing_b) == 'buy'
