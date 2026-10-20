"""The self-improvement loop must actually change trading behaviour.

Before this change `_apply_config_change` wrote `config['strategy_weights'][name]`.
Nothing in the execution path reads that key: StrategyManager loads its
weights from `config['strategies'][name]['weight']` at init and normalises
them into its own `strategy_weights` attribute. The only other reader was
the retrospective prompt itself (self_assessment.py:135), so the daily
"self-improvement" cycle was an echo chamber — it fed its own past writes
back to the LLM and never touched a live weight.

It also wrote in-memory only, so anything it did decide evaporated on the
next restart, and `execution.min_order_size` / `max_order_size` were
auto-appliable with no numeric clamp at all — an LLM proposing
`min_order_size: 100000` would have silently halted trading.
"""

import json
import os
import tempfile

import pytest
from unittest.mock import MagicMock

from src.agent.self_assessment import SelfAssessmentEngine
from src.agent.strategy_manager import StrategyManager


def base_config():
    return {
        'strategies': {
            'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0},
            'mean_reversion': {'type': 'technical', 'enabled': True, 'weight': 1.0},
            'rsi_strategy': {'type': 'technical', 'enabled': True, 'weight': 0.8},
        },
        'risk_limits': {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.03},
        'execution': {'min_order_size': 100, 'max_order_size': 10000},
    }


@pytest.fixture
def engine_rig():
    fd, db = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    fd, overlay = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(overlay)          # must not exist yet

    cfg = base_config()
    mgr = StrategyManager(cfg)
    engine = SelfAssessmentEngine(MagicMock(), MagicMock(), cfg, db_path=db,
                                  strategy_manager=mgr, overlay_path=overlay)
    yield engine, cfg, mgr, overlay
    engine.close()
    os.unlink(db)
    if os.path.exists(overlay):
        os.unlink(overlay)


def plan(target, proposed, current=1.0):
    return {'assessment_id': 1, 'parameter_adjustments': [
        {'target': target, 'current': current, 'proposed': proposed,
         'reasoning': 'test'}]}


# -------------------------------------------------- the loop actually closes

def test_weight_change_reaches_the_config_key_strategy_manager_reads(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 1.3),
                              require_approval=False)
    assert cfg['strategies']['momentum']['weight'] == pytest.approx(1.3)


def test_weight_change_reaches_the_live_strategy_manager(engine_rig):
    """Config alone is not enough — StrategyManager copied the weights into
    its own dict at init and never re-reads the config."""
    engine, cfg, mgr, _ = engine_rig
    before = mgr.strategy_weights['momentum']
    engine.apply_improvements(plan('strategy_weights.momentum', 1.3),
                              require_approval=False)
    assert mgr.strategy_weights['momentum'] != before


def test_live_weights_stay_normalised(engine_rig):
    """StrategyManager normalises at init; an applied change must preserve
    that invariant or the ensemble silently rescales every confidence."""
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 1.5),
                              require_approval=False)
    assert sum(mgr.strategy_weights.values()) == pytest.approx(1.0)


def test_relative_ordering_follows_the_proposal(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 2.0),
                              require_approval=False)
    assert mgr.strategy_weights['momentum'] > mgr.strategy_weights['mean_reversion']


def test_unknown_strategy_name_is_rejected_not_invented(engine_rig):
    """An LLM hallucinating a strategy must not create a phantom entry."""
    engine, cfg, mgr, _ = engine_rig
    applied, escalated = engine.apply_improvements(
        plan('strategy_weights.does_not_exist', 1.2), require_approval=False)
    assert applied == []
    assert 'does_not_exist' not in mgr.strategy_weights
    assert 'does_not_exist' not in cfg['strategies']


# --------------------------------------------------------------- persistence

def test_applied_change_is_persisted_to_the_overlay(engine_rig):
    engine, cfg, mgr, overlay = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 1.3),
                              require_approval=False)
    assert os.path.exists(overlay)
    saved = json.loads(open(overlay, encoding='utf-8').read())
    assert saved['strategies.momentum.weight'] == pytest.approx(1.3)


def test_overlay_reapplies_on_restart(engine_rig):
    engine, cfg, mgr, overlay = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 1.3),
                              require_approval=False)

    from src.agent.self_assessment import apply_overlay
    fresh = base_config()
    apply_overlay(fresh, overlay)
    assert fresh['strategies']['momentum']['weight'] == pytest.approx(1.3)


def test_missing_overlay_is_not_an_error():
    from src.agent.self_assessment import apply_overlay
    cfg = base_config()
    apply_overlay(cfg, '/nonexistent/path/overlay.json')
    assert cfg['strategies']['momentum']['weight'] == 1.0


def test_corrupt_overlay_is_ignored_rather_than_crashing_startup(tmp_path):
    from src.agent.self_assessment import apply_overlay
    bad = tmp_path / 'overlay.json'
    bad.write_text('{not json', encoding='utf-8')
    cfg = base_config()
    apply_overlay(cfg, str(bad))
    assert cfg['strategies']['momentum']['weight'] == 1.0


# ------------------------------------------------------------------- clamps

def test_weight_is_still_clamped_to_its_existing_bounds(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 999.0),
                              require_approval=False)
    assert cfg['strategies']['momentum']['weight'] <= 2.0


def test_negative_weight_is_clamped_to_zero(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', -5.0),
                              require_approval=False)
    assert cfg['strategies']['momentum']['weight'] >= 0.0


def test_min_order_size_cannot_be_raised_into_a_trading_halt(engine_rig):
    """Unclamped before: `min_order_size: 100000` would auto-apply and stop
    every order from ever meeting the minimum."""
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('execution.min_order_size', 100_000, current=100),
                              require_approval=False)
    assert cfg['execution']['min_order_size'] <= 1_000


def test_max_order_size_cannot_be_raised_without_bound(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('execution.max_order_size', 10_000_000, current=10_000),
                              require_approval=False)
    assert cfg['execution']['max_order_size'] <= 1_000_000


def test_min_order_size_cannot_exceed_max_order_size(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('execution.min_order_size', 900_000, current=100),
                              require_approval=False)
    assert cfg['execution']['min_order_size'] <= cfg['execution']['max_order_size']


def test_non_numeric_proposal_is_rejected(engine_rig):
    """The LLM returns free-form JSON; a string where a float belongs used to
    be written straight into the live config."""
    engine, cfg, mgr, _ = engine_rig
    applied, _ = engine.apply_improvements(
        plan('strategy_weights.momentum', 'aggressive'), require_approval=False)
    assert applied == []
    assert isinstance(cfg['strategies']['momentum']['weight'], (int, float))


def test_rejected_proposal_is_not_recorded_as_applied(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    engine.apply_improvements(plan('strategy_weights.momentum', 'aggressive'),
                              require_approval=False)
    rows = engine._conn.execute(
        "SELECT count(*) FROM improvements WHERE auto_applied = 1").fetchone()
    assert rows[0] == 0


# --------------------------------------------------------------- back-compat

def test_unsafe_targets_are_still_escalated_not_applied(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    applied, escalated = engine.apply_improvements(
        plan('risk_limits.max_position_size', 0.5), require_approval=False)
    assert applied == []
    assert len(escalated) == 1


def test_require_approval_still_escalates_everything(engine_rig):
    engine, cfg, mgr, _ = engine_rig
    applied, escalated = engine.apply_improvements(
        plan('strategy_weights.momentum', 1.3), require_approval=True)
    assert applied == []
    assert len(escalated) == 1
    assert cfg['strategies']['momentum']['weight'] == 1.0


def test_engine_works_without_a_strategy_manager():
    """main.py may construct the engine before strategies exist; it must
    degrade to config-only rather than raising."""
    fd, db = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    cfg = base_config()
    try:
        engine = SelfAssessmentEngine(MagicMock(), MagicMock(), cfg, db_path=db)
        engine.apply_improvements(plan('strategy_weights.momentum', 1.3),
                                  require_approval=False)
        assert cfg['strategies']['momentum']['weight'] == pytest.approx(1.3)
        engine.close()
    finally:
        os.unlink(db)
