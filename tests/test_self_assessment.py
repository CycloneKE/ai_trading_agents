"""
Unit tests for self_assessment.py.
"""
import pytest
import os
import tempfile
from unittest.mock import MagicMock
from src.agent.self_assessment import SelfAssessmentEngine


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_escalation_manager():
    return MagicMock()


@pytest.fixture
def mock_llm_orchestrator():
    llm = MagicMock()
    llm.enabled = True
    return llm


@pytest.fixture
def sample_config():
    return {
        "strategy_weights": {
            "momentum": 1.0,
            "mean_reversion": 1.0,
            "rsi_strategy": 0.8
        },
        "risk_limits": {
            "stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.03
        },
        "execution": {
            "min_order_size": 100
        }
    }


def test_self_assessment_init(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db):
    """Verify engine initializes and sets up SQLite tables."""
    engine = SelfAssessmentEngine(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db)
    assert os.path.exists(temp_db)
    
    # Check if table schema is queryable
    cur = engine._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    assert "assessments" in tables
    assert "improvements" in tables
    engine.close()


def test_apply_improvements_safe(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db):
    """Verify safe improvements (weights/min order size) are auto-applied."""
    engine = SelfAssessmentEngine(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db)
    
    plan = {
        "assessment_id": 1,
        "parameter_adjustments": [
            {
                "target": "strategy_weights.momentum",
                "current": 1.0,
                "proposed": 1.3,
                "reasoning": "Performance tilt"
            },
            {
                "target": "execution.min_order_size",
                "current": 100,
                "proposed": 120,
                "reasoning": "Avoid tiny lot fills"
            }
        ]
    }
    
    auto_applied, escalated = engine.apply_improvements(plan, require_approval=False)
    
    assert len(auto_applied) == 2
    assert len(escalated) == 0
    
    # Verify values actually modified in self.config
    assert engine.config["strategy_weights"]["momentum"] == 1.3
    assert engine.config["execution"]["min_order_size"] == 120
    
    # Verify logged to improvements table
    cur = engine._conn.execute("SELECT target, new_value, auto_applied FROM improvements")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "strategy_weights.momentum"
    assert rows[0][1] == "1.3"
    assert rows[0][2] == 1 # auto_applied
    
    engine.close()


def test_apply_improvements_escalate_unsafe(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db):
    """Verify risky improvements (stop loss/symbols) trigger operator escalation."""
    engine = SelfAssessmentEngine(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db)
    
    plan = {
        "assessment_id": 1,
        "parameter_adjustments": [
            {
                "target": "risk_limits.stop_loss_pct",
                "current": 0.05,
                "proposed": 0.08,
                "reasoning": "Market widening"
            }
        ],
        "symbol_actions": [
            {
                "symbol": "AAPL",
                "action": "remove",
                "reasoning": "Unprofitable symbol"
            }
        ]
    }
    
    auto_applied, escalated = engine.apply_improvements(plan, require_approval=False)
    
    assert len(auto_applied) == 0
    assert len(escalated) == 2
    
    # Config values should be unchanged
    assert engine.config["risk_limits"]["stop_loss_pct"] == 0.05
    
    # Escalation manager should be called twice
    assert mock_escalation_manager.create_escalation.call_count == 2
    
    engine.close()


def test_analyze_prediction_accuracy(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db):
    """Verify prediction accuracy computes correct percentage values."""
    engine = SelfAssessmentEngine(mock_llm_orchestrator, mock_escalation_manager, sample_config, temp_db)
    
    decisions = [
        # Buy that went up (correct)
        {"symbol": "SCOM", "action": "buy", "price": 30.0, "ts": "2026-07-09T10:00:00", "executed": 1, "skip_reason": None},
        {"symbol": "SCOM", "action": "hold", "price": 31.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None},
        # Buy that went down (incorrect)
        {"symbol": "KCB", "action": "buy", "price": 80.0, "ts": "2026-07-09T10:00:00", "executed": 1, "skip_reason": None},
        {"symbol": "KCB", "action": "hold", "price": 79.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None},
        # Veto that saved from buy drop (correct veto)
        {"symbol": "EQTY", "action": "buy", "price": 50.0, "ts": "2026-07-09T10:00:00", "executed": 0, "skip_reason": "llm_veto", "per_strategy_json": "{'momentum': {'action': 'buy'}}"},
        {"symbol": "EQTY", "action": "hold", "price": 48.0, "ts": "2026-07-09T10:01:00", "executed": 0, "skip_reason": None}
    ]
    
    metrics = engine._analyze_prediction_accuracy(decisions, [])
    
    assert metrics["buy_signals_count"] == 2
    assert metrics["buy_accuracy_pct"] == 50.0 # 1 out of 2 correct
    assert metrics["llm_vetos_count"] == 1
    assert metrics["veto_accuracy_pct"] == 100.0 # 1 out of 1 correct veto
    
    engine.close()
