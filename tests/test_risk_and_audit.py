"""
Test suite for Pillar 3: Risk Management, Cryptographic Audit Journal, Persistent Kill Switch, and Task Queue Engine.
"""
import pytest
import os
import time
from src.agent.audit_journal import AuditJournal
from src.agent.realtime_risk_manager import RealTimeRiskManager
from src.agent.task_queue import TaskQueueEngine

def test_audit_journal_crypto_hash_chain(tmp_path):
    """Test cryptographic SHA-256 hash chaining and audit trail verification."""
    db_file = str(tmp_path / "test_audit.db")
    journal = AuditJournal(db_path=db_file)
    
    # Log 3 events
    e1 = journal.log_event("STRATEGY", "Signal Generated", {"symbol": "AAPL", "action": "buy"})
    e2 = journal.log_event("RISK_CHECK", "Pre-trade risk check passed", {"symbol": "AAPL", "risk": "low"})
    e3 = journal.log_event("EXECUTION", "Order Executed", {"symbol": "AAPL", "qty": 10, "price": 180.0})
    
    assert e1 == 1
    assert e2 == 2
    assert e3 == 3
    
    # Verify hash chain integrity
    is_valid, count, msg = journal.verify_chain()
    assert is_valid is True
    assert count == 3
    assert "verified successfully" in msg.lower()

def test_persistent_kill_switch_sqlite(tmp_path):
    """Test SQLite persistent emergency stop kill-switch state."""
    db_file = str(tmp_path / "test_risk_state.db")
    
    # Initialize Risk Manager with custom DB
    rm1 = RealTimeRiskManager({}, database_manager=None)
    rm1.risk_state_db = db_file
    rm1._init_persistent_risk_state()
    
    # Enable Kill Switch
    rm1.set_persistent_kill_switch(True, reason="Test Emergency Stop Triggered")
    assert rm1.emergency_stop is True
    
    # Initialize a new Risk Manager instance connected to the same DB
    rm2 = RealTimeRiskManager({}, database_manager=None)
    rm2.risk_state_db = db_file
    rm2._init_persistent_risk_state()
    
    # Assert emergency stop is automatically restored from DB
    assert rm2.emergency_stop is True
    
    # Reset emergency stop
    res = rm2.reset_emergency_stop()
    assert res is True
    assert rm2.emergency_stop is False
    
    # Verify state in fresh 3rd instance
    rm3 = RealTimeRiskManager({}, database_manager=None)
    rm3.risk_state_db = db_file
    rm3._init_persistent_risk_state()
    assert rm3.emergency_stop is False

def test_task_queue_engine_async_execution():
    """Test decoupled Task Queue Engine background execution."""
    tq = TaskQueueEngine(num_workers=2)
    tq.start()
    
    # Submit background task
    def dummy_task(a, b):
        return a * b + 10
        
    task_id = tq.submit("dummy_task", dummy_task, 5, 6)
    
    # Wait for completion
    completed = False
    for _ in range(20):
        status = tq.get_task_status(task_id)
        if status and status.get("status") == "completed":
            completed = True
            assert status.get("result") == 40
            break
        time.sleep(0.1)
        
    assert completed is True
    tq.stop()
