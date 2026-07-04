"""Tests for the decision journal — especially change-detection, the
feature that keeps an idle symbol from writing 390 rows a day."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.decision_journal import DecisionJournal


def make():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return DecisionJournal(db_path=path, heartbeat_cycles=30), path


def dec(symbol='AAPL', cycle=0, action='hold', skip='hold', conf=0.0,
        executed=False, **kw):
    return {'symbol': symbol, 'cycle': cycle, 'action': action,
            'skip_reason': skip, 'ensemble_confidence': conf,
            'executed': executed, **kw}


def test_first_record_always_writes():
    j, p = make()
    try:
        assert j.record(dec(cycle=0)) is True
        assert len(j.recent('AAPL')) == 1
    finally:
        j.close(); os.unlink(p)


def test_unchanged_state_is_suppressed():
    j, p = make()
    try:
        j.record(dec(cycle=0, conf=0.05))
        # same coarse state next cycle -> not written
        assert j.record(dec(cycle=1, conf=0.09)) is False   # 0.05 and 0.09 both round to 0.1
        assert len(j.recent('AAPL')) == 1
    finally:
        j.close(); os.unlink(p)


def test_action_change_writes():
    j, p = make()
    try:
        j.record(dec(cycle=0, action='hold', skip='hold'))
        assert j.record(dec(cycle=1, action='buy', skip='below_confidence', conf=0.05)) is True
        assert len(j.recent('AAPL')) == 2
    finally:
        j.close(); os.unlink(p)


def test_skip_reason_change_writes():
    j, p = make()
    try:
        j.record(dec(cycle=0, action='buy', skip='pdt_guard', conf=0.5))
        assert j.record(dec(cycle=1, action='buy', skip='min_notional', conf=0.5)) is True
    finally:
        j.close(); os.unlink(p)


def test_executed_always_writes_even_if_unchanged():
    j, p = make()
    try:
        j.record(dec(cycle=0, action='buy', skip=None, conf=0.5, executed=True))
        # identical-looking executed decision still writes (every fill matters)
        assert j.record(dec(cycle=1, action='buy', skip=None, conf=0.5, executed=True)) is True
        assert len(j.recent('AAPL')) == 2
    finally:
        j.close(); os.unlink(p)


def test_heartbeat_writes_after_interval():
    j, p = make()
    try:
        j.record(dec(cycle=0, conf=0.0))
        assert j.record(dec(cycle=5, conf=0.0)) is False    # within heartbeat, unchanged
        assert j.record(dec(cycle=30, conf=0.0)) is True    # heartbeat elapsed
    finally:
        j.close(); os.unlink(p)


def test_recent_deserializes_json_and_filters_symbol():
    j, p = make()
    try:
        j.record(dec('AAPL', cycle=0, action='buy', skip=None, conf=0.6, executed=True,
                     per_strategy={'momentum': {'action': 'buy', 'confidence': 0.6}},
                     llm_verdict={'action': 'buy', 'reasoning': 'trend intact'}))
        j.record(dec('MSFT', cycle=0))
        aapl = j.recent('AAPL')
        assert len(aapl) == 1
        assert aapl[0]['per_strategy']['momentum']['confidence'] == 0.6
        assert aapl[0]['llm_verdict']['reasoning'] == 'trend intact'
        assert aapl[0]['executed'] is True
        assert len(j.recent()) == 2  # both symbols
    finally:
        j.close(); os.unlink(p)
