from src.api.activity_format import format_agent_activity


def test_executed_decision_maps_to_full_activity_record():
    decisions = [{
        'ts': '2026-07-10T09:31:05', 'symbol': 'SCOM', 'action': 'buy',
        'price': 34.2, 'quantity': 100, 'executed': True, 'skip_reason': None,
    }]
    out = format_agent_activity(decisions)
    assert len(out) == 1
    rec = out[0]
    assert rec['id'] == '2026-07-10T09:31:05-SCOM'
    assert rec['time'] == '09:31:05'
    assert rec['type'] == 'BUY'
    assert rec['symbol'] == 'SCOM'
    assert rec['price'] == 34.2
    assert rec['quantity'] == 100
    assert rec['reason'] == 'Executed'
    assert rec['message'] == 'BUY executed at 34.2'
    # legacy keys still present for the System-tab Agent Log
    assert rec['timestamp'] == '2026-07-10T09:31:05'
    assert rec['component'] == 'SCOM'


def test_blocked_decision_uses_skip_reason():
    decisions = [{'ts': '2026-07-10T10:00:00', 'symbol': 'EQTY', 'action': 'buy',
                  'price': None, 'executed': False, 'skip_reason': 'risk_limit'}]
    rec = format_agent_activity(decisions)[0]
    assert rec['type'] == 'BUY'
    assert rec['reason'] == 'Blocked: risk_limit'
    assert rec['message'] == 'BUY blocked: risk_limit'


def test_missing_fields_do_not_crash():
    rec = format_agent_activity([{}])[0]
    assert rec['type'] == 'HOLD'
    assert rec['time'] == ''
