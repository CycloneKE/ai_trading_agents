from src.api.agent_focus import build_agent_focus


def test_lanes_partition_decisions():
    positions = [{'symbol': 'SCOM', 'quantity': 100, 'unrealized_pl_pct': 2.1}]
    decisions = [
        {'ts': '2026-07-10T09:40:00', 'symbol': 'EQTY', 'action': 'hold',
         'executed': False, 'skip_reason': 'low_confidence'},
        {'ts': '2026-07-10T09:40:01', 'symbol': 'KCB', 'action': 'buy',
         'executed': True, 'price': 41.5},
    ]
    focus = build_agent_focus(positions, decisions, cash=60000, equity=100000)
    assert focus['holding'] == [{'symbol': 'SCOM', 'quantity': 100, 'unrealized_pl_pct': 2.1}]
    assert focus['reviewing'][0]['symbol'] == 'EQTY'
    assert focus['reviewing'][0]['reason'] == 'low_confidence'
    assert focus['traded'][0]['symbol'] == 'KCB'
    assert focus['cash']['deployed_pct'] == 40.0


def test_empty_inputs():
    focus = build_agent_focus([], [], cash=0, equity=0)
    assert focus == {'holding': [], 'reviewing': [], 'traded': [],
                     'cash': {'cash': 0, 'equity': 0, 'deployed_pct': 0.0}}
