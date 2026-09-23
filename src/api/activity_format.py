"""Shapes decision-journal records for the dashboard activity feeds.

One record serves two consumers: the Dashboard's AgentActivity panel
(type/symbol/price/reason) and the System tab's Agent Log
(timestamp/component/message).
"""
from typing import Any, Dict, List


def format_agent_activity(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in decisions:
        ts = d.get('ts') or ''
        action = (d.get('action') or 'hold').upper()
        if d.get('executed'):
            message = f"{action} executed at {d.get('price')}"
            reason = 'Executed'
        elif d.get('skip_reason') == 'hold':
            # 'hold' is the default every decision starts with: the
            # strategies found nothing to do. It is not a block, and
            # labelling it "Blocked: hold" read as though a gate were
            # stopping trades. Matches SymbolDrilldown's "No directional
            # signal".
            message = f"{action}: no signal"
            reason = 'No signal'
        elif d.get('skip_reason'):
            message = f"{action} blocked: {d.get('skip_reason')}"
            reason = f"Blocked: {d.get('skip_reason')}"
        else:
            message = action
            reason = 'Evaluated'
        out.append({
            'id': f"{ts}-{d.get('symbol')}",
            'timestamp': ts,
            'time': ts[11:19] if len(ts) >= 19 else '',
            'component': d.get('symbol'),
            'message': message,
            'type': action,
            'symbol': d.get('symbol'),
            'quantity': d.get('quantity'),
            'price': d.get('price'),
            'reason': reason,
        })
    return out
